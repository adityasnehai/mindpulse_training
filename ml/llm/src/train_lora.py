"""Milestone 6: LoRA fine-tune of Gemma 3 270M IT.

Pre-flight verifies the configured LoRA target modules actually exist on the
loaded model before training starts (docs/PRODUCT_SPEC.md section 13: "fail
clearly if no target modules match") — confirmed once already by manually
inspecting the real model (all 7 candidate modules present), but this check
runs for real at train time too rather than relying on that one-off check
staying true forever.

Runs on GPU (CUDA) when available. CPU-only fine-tuning was measured directly
on this machine before GPU support was added: a single forward+backward pass
on one 348-token example took over 175 seconds — several orders of magnitude
too slow for practical training (full run would have taken weeks). A real
NVIDIA RTX 4060 (8.6GB VRAM) is present and reachable from WSL2 via CUDA
passthrough (confirmed with nvidia-smi and torch.cuda.is_available()), so
this trains there instead; the CPU thread-tuning code from that earlier
attempt is dropped since it's irrelevant on GPU.
"""

import sys
from pathlib import Path

import torch
import yaml

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script requires the GPU — CPU-only training "
        "of this model was measured at >175s per single-example forward+backward pass, "
        "making full training impractical (see module docstring). Do not silently fall "
        "back to CPU."
    )

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_format import tokenize_and_mask  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_records(path: Path, split: str) -> list[dict]:
    import json

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                records.append(r)
    return records


def verify_lora_target_modules(model, candidate_modules: list[str]) -> list[str]:
    real_module_leaf_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [m for m in candidate_modules if m in real_module_leaf_names]
    if not matched:
        raise RuntimeError(
            f"None of the candidate LoRA target modules {candidate_modules} exist on the "
            f"loaded model. Real module names found: {sorted(real_module_leaf_names)}. "
            "Failing per docs/PRODUCT_SPEC.md section 13's explicit requirement, not "
            "silently proceeding with an empty/wrong target list."
        )
    return matched


class TokenizedSFTDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return tokenize_and_mask(self.records[idx], self.tokenizer, self.max_length)


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sft_path = processed_dir / "mindpulse_sft.jsonl"
    if not sft_path.exists():
        print(f"ERROR: {sft_path} not found. Run build_sft_dataset.py first.")
        return 1

    torch.manual_seed(config["seed"])

    print(f"Loading tokenizer and base model {train_cfg['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False  # not needed during training; saves memory regardless

    matched_modules = verify_lora_target_modules(model, train_cfg["candidate_lora_target_modules"])
    print(f"LoRA target modules verified present on the real model: {matched_modules}")

    lora_config = LoraConfig(
        r=train_cfg["lora_rank"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=matched_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = load_records(sft_path, "train")
    val_records = load_records(sft_path, "validation")
    print(f"Loaded {len(train_records)} train / {len(val_records)} validation records")

    train_dataset = TokenizedSFTDataset(train_records, tokenizer, train_cfg["max_sequence_length"])
    val_dataset = TokenizedSFTDataset(val_records, tokenizer, train_cfg["max_sequence_length"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)

    # Spec's original starting batch size (section 13) — plenty of headroom
    # on the real 8.6GB RTX 4060 for a 270M model with rank-16 LoRA.
    training_args = TrainingArguments(
        output_dir=str(artifacts_dir / "lora_checkpoints"),
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        report_to=[],
        seed=config["seed"],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    adapter_path = artifacts_dir / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Saved LoRA adapter to {adapter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
