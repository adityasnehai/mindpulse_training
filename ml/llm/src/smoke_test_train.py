"""Tiny end-to-end smoke test for train_lora.py's pipeline (model load, LoRA
wrap, tokenization, a couple of real training steps) on a handful of real
examples, before committing to the full run. Not a unit test — a deliberate
one-off check of the real, expensive path."""

import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import TokenizedSFTDataset, load_config, load_records, verify_lora_target_modules  # noqa: E402


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    sft_path = processed_dir / "mindpulse_sft.jsonl"

    t0 = time.time()
    print("Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    print(f"Loaded in {time.time() - t0:.1f}s")

    matched = verify_lora_target_modules(model, train_cfg["candidate_lora_target_modules"])
    lora_config = LoraConfig(r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
                              lora_dropout=train_cfg["lora_dropout"], target_modules=matched, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = load_records(sft_path, "train")[:8]
    print(f"Smoke test on {len(train_records)} real training records")
    train_dataset = TokenizedSFTDataset(train_records, tokenizer, train_cfg["max_sequence_length"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)

    args = TrainingArguments(
        output_dir="/tmp/smoke_test_output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=3,
        logging_steps=1,
        bf16=True,
        report_to=[],
        save_strategy="no",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, data_collator=collator)

    t1 = time.time()
    print("Starting 3-step smoke test...")
    trainer.train()
    print(f"Smoke test training completed in {time.time() - t1:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
