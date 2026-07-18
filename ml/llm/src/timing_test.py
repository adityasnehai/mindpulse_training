"""Precisely time N real forward+backward micro-batches (not wall-clocking a
whole Trainer.train() call, which includes one-time setup cost) to get a
real, non-guessed per-step estimate for the full training run."""

import os
import sys
import time
from pathlib import Path

_N_CORES = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _N_CORES - 2)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _N_CORES - 2)))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

torch.set_num_threads(max(1, _N_CORES - 2))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import TokenizedSFTDataset, load_config, load_records, verify_lora_target_modules  # noqa: E402


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    sft_path = processed_dir / "mindpulse_sft.jsonl"

    print(f"CPU cores available: {_N_CORES}, torch threads: {torch.get_num_threads()}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16)
    print(f"Model load: {time.time() - t0:.1f}s")

    matched = verify_lora_target_modules(model, train_cfg["candidate_lora_target_modules"])
    lora_config = LoraConfig(r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
                              lora_dropout=train_cfg["lora_dropout"], target_modules=matched, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    train_records = load_records(sft_path, "train")[:6]
    dataset = TokenizedSFTDataset(train_records, tokenizer, train_cfg["max_sequence_length"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)

    for gc_enabled in [True, False]:
        model.gradient_checkpointing_enable() if gc_enabled else model.gradient_checkpointing_disable()
        model.config.use_cache = not gc_enabled
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])

        seq_lengths = []
        step_times = []
        for i in range(3):
            batch = collator([dataset[i]])
            seq_lengths.append(batch["input_ids"].shape[1])
            t_step = time.time()
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            step_times.append(time.time() - t_step)

        label = "gradient_checkpointing=True" if gc_enabled else "gradient_checkpointing=False"
        print(f"\n[{label}]")
        print(f"  sequence lengths: {seq_lengths}")
        print(f"  per-microbatch times (s): {[f'{t:.2f}' for t in step_times]}")
        print(f"  mean: {sum(step_times)/len(step_times):.2f}s/microbatch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
