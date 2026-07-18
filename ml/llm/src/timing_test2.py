"""Single-microbatch timing at a lower thread count, to test whether the
high thread count (22) is causing oversubscription overhead rather than
speedup on a model this small."""

import os
import sys
import time
from pathlib import Path

_N_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
os.environ["OMP_NUM_THREADS"] = str(_N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(_N_THREADS)

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

torch.set_num_threads(_N_THREADS)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import TokenizedSFTDataset, load_config, load_records, verify_lora_target_modules  # noqa: E402


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    sft_path = processed_dir / "mindpulse_sft.jsonl"

    print(f"threads={_N_THREADS}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16)
    print(f"load={time.time()-t0:.1f}s", flush=True)

    matched = verify_lora_target_modules(model, train_cfg["candidate_lora_target_modules"])
    lora_config = LoraConfig(r=train_cfg["lora_rank"], lora_alpha=train_cfg["lora_alpha"],
                              lora_dropout=train_cfg["lora_dropout"], target_modules=matched, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.train()

    train_records = load_records(sft_path, "train")[:1]
    dataset = TokenizedSFTDataset(train_records, tokenizer, train_cfg["max_sequence_length"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)
    batch = collator([dataset[0]])
    print(f"seq_len={batch['input_ids'].shape[1]}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    t1 = time.time()
    outputs = model(**batch)
    t_fwd = time.time()
    outputs.loss.backward()
    t_bwd = time.time()
    optimizer.step()
    t_opt = time.time()

    print(f"forward={t_fwd-t1:.2f}s backward={t_bwd-t_fwd:.2f}s optstep={t_opt-t_bwd:.2f}s total={t_opt-t1:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
