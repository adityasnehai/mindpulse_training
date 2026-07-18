"""Milestone 6/7 bridge: merge the trained LoRA adapter into the base model
and save a complete Hugging Face checkpoint, per docs/PRODUCT_SPEC.md
section 13 ("Load the base model. Load the LoRA adapter. Merge adapter
weights into the base model. Save a complete Hugging Face checkpoint.").
This merged checkpoint (not the adapter-only one) is what Milestone 7
converts to .litertlm — LiteRT-LM needs full weights, not a base+adapter pair.
"""

import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()

    adapter_path = artifacts_dir / "lora_adapter"
    if not adapter_path.exists():
        print(f"ERROR: {adapter_path} not found. Run train_lora.py first.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16, device_map=device)

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("Merging adapter weights into base model...")
    merged = model.merge_and_unload()

    merged_path = artifacts_dir / "merged_checkpoint"
    merged_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    print(f"Saved merged checkpoint to {merged_path}")

    # Verify the merged model still produces sane output before trusting it
    # for conversion — a merge bug (e.g. wrong scaling) could silently
    # produce a model that "saves fine" but behaves differently from the
    # adapter+base combination that was actually evaluated in Milestone 6.
    print("Verifying merged model output matches adapter+base output on a real prompt...")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt_format import build_user_turn

    sample_input = {
        "goal": "stress_management", "changes": [{"feature": "stationary_fraction", "direction": "increase",
        "magnitude": "moderate", "duration_days": 4}], "user_context": "work_or_study_pressure",
        "available_time_minutes": 1, "previous_helpful_actions": [], "previous_unhelpful_actions": [],
        "candidate_actions": ["one_minute_breathing"],
    }
    messages = [{"role": "user", "content": build_user_turn(sample_input)}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        adapter_out = model.generate(**encoded, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        merged_out = merged.generate(**encoded, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    adapter_text = tokenizer.decode(adapter_out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    merged_text = tokenizer.decode(merged_out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)

    if adapter_text != merged_text:
        print("ERROR: merged model output does NOT match adapter+base output. Merge may be incorrect.")
        print(f"adapter: {adapter_text}")
        print(f"merged:  {merged_text}")
        return 1

    print(f"Verified match. Sample output: {merged_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
