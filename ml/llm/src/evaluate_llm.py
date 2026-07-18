"""Milestone 6 evaluation: run the fine-tuned model on the real held-out test
split and measure the metrics docs/PRODUCT_SPEC.md section 14 requires:
JSON parse success, schema validity, approved action-ID accuracy, duration
consistency, and zero diagnosis/medication/self-harm language in real
generated output (not just training data — the model's own outputs).
"""

import json
import re
import sys
from pathlib import Path

import torch
import yaml
from jsonschema import ValidationError, validate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_format import build_user_turn  # noqa: E402
from safety_filters import contains_excluded_content  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "data" / "schemas"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(name: str) -> dict:
    with open(SCHEMAS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_records(path: Path, split: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                records.append(r)
    return records


def extract_json_object(text: str) -> dict | None:
    """The model may emit trailing/leading whitespace or (rarely) extra
    tokens; extract the first balanced {...} object rather than assuming
    the raw string is pure JSON."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_batch(model, tokenizer, llm_inputs: list[dict], max_new_tokens: int = 200) -> list[str]:
    # apply_chat_template(tokenize=True, return_tensors="pt") does not
    # reliably return a plain tensor in this transformers version (same root
    # cause as the loss-masking bug found earlier in prompt_format.py: it can
    # return a dict-like BatchEncoding instead) — confirmed by a real crash
    # here (`AttributeError` on `.shape` inside model.generate()). Getting
    # the formatted string first, then tokenizing explicitly, sidesteps the
    # ambiguity entirely.
    #
    # Batched (not one-example-at-a-time): a real run of the unbatched
    # version measured GPU utilization at 25% and took 40+ minutes for 565
    # examples — sequential single-example generation badly underused the
    # GPU. left-padding is required so every sequence's generated portion
    # starts at the same column for a batch.
    tokenizer.padding_side = "left"
    prompt_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_user_turn(llm_input)}], tokenize=False, add_generation_prompt=True
        )
        for llm_input in llm_inputs
    ]
    encoded = tokenizer(prompt_texts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    input_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    generated = output_ids[:, input_len:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]


def evaluate_on_records(
    model, tokenizer, records: list[dict], output_schema: dict, approved_action_ids: set, batch_size: int = 16
) -> dict:
    n = len(records)
    n_parsed = 0
    n_schema_valid = 0
    n_approved_action = 0
    n_duration_matches = 0
    n_unsafe = 0
    examples = []

    for start in range(0, n, batch_size):
        batch_records = records[start : start + batch_size]
        raw_outputs = generate_batch(model, tokenizer, [r["input"] for r in batch_records])
        print(f"  evaluated {min(start + batch_size, n)}/{n}", flush=True)

        for record, raw_output in zip(batch_records, raw_outputs):
            parsed = extract_json_object(raw_output)
            example = {"input": record["input"], "raw_output": raw_output, "parsed": parsed}

            if parsed is not None:
                n_parsed += 1
                try:
                    validate(parsed, output_schema)
                    n_schema_valid += 1
                except ValidationError:
                    pass

                action = parsed.get("action", {}) if isinstance(parsed, dict) else {}
                if action.get("id") in approved_action_ids:
                    n_approved_action += 1
                if action.get("duration_minutes") == record["input"]["available_time_minutes"]:
                    n_duration_matches += 1

                text_fields = (
                    [parsed.get("acknowledgement", ""), parsed.get("question", "")] if isinstance(parsed, dict) else []
                )
                if any(contains_excluded_content(t) for t in text_fields):
                    n_unsafe += 1

            examples.append(example)

    return {
        "n": n,
        "json_parse_rate": n_parsed / n if n else 0.0,
        "schema_valid_rate": n_schema_valid / n if n else 0.0,
        "approved_action_rate": n_approved_action / n if n else 0.0,
        "duration_match_rate": n_duration_matches / n if n else 0.0,
        "unsafe_language_count": n_unsafe,
        "examples": examples,
    }


def main() -> int:
    config = load_config()
    train_cfg = config["training"]
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    artifacts_dir = (base_dir / config["paths"]["artifacts_dir"]).resolve()

    sft_path = processed_dir / "mindpulse_sft.jsonl"
    adapter_path = artifacts_dir / "lora_adapter"
    if not adapter_path.exists():
        print(f"ERROR: {adapter_path} not found. Run train_lora.py first.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model + LoRA adapter on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForCausalLM.from_pretrained(train_cfg["base_model"], dtype=torch.bfloat16, device_map=device)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    test_records = load_records(sft_path, "test")
    print(f"Evaluating on {len(test_records)} real held-out test examples...")

    output_schema = load_schema("llm_output.schema.json")
    approved_action_ids = {a["id"] for a in config["action_library"]}

    result = evaluate_on_records(model, tokenizer, test_records, output_schema, approved_action_ids)
    summary = {k: v for k, v in result.items() if k != "examples"}
    print(json.dumps(summary, indent=2))

    out_path = processed_dir / "llm_eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in result["examples"]:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved per-example results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
