"""Milestone 5: independent validation pass over the final written
mindpulse_sft.jsonl — re-validates from disk (not reusing in-memory objects
from build_sft_dataset.py) against the real schemas, checks action-ID
approval and keyword safety on the OUTPUT text itself (catching issues in
templated product-scenario phrasing too, not just source dialogue text), and
writes the manual-review sample docs/PRODUCT_SPEC.md section 11.3 requires
(>=300 training, >=100 validation examples).
"""

import json
import random
from pathlib import Path

import jsonschema
import yaml

from safety_filters import contains_excluded_content

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "data" / "schemas"


def load_schema(name: str) -> dict:
    with open(SCHEMAS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_records(records: list[dict], input_schema: dict, output_schema: dict, approved_action_ids: set) -> dict:
    n = len(records)
    schema_failures = []
    action_id_failures = []
    keyword_failures = []

    for i, record in enumerate(records):
        try:
            jsonschema.validate(record["input"], input_schema)
            jsonschema.validate(record["output"], output_schema)
        except jsonschema.ValidationError as exc:
            schema_failures.append({"index": i, "error": str(exc.message)})
            continue

        if record["output"]["action"]["id"] not in approved_action_ids:
            action_id_failures.append(i)

        text_fields = [record["output"]["acknowledgement"], record["output"]["question"]]
        if any(contains_excluded_content(t) for t in text_fields):
            keyword_failures.append(i)

    return {
        "total": n,
        "schema_valid": n - len(schema_failures),
        "schema_failures": schema_failures[:20],
        "n_schema_failures": len(schema_failures),
        "n_action_id_failures": len(action_id_failures),
        "n_keyword_failures": len(keyword_failures),
        "keyword_failure_indices": keyword_failures[:20],
    }


def write_manual_review_sample(records: list[dict], out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    by_split = {"train": [], "validation": [], "test": []}
    for r in records:
        by_split.setdefault(r["split"], []).append(r)

    sample = []
    sample += rng.sample(by_split["train"], min(300, len(by_split["train"])))
    sample += rng.sample(by_split["validation"], min(100, len(by_split["validation"])))

    with open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    sft_path = processed_dir / "mindpulse_sft.jsonl"

    if not sft_path.exists():
        print(f"ERROR: {sft_path} not found. Run build_sft_dataset.py first.")
        return 1

    records = [json.loads(line) for line in open(sft_path, "r", encoding="utf-8")]
    input_schema = load_schema("llm_input.schema.json")
    output_schema = load_schema("llm_output.schema.json")
    approved_action_ids = {a["id"] for a in config["action_library"]}

    result = validate_records(records, input_schema, output_schema, approved_action_ids)
    print(json.dumps({k: v for k, v in result.items() if k not in ("schema_failures", "keyword_failure_indices")}, indent=2))

    if result["n_schema_failures"] or result["n_action_id_failures"] or result["n_keyword_failures"]:
        print("VALIDATION FAILED — see failures above.")
        return 1

    split_counts = {}
    for r in records:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    print(f"Split counts: {split_counts}")

    review_path = processed_dir / "manual_review_sample.jsonl"
    write_manual_review_sample(records, review_path, config["seed"])
    print(f"Manual-review sample written to {review_path}")

    print("All records passed independent validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
