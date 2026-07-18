"""Milestone 5: download the two real LLM source datasets.

EmpatheticDialogues (official Facebook/ParlAI release, CC BY-NC 4.0) and
ESConv (official thu-coai release, academic research use only). No
substitute or synthetic dialogue data is used — if either download fails,
this script fails loudly rather than fabricating conversations.
"""

import sys
import tarfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gemma270m_lora.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> int:
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()

    ed_dir = raw_dir / "empatheticdialogues"
    ed_archive = raw_dir / "empatheticdialogues.tar.gz"
    esconv_path = raw_dir / "ESConv.json"

    if not ed_dir.exists():
        url = config["datasets"]["empathetic_dialogues"]["source_url"]
        print(f"Downloading EmpatheticDialogues from {url}")
        try:
            download(url, ed_archive)
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: could not download EmpatheticDialogues: {exc}", file=sys.stderr)
            return 1
        print(f"Extracting {ed_archive} -> {raw_dir}")
        with tarfile.open(ed_archive, "r:gz") as tar:
            tar.extractall(path=raw_dir)
    else:
        print(f"EmpatheticDialogues already present at {ed_dir}")

    if not esconv_path.exists():
        url = config["datasets"]["esconv"]["source_url"]
        print(f"Downloading ESConv from {url}")
        try:
            download(url, esconv_path)
        except requests.exceptions.RequestException as exc:
            print(f"ERROR: could not download ESConv: {exc}", file=sys.stderr)
            return 1
    else:
        print(f"ESConv already present at {esconv_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
