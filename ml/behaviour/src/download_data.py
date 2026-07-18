"""Download the StudentLife dataset archive (Milestone 1).

Tries the official source first (https://studentlife.cs.dartmouth.edu/dataset/dataset.tar.bz2),
then falls back to a documented, real, working alternate path — a manually-
downloaded Kaggle mirror archive — since the official host was confirmed
unreachable during this project's real build (TCP connect timeout, verified
independently via curl, WebFetch, and Python requests; see docs/DATA_CARD.md).
Does not fabricate or substitute any data at any point: every path here either
extracts real downloaded bytes or fails loudly with the exact next step.
"""

import sys
import tarfile
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
ARCHIVE_NAME = "dataset.tar.bz2"
KAGGLE_MIRROR_URL = "https://www.kaggle.com/datasets/dartweichen/student-life"
KAGGLE_ZIP_NAME = "archive.zip"

# Same privacy-scoped file list used throughout this project (docs/PRODUCT_SPEC.md
# section 6.1) — only these paths are ever extracted from the Kaggle mirror zip,
# regardless of what else the zip contains (it also has audio/gps/wifi/call_log/sms,
# none of which this project uses or wants on disk).
KAGGLE_ZIP_MEMBERS = [
    "dataset/sensing/activity/*",
    "dataset/sensing/phonelock/*",
    "dataset/sensing/phonecharge/*",
    "dataset/EMA/EMA_definition.json",
    "dataset/EMA/response/Stress/*",
    "dataset/EMA/response/Mood/*",
    "dataset/EMA/response/Sleep/*",
    "dataset/EMA/response/Activity/*",
    "dataset/survey/PerceivedStressScale.csv",
    "dataset/survey/PHQ-9.csv",
    "dataset/survey/psqi.csv",
]


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def extract_official(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=extract_to)


def extract_kaggle_zip(zip_path: Path, extract_to: Path) -> None:
    """Matches this project's real, verified extraction: only the
    privacy-scoped member list, never the full archive (which also contains
    audio/gps/wifi/bluetooth/call_log/sms — explicitly excluded per
    docs/PRODUCT_SPEC.md section 6.1)."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        all_names = zf.namelist()
        to_extract = []
        for pattern in KAGGLE_ZIP_MEMBERS:
            if pattern.endswith("/*"):
                prefix = pattern[:-1]
                to_extract.extend(n for n in all_names if n.startswith(prefix))
            else:
                to_extract.extend(n for n in all_names if n == pattern)
        if not to_extract:
            raise RuntimeError(
                f"{zip_path} does not contain any of the expected StudentLife paths — "
                "this may not be the right archive. Expected members like "
                "'dataset/sensing/activity/...'."
            )
        zf.extractall(path=extract_to, members=to_extract)


def try_official_download(url: str, raw_dir: Path, archive_path: Path) -> bool:
    print(f"Attempting official source: {url}")
    try:
        download(url, archive_path)
    except requests.exceptions.RequestException as exc:
        print(f"Official source unreachable: {exc}", file=sys.stderr)
        return False
    print(f"Extracting {archive_path} -> {raw_dir}")
    extract_official(archive_path, raw_dir)
    return True


def try_kaggle_fallback(raw_dir: Path) -> bool:
    kaggle_zip = raw_dir / KAGGLE_ZIP_NAME
    if not kaggle_zip.exists():
        return False
    print(f"Found {kaggle_zip} — extracting privacy-scoped StudentLife files from it.")
    extract_kaggle_zip(kaggle_zip, raw_dir)
    return True


def main() -> int:
    config = load_config()
    url = config["dataset"]["source_url"]
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()
    archive_path = raw_dir / ARCHIVE_NAME

    if (raw_dir / "dataset").exists():
        print(f"Already extracted at {raw_dir / 'dataset'} — skipping download.")
        return 0

    if try_official_download(url, raw_dir, archive_path):
        print("Done (official source).")
        return 0

    print(f"\nTrying documented fallback: a Kaggle mirror of the same dataset.")
    if try_kaggle_fallback(raw_dir):
        print("Done (Kaggle mirror fallback).")
        return 0

    print(
        "\nERROR: could not obtain the dataset automatically. This script does not "
        "fall back to synthetic/mock data — here is the real, verified-working manual path:\n"
        f"\n  1. Download the archive from {KAGGLE_MIRROR_URL} (free Kaggle account, "
        "no approval/review wait — unlike the official Dartmouth host, which was "
        "unreachable when this project was built; see docs/DATA_CARD.md).\n"
        f"  2. Place the downloaded file at: {raw_dir / KAGGLE_ZIP_NAME}\n"
        "  3. Re-run this script — it will detect the file and extract the "
        "privacy-scoped subset automatically.\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
