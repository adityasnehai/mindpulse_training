"""Milestone 4 development evaluation: compare the TCN autoencoder against the
Milestone 3 baselines using the identical harness (same GroupKFold(participant_id,
5) split, same real within-person high-stress EMA labels, same
evaluate_against_labels scoring) — an apples-to-apples test of whether the
temporal encoder is worth its added complexity, per docs/PRODUCT_SPEC.md
section 8.1.

The encoder's anomaly score is the same robust-median/MAD standardized-
distance formula as the population z-score baseline (section 8.5's drift
score), but computed on the encoder's 8-dim embeddings instead of the raw
24-dim window. Final Leave-One-Participant-Out evaluation and personal
(rather than population) baselines are a separate, later step
(train_encoder.py + this module's LOSO path) — this function specifically
answers the Milestone 3/8.1 "is the encoder better than the simple
baselines" question under the same test conditions already used for them.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ema_labels import build_high_stress_labels
from train_baselines import EPSILON, evaluate_against_labels, robust_zscore_scores
from train_encoder import get_embeddings, train_encoder

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
N_FOLDS = 5
RANDOM_STATE = 42


def encoder_group_kfold(
    windows: np.ndarray, meta: pd.DataFrame, config: dict, val_fraction: float = 0.1
) -> np.ndarray:
    """Train one TCN autoencoder per GroupKFold fold (fit only on that fold's
    training participants), embed all windows, and score the held-out fold
    with the same robust-median/MAD anomaly score as the z-score baseline —
    fit on that fold's TRAIN embeddings only, applied to TEST embeddings."""
    n_windows = windows.shape[0]
    groups = meta["participant_id"].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = np.zeros(n_windows)

    rng = np.random.default_rng(RANDOM_STATE)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(windows, groups=groups)):
        flat_train = windows[train_idx].reshape(len(train_idx), -1)
        scaler = StandardScaler().fit(flat_train)

        def scale(w):
            n = w.shape[0]
            return scaler.transform(w.reshape(n, -1)).reshape(w.shape)

        train_windows_scaled = scale(windows[train_idx])
        test_windows_scaled = scale(windows[test_idx])

        # Carve a validation split out of the fold's training participants for early stopping.
        n_train = len(train_idx)
        val_size = max(1, int(n_train * val_fraction))
        perm = rng.permutation(n_train)
        val_local_idx, fit_local_idx = perm[:val_size], perm[val_size:]

        model = train_encoder(
            train_windows_scaled[fit_local_idx],
            val_windows=train_windows_scaled[val_local_idx],
            config=config,
            seed=RANDOM_STATE,
        )

        train_embeddings = get_embeddings(model, train_windows_scaled)
        test_embeddings = get_embeddings(model, test_windows_scaled)

        fold_scores = robust_zscore_scores(train_embeddings, test_embeddings)
        scores[test_idx] = fold_scores
        print(f"  fold {fold_idx + 1}/{N_FOLDS} done ({len(test_idx)} test windows)")

    return scores


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()

    windows = np.load(processed_dir / "sequences.npy")
    meta = pd.read_parquet(processed_dir / "sequences_meta.parquet")
    print(f"Loaded {len(windows)} windows from {meta['participant_id'].nunique()} participants")

    dataset_root = raw_dir / "dataset"
    stress_dir = dataset_root / "EMA" / "response" / "Stress"
    ema_def = dataset_root / "EMA" / "EMA_definition.json"
    labels = build_high_stress_labels(stress_dir, ema_def, sorted(meta["participant_id"].unique()))
    print(f"Built {len(labels)} within-person high-stress labels")

    print("Training TCN autoencoder per GroupKFold(participant_id, 5) fold...")
    scores = encoder_group_kfold(windows, meta, config)

    result = evaluate_against_labels(scores, meta, labels)
    print("\n=== TCN encoder (dev GroupKFold) ===")
    print(f"n_matched={result['n_matched']}  spearman_r={result['spearman_r']}  roc_auc={result['roc_auc']}")

    baseline_path = processed_dir / "baseline_results.yaml"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_results = yaml.safe_load(f)
        print("\n=== Comparison vs. Milestone 3 baselines ===")
        for name, r in baseline_results.items():
            auc = r["roc_auc"]
            print(f"{name:20s} roc_auc={auc}")
        print(f"{'tcn_encoder':20s} roc_auc={result['roc_auc']}")

    out_path = processed_dir / "encoder_dev_results.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
