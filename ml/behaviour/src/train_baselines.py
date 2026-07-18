"""Milestone 3: train and evaluate the 4 baselines required before the temporal
encoder (docs/PRODUCT_SPEC.md section 8.1), using GroupKFold(participant_id, 5)
per section 8.4 so no participant's windows appear in both train and test.

Each baseline scores every window with an anomaly score (higher = more
anomalous). Evaluated against the real within-person high-stress-day labels
(ema_labels.py) via Spearman correlation and ROC-AUC, matched on the window's
end date (the day the "current" drift score would be computed for in the
deployed product).

This intentionally evaluates each baseline as a population-trained (not
per-person) anomaly detector under GroupKFold — the deployed product instead
uses a personal median/MAD baseline per docs/PRODUCT_SPEC.md section 8.5; that
is a separate, later mechanism. The purpose here is exactly what section 8.1
asks: establish whether a temporal encoder (Milestone 4) is worth its added
complexity relative to these simpler population baselines, using a common,
reusable evaluation harness.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

from ema_labels import build_high_stress_labels

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
EPSILON = 0.001
N_FOLDS = 5
RANDOM_STATE = 42


def robust_zscore_scores(train_flat: np.ndarray, test_flat: np.ndarray) -> np.ndarray:
    """Per-feature median/MAD anomaly score, matching the product's personal-
    baseline drift-score formula (section 8.5) but fit on the training fold."""
    median = np.median(train_flat, axis=0)
    mad = np.median(np.abs(train_flat - median), axis=0)
    standardized = np.abs(test_flat - median) / (mad + EPSILON)
    return standardized.mean(axis=1)


def isolation_forest_scores(train_flat: np.ndarray, test_flat: np.ndarray) -> np.ndarray:
    model = IsolationForest(random_state=RANDOM_STATE, n_estimators=200)
    model.fit(train_flat)
    return -model.score_samples(test_flat)  # higher = more anomalous


def one_class_svm_scores(train_flat: np.ndarray, test_flat: np.ndarray) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    model.fit(train_flat)
    return -model.decision_function(test_flat)  # higher = more anomalous


def dense_autoencoder_scores(train_flat: np.ndarray, test_flat: np.ndarray) -> np.ndarray:
    """Small dense autoencoder via sklearn's MLPRegressor (input=output,
    bottleneck hidden layer) — deliberately not TensorFlow/Keras here, since
    this baseline exists only to be compared against, not to be the deployed
    model; keeping Milestone 3 on already-installed dependencies."""
    model = MLPRegressor(
        hidden_layer_sizes=(64, 8, 64),
        activation="relu",
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
    )
    model.fit(train_flat, train_flat)
    reconstructed = model.predict(test_flat)
    return np.mean((test_flat - reconstructed) ** 2, axis=1)


BASELINES = {
    "robust_zscore": robust_zscore_scores,
    "isolation_forest": isolation_forest_scores,
    "one_class_svm": one_class_svm_scores,
    "dense_autoencoder": dense_autoencoder_scores,
}


def evaluate_against_labels(scores: np.ndarray, meta: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Join per-window scores (keyed on participant_id + window_end_date) to
    within-person high-stress labels on the same date, then compute Spearman
    correlation (score vs severity) and ROC-AUC (score vs high_stress)."""
    joined = meta.assign(score=scores).merge(
        labels, left_on=["participant_id", "window_end_date"], right_on=["participant_id", "date"], how="inner"
    )
    if len(joined) < 10 or joined["high_stress"].nunique() < 2:
        return {"n_matched": len(joined), "spearman_r": None, "spearman_p": None, "roc_auc": None}

    rho, p_value = spearmanr(joined["score"], joined["severity"])
    auc = roc_auc_score(joined["high_stress"], joined["score"])
    return {
        "n_matched": int(len(joined)),
        "spearman_r": float(rho),
        "spearman_p": float(p_value),
        "roc_auc": float(auc),
    }


def run_group_kfold(windows: np.ndarray, meta: pd.DataFrame, labels: pd.DataFrame) -> dict:
    n_windows = windows.shape[0]
    flat = windows.reshape(n_windows, -1)
    groups = meta["participant_id"].to_numpy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_scores = {name: np.zeros(n_windows) for name in BASELINES}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(flat, groups=groups)):
        scaler = StandardScaler().fit(flat[train_idx])
        train_scaled = scaler.transform(flat[train_idx])
        test_scaled = scaler.transform(flat[test_idx])

        for name, fn in BASELINES.items():
            all_scores[name][test_idx] = fn(train_scaled, test_scaled)
        print(f"  fold {fold_idx + 1}/{N_FOLDS} done ({len(test_idx)} test windows)")

    results = {}
    for name, scores in all_scores.items():
        results[name] = evaluate_against_labels(scores, meta, labels)
    return results


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()

    windows_path = processed_dir / "sequences.npy"
    meta_path = processed_dir / "sequences_meta.parquet"
    if not windows_path.exists() or not meta_path.exists():
        print("ERROR: run build_sequences.py first.")
        return 1

    windows = np.load(windows_path)
    meta = pd.read_parquet(meta_path)
    print(f"Loaded {len(windows)} windows from {meta['participant_id'].nunique()} participants")

    dataset_root = raw_dir / "dataset"
    stress_dir = dataset_root / "EMA" / "response" / "Stress"
    ema_def = dataset_root / "EMA" / "EMA_definition.json"
    labels = build_high_stress_labels(stress_dir, ema_def, sorted(meta["participant_id"].unique()))
    print(f"Built {len(labels)} within-person high-stress labels from real Stress EMA data")

    print("Running GroupKFold(participant_id, 5 folds) for each baseline...")
    results = run_group_kfold(windows, meta, labels)

    print("\n=== Baseline comparison (Spearman r vs severity, ROC-AUC vs high_stress) ===")
    for name, r in results.items():
        print(f"{name:20s} n_matched={r['n_matched']:4d}  spearman_r={r['spearman_r']}  roc_auc={r['roc_auc']}")

    out_path = processed_dir / "baseline_results.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
