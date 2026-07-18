"""Milestone 4 final evaluation: Leave-One-Participant-Out, matching the
ACTUALLY DEPLOYED mechanism (docs/PRODUCT_SPEC.md sections 8.4/8.5) — not the
population-generalization test in evaluate.py's GroupKFold comparison.

For each held-out participant:
  1. The TCN encoder is trained on all 48 OTHER participants only.
  2. That participant's own first 14 valid calendar days (real rows in
     daily_features.parquet, not window count) define their PERSONAL
     baseline: baseline_center = median, baseline_scale = MAD, over the
     embeddings of windows fully contained in that 14-day span.
  3. Later windows (window_end_date after the 14-day baseline period) are
     scored against that personal baseline via the drift-score formula
     (standardized_difference, epsilon=0.001) and evaluated against real
     within-person high-stress EMA labels.
  4. No held-out participant's future data influences scaling, training, or
     threshold selection at any point — the scaler and the encoder are both
     fit only on the other 48 participants.

Participants without at least 14 valid days plus at least one window
afterward cannot be evaluated this way and are skipped, with the count
reported rather than silently dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from ema_labels import build_high_stress_labels
from train_baselines import EPSILON, evaluate_against_labels
from train_encoder import get_embeddings, train_encoder

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
RANDOM_STATE = 42
BASELINE_DAYS = 14


def personal_drift_scores(baseline_embeddings: np.ndarray, eval_embeddings: np.ndarray) -> np.ndarray:
    """docs/PRODUCT_SPEC.md section 8.5 drift-score formula, applied per-person."""
    center = np.median(baseline_embeddings, axis=0)
    scale = np.median(np.abs(baseline_embeddings - center), axis=0)
    standardized = np.abs(eval_embeddings - center) / (scale + EPSILON)
    return standardized.mean(axis=1)


def run_loso(
    windows: np.ndarray, meta: pd.DataFrame, daily_df: pd.DataFrame, config: dict
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Returns (scores_for_evaluable_windows, meta_for_evaluable_windows, per_participant_report)."""
    all_scores = []
    all_meta_rows = []
    per_participant = {}

    participant_ids = sorted(meta["participant_id"].unique())
    for pid in participant_ids:
        valid_dates = sorted(daily_df.loc[daily_df["participant_id"] == pid, "date"].unique())
        if len(valid_dates) < BASELINE_DAYS:
            per_participant[pid] = {"status": "skipped_insufficient_valid_days", "valid_days": len(valid_dates)}
            continue
        baseline_cutoff_date = valid_dates[BASELINE_DAYS - 1]  # 14th valid day (inclusive)

        own_mask = meta["participant_id"] == pid
        own_idx = np.where(own_mask.to_numpy())[0]
        own_meta = meta.iloc[own_idx]

        baseline_idx = own_idx[own_meta["window_end_date"].to_numpy() <= baseline_cutoff_date]
        eval_idx = own_idx[own_meta["window_end_date"].to_numpy() > baseline_cutoff_date]

        if len(baseline_idx) == 0 or len(eval_idx) == 0:
            per_participant[pid] = {
                "status": "skipped_no_baseline_or_eval_windows",
                "baseline_windows": int(len(baseline_idx)),
                "eval_windows": int(len(eval_idx)),
            }
            continue

        other_idx = np.where(~own_mask.to_numpy())[0]
        scaler = StandardScaler().fit(windows[other_idx].reshape(len(other_idx), -1))

        def scale(idx):
            w = windows[idx]
            return scaler.transform(w.reshape(len(idx), -1)).reshape(w.shape)

        # Validation split for early stopping carved from the OTHER 48
        # participants only — never from the held-out participant's own data.
        rng = np.random.default_rng(RANDOM_STATE)
        n_other = len(other_idx)
        val_size = max(1, int(n_other * 0.1))
        perm = rng.permutation(n_other)
        val_local, fit_local = perm[:val_size], perm[val_size:]
        other_scaled = scale(other_idx)
        model = train_encoder(
            other_scaled[fit_local], val_windows=other_scaled[val_local], config=config, seed=RANDOM_STATE
        )

        baseline_embeddings = get_embeddings(model, scale(baseline_idx))
        eval_embeddings = get_embeddings(model, scale(eval_idx))

        scores = personal_drift_scores(baseline_embeddings, eval_embeddings)
        all_scores.append(scores)
        all_meta_rows.append(own_meta.iloc[np.searchsorted(own_idx, eval_idx)])
        per_participant[pid] = {
            "status": "evaluated",
            "baseline_windows": int(len(baseline_idx)),
            "eval_windows": int(len(eval_idx)),
        }
        print(f"  {pid}: trained on 48 others, {len(baseline_idx)} baseline windows, {len(eval_idx)} eval windows")

    if not all_scores:
        return np.array([]), pd.DataFrame(columns=meta.columns), per_participant

    return np.concatenate(all_scores), pd.concat(all_meta_rows, ignore_index=True), per_participant


def main() -> int:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = (base_dir / config["paths"]["processed_dir"]).resolve()
    raw_dir = (base_dir / config["paths"]["raw_dir"]).resolve()

    windows = np.load(processed_dir / "sequences.npy")
    meta = pd.read_parquet(processed_dir / "sequences_meta.parquet")
    daily_df = pd.read_parquet(processed_dir / "daily_features.parquet")
    print(f"Loaded {len(windows)} windows, {daily_df['participant_id'].nunique()} participants total")

    dataset_root = raw_dir / "dataset"
    labels = build_high_stress_labels(
        dataset_root / "EMA" / "response" / "Stress",
        dataset_root / "EMA" / "EMA_definition.json",
        sorted(meta["participant_id"].unique()),
    )

    print(f"Running LOSO over {meta['participant_id'].nunique()} participants (this trains one model per participant)...")
    scores, eval_meta, per_participant = run_loso(windows, meta, daily_df, config)

    n_evaluated = sum(1 for v in per_participant.values() if v["status"] == "evaluated")
    n_skipped = len(per_participant) - n_evaluated
    print(f"\nEvaluated {n_evaluated} participants, skipped {n_skipped} (insufficient data)")

    result = evaluate_against_labels(scores, eval_meta, labels) if len(scores) else {
        "n_matched": 0, "spearman_r": None, "spearman_p": None, "roc_auc": None
    }
    print("\n=== TCN encoder (LOSO, personal baseline) ===")
    print(result)

    out = {"result": result, "n_participants_evaluated": n_evaluated, "n_participants_skipped": n_skipped,
           "per_participant": per_participant}
    out_path = processed_dir / "encoder_loso_results.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f)
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
