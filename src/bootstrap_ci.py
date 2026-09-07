"""
Milestone 17: Bootstrap confidence intervals on detection metrics.

Every F1 reported so far is a single point estimate from 21 events. With a
sample that small, that number could look very different with a slightly
different set of events. This resamples the EVENT LIST (not the spikes --
those are fixed once the detector runs) with replacement 2,000 times,
recomputes precision/recall/F1 against each resampled event set, and
reports the 95% interval across those resamples.

This directly answers: "is F1=0.56 solid, or would a slightly different
sample of GTA VI events have given something like F1=0.30 or F1=0.75?"

Run:
    python src/bootstrap_ci.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import (
    KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD,
    compute_rolling_zscore,
)
from evaluate_detection import evaluate

N_BOOTSTRAP = 2000
CI_PERCENTILES = (2.5, 97.5)
RANDOM_SEED = 42


def bootstrap_metric_ci(spikes: pd.Series, events_df: pd.DataFrame, tolerance_days: int,
                          n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_events = len(events_df)

    precisions, recalls, f1s = [], [], []
    for _ in range(n_bootstrap):
        # Resample events WITH replacement -- this simulates "what if we'd
        # happened to document a slightly different set of GTA VI events"
        sample_idx = rng.integers(0, n_events, size=n_events)
        resampled_events = events_df.iloc[sample_idx].reset_index(drop=True)
        scores = evaluate(spikes, resampled_events, tolerance_days)
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])

    def ci(values):
        lower, upper = np.percentile(values, CI_PERCENTILES)
        return round(np.mean(values), 4), round(lower, 4), round(upper, 4)

    p_mean, p_lo, p_hi = ci(precisions)
    r_mean, r_lo, r_hi = ci(recalls)
    f_mean, f_lo, f_hi = ci(f1s)

    return {
        "precision_mean": p_mean, "precision_ci_lower": p_lo, "precision_ci_upper": p_hi,
        "recall_mean": r_mean, "recall_ci_lower": r_lo, "recall_ci_upper": r_hi,
        "f1_mean": f_mean, "f1_ci_lower": f_lo, "f1_ci_upper": f_hi,
    }


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    z_baseline = compute_rolling_zscore(df["trends_signal"], ROLLING_WINDOW)
    baseline_spikes = df["date"][z_baseline >= Z_SCORE_THRESHOLD]

    z_combined = compute_rolling_zscore(df["hype_index_equal"], ROLLING_WINDOW)
    combined_spikes = df["date"][z_combined >= Z_SCORE_THRESHOLD]

    print(f"Bootstrapping {N_BOOTSTRAP} resamples of the {len(events_df)}-event "
          f"registry for each method...\n")

    baseline_ci = bootstrap_metric_ci(baseline_spikes, events_df, EVENT_MATCH_WINDOW_DAYS,
                                        N_BOOTSTRAP, RANDOM_SEED)
    combined_ci = bootstrap_metric_ci(combined_spikes, events_df, EVENT_MATCH_WINDOW_DAYS,
                                        N_BOOTSTRAP, RANDOM_SEED + 1)

    results = pd.DataFrame([
        {"method": "Baseline (Trends only)", **baseline_ci},
        {"method": "Combined (Hype Index)", **combined_ci},
    ])

    print("F1 with 95% bootstrap confidence intervals:")
    for _, row in results.iterrows():
        print(f"  {row['method']}: F1 = {row['f1_mean']:.3f} "
              f"[{row['f1_ci_lower']:.3f}, {row['f1_ci_upper']:.3f}]")

    baseline_range = baseline_ci["f1_ci_upper"] - baseline_ci["f1_ci_lower"]
    combined_range = combined_ci["f1_ci_upper"] - combined_ci["f1_ci_lower"]
    overlap = not (combined_ci["f1_ci_lower"] > baseline_ci["f1_ci_upper"] or
                   baseline_ci["f1_ci_lower"] > combined_ci["f1_ci_upper"])

    print(f"\nBaseline CI width: {baseline_range:.3f}")
    print(f"Combined CI width: {combined_range:.3f}")

    if overlap:
        print(
            "\nThe two methods' confidence intervals OVERLAP. Combined "
            "scoring a higher point estimate doesn't mean it's decisively "
            "better -- with this event sample size, the difference could "
            "plausibly be noise. Report both the point estimate and this "
            "overlap honestly rather than treating the higher F1 as proven."
        )
    else:
        print(
            "\nThe two methods' confidence intervals do NOT overlap -- the "
            "combined method's advantage over the baseline looks like a real "
            "difference, not just noise from a small event sample."
        )

    out_path = PROCESSED_DATA_DIR / "bootstrap_ci.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
