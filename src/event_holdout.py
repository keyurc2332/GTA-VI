"""
Milestone 15: Discovery/evaluation separation via a dev/holdout event split.

The methodological issue this fixes: every previous evaluation (baseline
comparison, sensitivity grid, robustness check) picked whichever detector
parameters scored best against the FULL event registry. That's tuning on
your own test set -- the resulting F1 numbers look better than a genuinely
unseen evaluation would produce.

Fix: split the 21 events chronologically -- roughly the first two-thirds
(dev) and the most recent third (holdout). Grid-search window/threshold
using ONLY the dev events. Freeze that choice. Evaluate ONCE on the
holdout events, which were never used for any tuning decision.

This is a chronological split (not random) deliberately: it mirrors real
deployment, where you'd tune a detector on past events and then apply it
to events you didn't have when designing it, rather than a random shuffle
that assumes performance is time-invariant.

Run:
    python src/event_holdout.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS
from evaluate_detection import evaluate

WINDOW_GRID = [4, 8, 12]
THRESHOLD_GRID = [1.5, 2.0, 2.5]
DEV_FRACTION = 2 / 3


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])
    events_df = events_df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(events_df) * DEV_FRACTION)
    dev_events = events_df.iloc[:split_idx].reset_index(drop=True)
    holdout_events = events_df.iloc[split_idx:].reset_index(drop=True)

    print(f"Dev events ({len(dev_events)}): {dev_events['date'].min().date()} to {dev_events['date'].max().date()}")
    print(f"Holdout events ({len(holdout_events)}): {holdout_events['date'].min().date()} to {holdout_events['date'].max().date()}\n")

    # --- Step 1: grid search on DEV events only ---
    print("Grid-searching window/threshold on DEV events only...\n")
    dev_rows = []
    for window in WINDOW_GRID:
        z = compute_rolling_zscore(df["hype_index_equal"], window)
        for threshold in THRESHOLD_GRID:
            spikes = df["date"][z >= threshold]
            scores = evaluate(spikes, dev_events, EVENT_MATCH_WINDOW_DAYS)
            dev_rows.append({"window": window, "threshold": threshold, **scores})

    dev_results = pd.DataFrame(dev_rows)
    print(dev_results.to_string(index=False))

    best_row = dev_results.loc[dev_results["f1"].idxmax()]
    best_window, best_threshold = int(best_row["window"]), best_row["threshold"]
    print(f"\nBest on DEV: window={best_window}, threshold={best_threshold} "
          f"(dev F1={best_row['f1']:.3f})")

    # --- Step 2: freeze that choice, evaluate ONCE on HOLDOUT ---
    print(f"\nApplying frozen parameters (window={best_window}, threshold={best_threshold}) "
          f"to HOLDOUT events (never used for tuning)...\n")

    z_final = compute_rolling_zscore(df["hype_index_equal"], best_window)
    spikes_final = df["date"][z_final >= best_threshold]
    holdout_scores = evaluate(spikes_final, holdout_events, EVENT_MATCH_WINDOW_DAYS)

    print(f"HOLDOUT result: precision={holdout_scores['precision']:.3f}, "
          f"recall={holdout_scores['recall']:.3f}, f1={holdout_scores['f1']:.3f}")

    # --- Comparison: what if we'd (incorrectly) tuned on everything? ---
    print("\nFor comparison -- what tuning on the FULL event set (dev+holdout "
          "combined) would have shown, i.e. the leakage scenario this split avoids:")
    full_rows = []
    for window in WINDOW_GRID:
        z = compute_rolling_zscore(df["hype_index_equal"], window)
        for threshold in THRESHOLD_GRID:
            spikes = df["date"][z >= threshold]
            scores = evaluate(spikes, events_df, EVENT_MATCH_WINDOW_DAYS)
            full_rows.append({"window": window, "threshold": threshold, **scores})
    full_results = pd.DataFrame(full_rows)
    full_best = full_results.loc[full_results["f1"].idxmax()]
    print(f"  Full-set-tuned F1: {full_best['f1']:.3f} (window={int(full_best['window'])}, "
          f"threshold={full_best['threshold']})")
    print(f"  Honest holdout F1: {holdout_scores['f1']:.3f}")

    gap = full_best["f1"] - holdout_scores["f1"]
    if gap > 0.05:
        print(f"\n  The full-set-tuned number is {gap:.3f} higher than the honest "
              f"holdout result -- a meaningful gap. This is exactly the kind of "
              f"optimistic bias that comes from evaluating on the same events "
              f"used to pick parameters. The holdout F1 is the more trustworthy "
              f"number to report.")
    else:
        print(f"\n  The gap between full-set-tuned and honest holdout F1 is small "
              f"({gap:.3f}) -- the detector's performance doesn't appear to be "
              f"an artifact of tuning on the test events.")

    result_summary = pd.DataFrame([{
        "n_dev_events": len(dev_events),
        "n_holdout_events": len(holdout_events),
        "best_window": best_window,
        "best_threshold": best_threshold,
        "dev_f1": round(best_row["f1"], 4),
        "holdout_f1": round(holdout_scores["f1"], 4),
        "holdout_precision": round(holdout_scores["precision"], 4),
        "holdout_recall": round(holdout_scores["recall"], 4),
        "full_set_tuned_f1": round(full_best["f1"], 4),
        "tuning_gap": round(gap, 4),
    }])
    out_path = PROCESSED_DATA_DIR / "event_holdout.csv"
    result_summary.to_csv(out_path, index=False)
    dev_results.to_csv(PROCESSED_DATA_DIR / "event_holdout_dev_grid.csv", index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
