"""
Milestone 11: Spike detector sensitivity analysis.

The spike detector uses ROLLING_WINDOW=8 and Z_SCORE_THRESHOLD=1.5 with no
data-driven justification for those exact numbers. This runs a grid search
across reasonable alternatives and evaluates each against the verified
event list, so the choice can be defended with "this combination performed
best/robustly across the grid" rather than "it looked good."

Run:
    python src/spike_sensitivity.py
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

EVENTS_DF = pd.DataFrame(KNOWN_EVENTS)
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Grid search: window in {WINDOW_GRID}, threshold in {THRESHOLD_GRID}\n")

    rows = []
    for window in WINDOW_GRID:
        z = compute_rolling_zscore(df["hype_index_equal"], window)
        for threshold in THRESHOLD_GRID:
            spikes = df["date"][z >= threshold]
            scores = evaluate(spikes, EVENTS_DF, EVENT_MATCH_WINDOW_DAYS)
            rows.append({"window": window, "threshold": threshold, **scores})

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["f1"].idxmax()]
    print(f"\nBest combination on this event set: window={int(best_row['window'])}, "
          f"threshold={best_row['threshold']} (F1={best_row['f1']:.3f})")

    # Robustness check: how much does F1 vary across the whole grid?
    f1_std = results_df["f1"].std()
    f1_range = results_df["f1"].max() - results_df["f1"].min()
    print(f"\nF1 across the full grid: mean={results_df['f1'].mean():.3f}, "
          f"std={f1_std:.3f}, range={f1_range:.3f}")
    if f1_range < 0.15:
        print("The result is fairly stable across reasonable parameter choices -- "
              "the original window=8, threshold=1.5 wasn't a lucky pick that "
              "happens to look good only at that exact setting.")
    else:
        print("The result varies meaningfully depending on these parameters. "
              "The chosen window=8, threshold=1.5 should be reported as ONE "
              "reasonable choice among several, not the objectively correct one.")

    # Pivot to a heatmap-ready shape for the dashboard
    heatmap_df = results_df.pivot(index="window", columns="threshold", values="f1")
    out_path = PROCESSED_DATA_DIR / "spike_sensitivity.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
