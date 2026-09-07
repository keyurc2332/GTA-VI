"""
Milestone 16: Signal ablation study.

Every evaluation so far compares "Trends alone" vs "all three signals
combined" -- but that skips a real question: does EVERY signal contribute,
or is one of them dead weight? Maybe YouTube adds nothing. Maybe Wikipedia
alone is surprisingly strong. This tests every single signal and every
pair, not just the two endpoints.

Uses the project's stated default detector (window=8, threshold=1.5)
throughout, so differences reflect signal content, not parameter tuning --
that's a separate question, handled by event_holdout.py.

Run:
    python src/signal_ablation.py
"""

import sys
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD
from evaluate_detection import evaluate

ALL_SIGNALS = {
    "trends_signal": "Google Trends",
    "wiki_signal": "Wikipedia",
    "youtube_signal": "YouTube",
}

EVENTS_DF = pd.DataFrame(KNOWN_EVENTS)
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def build_combo_signal(df: pd.DataFrame, signal_cols: tuple) -> pd.Series:
    """Simple average of whichever signal columns are in this combination."""
    return df[list(signal_cols)].mean(axis=1)


def main():
    path = PROCESSED_DATA_DIR / "hype_index_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    signal_keys = list(ALL_SIGNALS.keys())
    all_combos = []
    for r in range(1, len(signal_keys) + 1):
        all_combos.extend(combinations(signal_keys, r))

    print(f"Testing {len(all_combos)} signal combinations "
          f"(window={ROLLING_WINDOW}, threshold={Z_SCORE_THRESHOLD}, "
          f"the project's stated default detector)\n")

    rows = []
    for combo in all_combos:
        combo_signal = build_combo_signal(df, combo)
        z = compute_rolling_zscore(combo_signal, ROLLING_WINDOW)
        spikes = df["date"][z >= Z_SCORE_THRESHOLD]
        scores = evaluate(spikes, EVENTS_DF, EVENT_MATCH_WINDOW_DAYS)
        combo_label = " + ".join(ALL_SIGNALS[c] for c in combo)
        rows.append({"signals": combo_label, "n_signals": len(combo), **scores})

    results_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    all_three_row = results_df[results_df["n_signals"] == 3].iloc[0]
    single_signal_best = results_df[results_df["n_signals"] == 1].iloc[0]

    print(f"\nBest overall: {best['signals']} (F1={best['f1']:.3f})")
    print(f"All three combined: F1={all_three_row['f1']:.3f}")
    print(f"Best single signal alone: {single_signal_best['signals']} (F1={single_signal_best['f1']:.3f})")

    if best["n_signals"] < 3:
        print(
            f"\nThe best-performing combination on this event set does NOT "
            f"use all three signals. That's a real finding worth reporting: "
            f"adding every available signal isn't automatically better than "
            f"a more selective combination."
        )
    else:
        print(
            f"\nUsing all three signals together performs best on this event "
            f"set -- combining signals earns its complexity rather than just "
            f"adding it for its own sake."
        )

    worst_single = results_df[results_df["n_signals"] == 1].sort_values("f1").iloc[0]
    print(f"\nWeakest individual signal: {worst_single['signals']} (F1={worst_single['f1']:.3f}) "
          f"-- worth checking whether it's pulling its weight in the full combination.")

    out_path = PROCESSED_DATA_DIR / "signal_ablation.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
