"""
Milestone 23: Extended signal ablation with YouTube split.

The original ablation (signal_ablation.py) tested Trends/Wikipedia/YouTube
as three signals, and found the blended YouTube signal doesn't help. But
"YouTube" was one blurred number mixing upload activity with lifetime view
counts. This tests whether creator activity and audience engagement --
now split into two separate signals -- behave differently from each other,
rather than assuming the whole blend is equally uninformative.

Run:
    python src/youtube_signal_ablation.py
"""

import sys
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD, compute_rolling_zscore
from evaluate_detection import evaluate

ALL_SIGNALS = {
    "trends_signal": "Google Trends",
    "wiki_signal": "Wikipedia",
    "youtube_creator_signal": "YouTube (creator activity)",
    "youtube_audience_signal": "YouTube (audience engagement)",
}

EVENTS_DF = pd.DataFrame(KNOWN_EVENTS)
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])


def build_combo_signal(df: pd.DataFrame, signal_cols: tuple) -> pd.Series:
    return df[list(signal_cols)].mean(axis=1)


def main():
    path = PROCESSED_DATA_DIR / "hype_index_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    missing = [c for c in ALL_SIGNALS if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns {missing}. Re-run build_hype_index.py "
              f"with the updated script that splits YouTube into creator/"
              f"audience signals first.")
        return

    signal_keys = list(ALL_SIGNALS.keys())
    all_combos = []
    for r in range(1, len(signal_keys) + 1):
        all_combos.extend(combinations(signal_keys, r))

    print(f"Testing {len(all_combos)} signal combinations with YouTube split "
          f"into creator-activity vs audience-engagement "
          f"(window={ROLLING_WINDOW}, threshold={Z_SCORE_THRESHOLD})\n")

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
    print(f"\nBest overall: {best['signals']} (F1={best['f1']:.3f})")

    # Specifically: does creator activity or audience engagement look more
    # informative on its own?
    creator_alone = results_df[results_df["signals"] == "YouTube (creator activity)"]
    audience_alone = results_df[results_df["signals"] == "YouTube (audience engagement)"]
    if not creator_alone.empty and not audience_alone.empty:
        c_f1 = creator_alone["f1"].iloc[0]
        a_f1 = audience_alone["f1"].iloc[0]
        print(f"\nYouTube creator-activity alone: F1={c_f1:.3f}")
        print(f"YouTube audience-engagement alone: F1={a_f1:.3f}")
        if abs(c_f1 - a_f1) > 0.05:
            better = "creator activity" if c_f1 > a_f1 else "audience engagement"
            print(
                f"\nSplitting the blended YouTube signal reveals a real "
                f"difference: {better} alone is more informative than the "
                f"other. The original blended signal was averaging a more "
                f"useful sub-signal together with a less useful one, "
                f"potentially diluting it."
            )
        else:
            print(
                f"\nCreator activity and audience engagement perform "
                f"similarly once separated -- the original blended signal "
                f"wasn't hiding a meaningfully stronger sub-signal."
            )

    out_path = PROCESSED_DATA_DIR / "youtube_signal_ablation.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
