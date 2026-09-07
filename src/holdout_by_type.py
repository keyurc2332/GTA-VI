"""
Milestone 20: Holdout by event type.

The chronological holdout (event_holdout.py) tests generalization across
TIME. This tests something different: generalization across CATEGORY. Tune
the detector using only "predictable" event types (official announcements,
marketing beats, controversy), then test on "less scripted" types (leaks,
rumors) that the detector never saw during tuning.

This answers a sharper question than the chronological split: is the
detector learning to recognize a general attention shock, or is it
implicitly keyed to the specific rhythm of official Rockstar announcements
(which follow a predictable PR cadence) and unable to generalize to
messier, less predictable event types like leaks?

Run:
    python src/holdout_by_type.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, compute_rolling_zscore
from evaluate_detection import evaluate

WINDOW_GRID = [4, 8, 12]
THRESHOLD_GRID = [1.5, 2.0, 2.5]

DEV_TYPES = ["official", "marketing", "controversy"]
HOLDOUT_TYPES = ["leak", "rumor"]


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    dev_events = events_df[events_df["type"].isin(DEV_TYPES)].reset_index(drop=True)
    holdout_events = events_df[events_df["type"].isin(HOLDOUT_TYPES)].reset_index(drop=True)

    print(f"Dev types {DEV_TYPES}: {len(dev_events)} events")
    print(f"Holdout types {HOLDOUT_TYPES}: {len(holdout_events)} events\n")

    print("Grid-searching window/threshold on DEV event TYPES only...\n")
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
    print(f"\nBest on dev types: window={best_window}, threshold={best_threshold} "
          f"(dev F1={best_row['f1']:.3f})")

    print(f"\nApplying to HOLDOUT event TYPES (leaks/rumors -- categories "
          f"never used for tuning)...\n")
    z_final = compute_rolling_zscore(df["hype_index_equal"], best_window)
    spikes_final = df["date"][z_final >= best_threshold]
    holdout_scores = evaluate(spikes_final, holdout_events, EVENT_MATCH_WINDOW_DAYS)

    print(f"Holdout-by-type result: precision={holdout_scores['precision']:.3f}, "
          f"recall={holdout_scores['recall']:.3f}, f1={holdout_scores['f1']:.3f}")

    gap = best_row["f1"] - holdout_scores["f1"]
    if gap > 0.15:
        print(
            f"\nThere's a real drop ({gap:.3f}) moving from predictable, "
            f"scripted event types (official/marketing/controversy) to "
            f"messier ones (leaks/rumors). This suggests the detector may be "
            f"partly keyed to the specific rhythm of official Rockstar "
            f"announcements rather than detecting a truly general attention "
            f"shock -- leaks and rumors don't follow that same predictable "
            f"cadence."
        )
    else:
        print(
            f"\nPerformance holds up reasonably well moving from official/"
            f"marketing/controversy events to leaks/rumors (gap: {gap:.3f}) "
            f"-- the detector doesn't appear to be narrowly keyed to "
            f"scripted announcement patterns."
        )

    result_summary = pd.DataFrame([{
        "dev_types": ", ".join(DEV_TYPES),
        "holdout_types": ", ".join(HOLDOUT_TYPES),
        "n_dev_events": len(dev_events),
        "n_holdout_events": len(holdout_events),
        "best_window": best_window,
        "best_threshold": best_threshold,
        "dev_f1": round(best_row["f1"], 4),
        "holdout_f1": round(holdout_scores["f1"], 4),
        "holdout_precision": round(holdout_scores["precision"], 4),
        "holdout_recall": round(holdout_scores["recall"], 4),
        "gap": round(gap, 4),
    }])
    out_path = PROCESSED_DATA_DIR / "holdout_by_type.csv"
    result_summary.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
