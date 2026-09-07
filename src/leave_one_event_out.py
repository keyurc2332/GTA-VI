"""
Milestone 19: Leave-one-event-out (LOO) cross-validation.

The single 14/7 holdout split in event_holdout.py showed a big generalization
gap (F1 0.474 -> 0.076), but that's ONE split -- with only 21 events, a
single train/test division is itself a small, noisy sample. LOO uses the
data far more efficiently: for each of the 21 events, tune window/threshold
on the OTHER 20, then check whether that one held-out event would have been
caught. Repeat 21 times and aggregate.

This answers: was the 14/7 result an unlucky specific split, or does the
generalization problem show up consistently across many different held-out
events?

Run:
    python src/leave_one_event_out.py
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


def tune_on_events(df: pd.DataFrame, events_df: pd.DataFrame) -> tuple:
    """Grid search window/threshold on the given events, return the best combo."""
    best_f1, best_window, best_threshold = -1, None, None
    for window in WINDOW_GRID:
        z = compute_rolling_zscore(df["hype_index_equal"], window)
        for threshold in THRESHOLD_GRID:
            spikes = df["date"][z >= threshold]
            scores = evaluate(spikes, events_df, EVENT_MATCH_WINDOW_DAYS)
            if scores["f1"] > best_f1:
                best_f1, best_window, best_threshold = scores["f1"], window, threshold
    return best_window, best_threshold, best_f1


def was_event_caught(df: pd.DataFrame, event_date: pd.Timestamp, window: int,
                       threshold: float, tolerance_days: int) -> bool:
    z = compute_rolling_zscore(df["hype_index_equal"], window)
    spikes = df["date"][z >= threshold]
    return bool((abs(spikes - event_date) <= pd.Timedelta(days=tolerance_days)).any())


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])
    n_events = len(events_df)

    print(f"Running leave-one-event-out cross-validation across {n_events} events...\n")

    results = []
    for i in range(n_events):
        held_out_event = events_df.iloc[[i]]
        remaining_events = events_df.drop(events_df.index[i]).reset_index(drop=True)

        window, threshold, dev_f1 = tune_on_events(df, remaining_events)
        caught = was_event_caught(df, held_out_event["date"].iloc[0], window,
                                    threshold, EVENT_MATCH_WINDOW_DAYS)

        results.append({
            "held_out_event": held_out_event["label"].iloc[0],
            "event_type": held_out_event.get("type", pd.Series(["unknown"])).iloc[0],
            "tuned_window": window,
            "tuned_threshold": threshold,
            "dev_f1_on_remaining_20": round(dev_f1, 3),
            "held_out_event_caught": caught,
        })

    results_df = pd.DataFrame(results)
    catch_rate = results_df["held_out_event_caught"].mean()

    print(results_df.to_string(index=False))
    print(f"\nOverall LOO catch rate: {results_df['held_out_event_caught'].sum()}/{n_events} "
          f"({catch_rate:.1%})")

    print("\nCatch rate by event type:")
    by_type = results_df.groupby("event_type")["held_out_event_caught"].agg(["sum", "count"])
    by_type["rate"] = by_type["sum"] / by_type["count"]
    print(by_type.to_string())

    if catch_rate < 0.4:
        print(
            f"\nThe LOO catch rate ({catch_rate:.1%}) confirms the single 14/7 "
            f"holdout split wasn't just an unlucky division -- across 21 "
            f"independent leave-one-out trials, the detector consistently "
            f"struggles to catch an event it wasn't tuned to expect. This is "
            f"a genuine, repeated generalization limitation, not a fluke of "
            f"one split."
        )
    elif catch_rate < 0.65:
        print(
            f"\nThe LOO catch rate ({catch_rate:.1%}) is moderate -- better "
            f"than the single 14/7 split suggested, but still short of the "
            f"~76% recall seen when the detector is tuned on the full event "
            f"set. Some genuine generalization gap remains."
        )
    else:
        print(
            f"\nThe LOO catch rate ({catch_rate:.1%}) is reasonably close to "
            f"full-set performance -- the earlier 14/7 split may have been "
            f"an unusually hard specific division rather than a sign of "
            f"consistent generalization failure."
        )

    out_path = PROCESSED_DATA_DIR / "leave_one_event_out.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
