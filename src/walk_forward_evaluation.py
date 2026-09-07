"""
Milestone 21: Walk-forward evaluation.

The chronological holdout (event_holdout.py) uses ONE split. Leave-one-out
(leave_one_event_out.py) is more thorough, but has a subtle unrealism: when
testing on an EARLY event, LOO still tunes using LATER events -- future
information leaking into a decision about the past, which no real
deployed system could do.

Walk-forward fixes this. For each event, in chronological order, tune the
detector using ONLY events that happened before it, then check whether
that next event gets caught. This simulates literally standing at each
point in time with only past information available -- as close as this
project gets to "would this have worked in production."

Run:
    python src/walk_forward_evaluation.py
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
MIN_TRAIN_EVENTS = 4  # need at least this many past events before we can tune anything


def tune_on_events(df: pd.DataFrame, events_df: pd.DataFrame) -> tuple:
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
    events_df = events_df.sort_values("date").reset_index(drop=True)
    n_events = len(events_df)

    print(f"Walk-forward evaluation across {n_events} chronologically ordered events.")
    print(f"First {MIN_TRAIN_EVENTS} events skipped (not enough history yet to tune anything).\n")

    results = []
    for i in range(MIN_TRAIN_EVENTS, n_events):
        past_events = events_df.iloc[:i]  # STRICTLY only events before this one
        test_event = events_df.iloc[i]

        window, threshold, train_f1 = tune_on_events(df, past_events)
        caught = was_event_caught(df, test_event["date"], window, threshold, EVENT_MATCH_WINDOW_DAYS)

        results.append({
            "test_event": test_event["label"],
            "test_event_date": test_event["date"],
            "event_type": test_event.get("type", "unknown"),
            "n_past_events_available": i,
            "tuned_window": window,
            "tuned_threshold": threshold,
            "train_f1": round(train_f1, 3),
            "caught": caught,
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    catch_rate = results_df["caught"].mean()
    print(f"\nWalk-forward catch rate: {int(results_df['caught'].sum())}/{len(results_df)} "
          f"({catch_rate:.1%})")

    # Does performance improve as more history accumulates? Split into first
    # half vs second half of the walk-forward trials to check.
    midpoint = len(results_df) // 2
    early_rate = results_df.iloc[:midpoint]["caught"].mean() if midpoint > 0 else np.nan
    late_rate = results_df.iloc[midpoint:]["caught"].mean()
    print(f"\nCatch rate, first half of trials (less history available): {early_rate:.1%}")
    print(f"Catch rate, second half of trials (more history available): {late_rate:.1%}")

    if late_rate > early_rate + 0.15:
        print(
            "\nCatch rate improves meaningfully as more historical events "
            "accumulate -- consistent with a detector that genuinely gets "
            "more reliable with more tuning data, not one with a fixed, "
            "data-independent ceiling."
        )
    elif early_rate > late_rate + 0.15:
        print(
            "\nCatch rate is actually higher early on, when less history was "
            "available -- worth investigating whether the later test events "
            "(likely the event-dense mid-2026 stretch) are simply harder to "
            "catch regardless of how much tuning data preceded them."
        )
    else:
        print(
            "\nCatch rate doesn't show a strong trend with more accumulated "
            "history -- more past events to tune on didn't reliably make the "
            "detector better in this walk-forward simulation."
        )

    out_path = PROCESSED_DATA_DIR / "walk_forward_evaluation.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    print(
        "\nReading this: unlike leave-one-out, no test event's tuning here "
        "ever used information from AFTER that event -- this is the "
        "strictest, most realistic generalization test in the project."
    )


if __name__ == "__main__":
    main()
