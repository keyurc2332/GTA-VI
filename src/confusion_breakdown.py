"""
Milestone 22: Explicit detection/attribution breakdown.

Every evaluation so far reports precision/recall/F1 as summary numbers.
This makes the underlying reality explicit and readable:

  HITS            : known events that a detected spike caught
  MISSES          : known events with no nearby detected spike at all
  FALSE POSITIVES : detected spikes that don't correspond to any known event

This is the same information F1 summarizes, but as three concrete lists
instead of one abstract number -- easier to actually read and reason about.

Run:
    python src/confusion_breakdown.py
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


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    z = compute_rolling_zscore(df["hype_index_equal"], ROLLING_WINDOW)
    detected_spikes = df["date"][z >= Z_SCORE_THRESHOLD].reset_index(drop=True)

    print(f"Using the project's default detector (window={ROLLING_WINDOW}, "
          f"threshold={Z_SCORE_THRESHOLD}) against {len(events_df)} evaluable events.\n")

    # Hits and misses -- for each event, is there a spike within tolerance?
    hits, misses = [], []
    for _, event in events_df.iterrows():
        matches = detected_spikes[(detected_spikes - event["date"]).abs() <= pd.Timedelta(days=EVENT_MATCH_WINDOW_DAYS)]
        if len(matches):
            hits.append({
                "event": event["label"], "event_date": event["date"],
                "event_type": event.get("type", "unknown"),
                "matched_spike_week": matches.iloc[0],
                "days_offset": (matches.iloc[0] - event["date"]).days,
            })
        else:
            misses.append({
                "event": event["label"], "event_date": event["date"],
                "event_type": event.get("type", "unknown"),
            })

    # False positives -- detected spikes with NO nearby event at all
    false_positives = []
    for spike_date in detected_spikes:
        matches = events_df[(events_df["date"] - spike_date).abs() <= pd.Timedelta(days=EVENT_MATCH_WINDOW_DAYS)]
        if matches.empty:
            false_positives.append({"spike_week": spike_date})

    hits_df = pd.DataFrame(hits)
    misses_df = pd.DataFrame(misses)
    fp_df = pd.DataFrame(false_positives)

    print(f"HITS: {len(hits_df)} known events had a matching detected spike\n")
    if not hits_df.empty:
        print(hits_df.to_string(index=False))

    print(f"\nMISSES: {len(misses_df)} known events had NO detected spike nearby\n")
    if not misses_df.empty:
        print(misses_df.to_string(index=False))
        print("\nMisses by type:")
        print(misses_df["event_type"].value_counts().to_string())

    print(f"\nFALSE POSITIVES: {len(fp_df)} detected spikes matched no known event\n")
    if not fp_df.empty:
        print(fp_df.to_string(index=False))

    total_events = len(events_df)
    total_spikes = len(detected_spikes)
    print(f"\nSummary: {len(hits_df)}/{total_events} events caught, "
          f"{len(misses_df)}/{total_events} events missed, "
          f"{len(fp_df)}/{total_spikes} detected spikes were false positives "
          f"(unexplained by any known event -- some of these may be real "
          f"events not yet in the registry, others may be noise).")

    hits_df.to_csv(PROCESSED_DATA_DIR / "confusion_hits.csv", index=False)
    misses_df.to_csv(PROCESSED_DATA_DIR / "confusion_misses.csv", index=False)
    fp_df.to_csv(PROCESSED_DATA_DIR / "confusion_false_positives.csv", index=False)
    print(f"\nSaved 3 files -> {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
