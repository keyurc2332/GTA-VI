"""
Milestone 18: Event response curves.

event_impact.py answers "how much did attention change, before vs after?"
with a single before/after comparison. This goes further: it builds the
full week-by-week trajectory from -4 to +4 weeks around EVERY event, then
averages those trajectories by event TYPE (official/leak/rumor/controversy/
marketing) to test a real hypothesis -- do different kinds of events
produce different attention SHAPES, not just different sizes?

For example: does a trailer create a sharp, short-lived spike while a delay
announcement creates a slower, more sustained response? That's a much more
interesting finding than "X happened, attention went up."

Honesty note: these are observed average shapes across a modest number of
events per type (some types have very few events) -- read them as
suggestive patterns, not statistically confirmed archetypes.

Run:
    python src/event_response_curves.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS

WINDOW_WEEKS = 4  # weeks before/after to trace


def build_response_curve(df: pd.DataFrame, event_date: pd.Timestamp, signal_col: str) -> np.ndarray:
    """
    Extract a -WINDOW_WEEKS to +WINDOW_WEEKS trajectory around an event,
    normalized to the pre-event baseline (mean of the weeks before the
    event) so events with different absolute signal levels can be
    compared on the same shape-relative scale.
    """
    curve = np.full(2 * WINDOW_WEEKS + 1, np.nan)
    for offset in range(-WINDOW_WEEKS, WINDOW_WEEKS + 1):
        target_date = event_date + pd.Timedelta(weeks=offset)
        # find the closest week within 3 days
        matches = df[(df["date"] - target_date).abs() <= pd.Timedelta(days=3)]
        if not matches.empty:
            curve[offset + WINDOW_WEEKS] = matches[signal_col].iloc[0]

    baseline = np.nanmean(curve[:WINDOW_WEEKS]) if not np.all(np.isnan(curve[:WINDOW_WEEKS])) else np.nan
    if baseline and baseline > 1e-9:
        curve = curve / baseline  # relative to pre-event baseline (1.0 = no change)
    return curve


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    offsets = list(range(-WINDOW_WEEKS, WINDOW_WEEKS + 1))
    all_curves = []

    for _, event in events_df.iterrows():
        curve = build_response_curve(df, event["date"], "hype_index_equal")
        if np.all(np.isnan(curve)):
            continue
        for offset, value in zip(offsets, curve):
            all_curves.append({
                "event": event["label"],
                "type": event.get("type", "unknown"),
                "offset_weeks": offset,
                "relative_hype": value,
            })

    curves_df = pd.DataFrame(all_curves)

    if curves_df.empty:
        print("No events had enough surrounding data to build a response curve.")
        return

    print(f"Built response curves for {curves_df['event'].nunique()} events.\n")

    # Average curve by event type
    type_avg = curves_df.groupby(["type", "offset_weeks"])["relative_hype"].mean().reset_index()
    type_counts = curves_df.groupby("type")["event"].nunique()

    print("Average relative attention by event type (1.0 = pre-event baseline):\n")
    for event_type in type_avg["type"].unique():
        n = type_counts.get(event_type, 0)
        subset = type_avg[type_avg["type"] == event_type].sort_values("offset_weeks")
        print(f"{event_type} (n={n} events):")
        print(subset[["offset_weeks", "relative_hype"]].to_string(index=False))
        print()

    # Characterize each type's shape: peak height and how fast it decays
    print("Shape summary per event type:")
    for event_type in type_avg["type"].unique():
        subset = type_avg[type_avg["type"] == event_type].sort_values("offset_weeks")
        peak_offset = subset.loc[subset["relative_hype"].idxmax(), "offset_weeks"]
        peak_value = subset["relative_hype"].max()
        # decay: value 2 weeks after peak vs peak itself
        post_peak = subset[subset["offset_weeks"] == min(peak_offset + 2, WINDOW_WEEKS)]
        decay_value = post_peak["relative_hype"].iloc[0] if not post_peak.empty else np.nan

        print(f"  {event_type}: peaks at week {int(peak_offset)} "
              f"({peak_value:.2f}x baseline), "
              f"{decay_value:.2f}x baseline two weeks after peak" if not np.isnan(decay_value)
              else f"  {event_type}: peaks at week {int(peak_offset)} ({peak_value:.2f}x baseline)")

    curves_df.to_csv(PROCESSED_DATA_DIR / "event_response_curves.csv", index=False)
    type_avg.to_csv(PROCESSED_DATA_DIR / "event_response_by_type.csv", index=False)
    print(f"\nSaved 2 files -> {PROCESSED_DATA_DIR}")

    print(
        "\nReading this: some event types have very few examples (a single "
        "controversy event, for instance), so their 'average shape' is really "
        "just that one event's shape -- read those as descriptive, not a "
        "validated pattern for that category in general."
    )


if __name__ == "__main__":
    main()
