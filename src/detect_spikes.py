"""
Milestone 3: Automatic spike detection + event attribution.

Rather than manually deciding "the trailer caused a spike," we let the data
flag unusual weeks first (via rolling z-score), then separately check which
of those weeks line up with known GTA VI events. This keeps the causal
claim honest: "detected spike coincided with X" rather than "X caused Y."

Run:
    python src/detect_spikes.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR, START_DATE

ROLLING_WINDOW = 8       # weeks used to compute rolling mean/std
Z_SCORE_THRESHOLD = 1.5  # weeks above this are flagged as spikes
EVENT_MATCH_WINDOW_DAYS = 10  # how close a detected spike must be to a known event

# Known GTA VI events -- loaded from a structured registry (data/events.csv)
# rather than kept inline, so each event carries a type (official/leak/
# rumor/controversy/marketing), a named source, and a confidence rating.
# Verified against real reporting (Netflix Tudum, Rockstar Newswire, Variety,
# Game Developer, GTA Wiki, CBC, GameInformer, Dexerto) rather than guessed.
# Expanded from an initial 7 to 21 -- stopped there rather than padding to
# an arbitrary round number with weakly-sourced rumors; every entry has a
# named source and an honest confidence rating (low/medium/high) reflecting
# how solid that source actually is.
EVENTS_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "events.csv"


def load_known_events() -> list:
    """
    Load the event registry, filtered to only events within the observable
    signal window (START_DATE onward). Events before that date exist in
    events.csv for historical documentation, but the detector has zero
    signal data to catch them with -- including them in evaluation would
    guarantee misses regardless of detector quality, artificially deflating
    recall/F1 for every downstream evaluation script.
    """
    if not EVENTS_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Event registry not found at {EVENTS_REGISTRY_PATH}. "
            f"This file should ship with the project -- check it wasn't "
            f"accidentally left out of the copy."
        )
    events_df = pd.read_csv(EVENTS_REGISTRY_PATH)
    events_df["date"] = pd.to_datetime(events_df["date"])

    signal_start = pd.to_datetime(START_DATE)
    before_count = len(events_df)
    in_window = events_df[events_df["date"] >= signal_start].copy()
    excluded_count = before_count - len(in_window)

    if excluded_count > 0:
        excluded = events_df[events_df["date"] < signal_start]
        print(
            f"[events] Excluded {excluded_count} event(s) before the signal "
            f"window ({START_DATE}) from evaluation -- the detector has no "
            f"data to catch them with: "
            f"{', '.join(excluded['label'].tolist())}. Kept in events.csv "
            f"for historical context, not used for scoring."
        )

    in_window["date"] = in_window["date"].dt.strftime("%Y-%m-%d")
    return in_window.to_dict("records")


KNOWN_EVENTS = load_known_events()


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


def attribute_events(spike_dates: pd.Series, events: list) -> pd.DataFrame:
    events_df = pd.DataFrame(events)
    events_df["date"] = pd.to_datetime(events_df["date"])

    rows = []
    for spike_date in spike_dates:
        matches = events_df[
            (events_df["date"] - spike_date).abs() <= pd.Timedelta(days=EVENT_MATCH_WINDOW_DAYS)
        ]
        if len(matches):
            for _, m in matches.iterrows():
                rows.append({
                    "spike_week": spike_date,
                    "matched_event": m["label"],
                    "event_date": m["date"],
                    "days_offset": (spike_date - m["date"]).days,
                })
        else:
            rows.append({
                "spike_week": spike_date,
                "matched_event": "UNEXPLAINED",
                "event_date": pd.NaT,
                "days_offset": np.nan,
            })
    return pd.DataFrame(rows)


def main():
    path = PROCESSED_DATA_DIR / "hype_index_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["z_score"] = compute_rolling_zscore(df["hype_index_equal"], ROLLING_WINDOW)
    df["is_spike"] = df["z_score"] >= Z_SCORE_THRESHOLD

    spikes = df[df["is_spike"]].copy()
    print(f"Detected {len(spikes)} spike weeks out of {len(df)} total weeks "
          f"(z >= {Z_SCORE_THRESHOLD}, rolling window = {ROLLING_WINDOW} weeks)\n")

    if spikes.empty:
        print("No spikes detected -- consider lowering Z_SCORE_THRESHOLD.")
        return

    attribution = attribute_events(spikes["date"], KNOWN_EVENTS)

    print("Spike weeks and their attribution:")
    print(attribution.to_string(index=False))

    unexplained = attribution[attribution["matched_event"] == "UNEXPLAINED"]
    explained = attribution[attribution["matched_event"] != "UNEXPLAINED"]

    print(f"\n{len(explained)} spikes matched a known event.")
    print(f"{len(unexplained)} spikes are UNEXPLAINED -- these are the genuinely "
          f"interesting ones worth investigating manually (check news/social media "
          f"for that week).")

    out_path = PROCESSED_DATA_DIR / "spike_attribution.csv"
    attribution.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Also save the full dataframe with z-scores for use in the dashboard later
    full_out_path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df.to_csv(full_out_path, index=False)
    print(f"Saved -> {full_out_path}")


if __name__ == "__main__":
    main()
