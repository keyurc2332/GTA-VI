"""
Milestone 7: Evaluation framework.

The question this answers: does combining three signals into a Hype Index
actually detect real events better than a single signal alone, or was
combining them pointless complexity?

Two detectors are run head-to-head against the same verified event list:
  BASELINE : rolling z-score spike detection on Google Trends alone
  COMBINED : rolling z-score spike detection on the full Hype Index

Both are scored with precision / recall / F1 using the same tolerance
window, so the comparison is apples-to-apples.

Honesty note: with a modest number of verified events, precision/recall here are
informative but not statistically powerful -- treat these as a directional
comparison between the two methods, not a rigorously significant result.
A larger verified-event corpus would be needed for that.

Run:
    python src/evaluate_detection.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD

EVENTS_DF = pd.DataFrame(KNOWN_EVENTS)
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std = series.rolling(window=window, min_periods=3).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def detect_spikes(df: pd.DataFrame, signal_col: str) -> pd.Series:
    z = compute_rolling_zscore(df[signal_col], ROLLING_WINDOW)
    return df["date"][z >= Z_SCORE_THRESHOLD]


def evaluate(detected_dates: pd.Series, events_df: pd.DataFrame, tolerance_days: int) -> dict:
    """
    Precision: of the weeks we flagged, how many were near a real event?
    Recall:    of the real events, how many did we successfully flag?
    """
    detected_dates = pd.Series(detected_dates).reset_index(drop=True)

    # Precision -- for each detected spike, is there a real event within tolerance?
    true_positives_detected = 0
    for d in detected_dates:
        if (abs(events_df["date"] - d) <= pd.Timedelta(days=tolerance_days)).any():
            true_positives_detected += 1
    precision = true_positives_detected / len(detected_dates) if len(detected_dates) else 0.0

    # Recall -- for each real event, was it caught by at least one detected spike?
    events_caught = 0
    for e in events_df["date"]:
        if (abs(detected_dates - e) <= pd.Timedelta(days=tolerance_days)).any():
            events_caught += 1
    recall = events_caught / len(events_df) if len(events_df) else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "spikes_detected": len(detected_dates),
        "events_total": len(events_df),
        "events_caught": events_caught,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Evaluating against {len(EVENTS_DF)} verified events "
          f"(tolerance = +/-{EVENT_MATCH_WINDOW_DAYS} days)\n")

    baseline_spikes = detect_spikes(df, "trends_signal")
    combined_spikes = detect_spikes(df, "hype_index_equal")

    baseline_scores = evaluate(baseline_spikes, EVENTS_DF, EVENT_MATCH_WINDOW_DAYS)
    combined_scores = evaluate(combined_spikes, EVENTS_DF, EVENT_MATCH_WINDOW_DAYS)

    results = pd.DataFrame([
        {"method": "Baseline (Trends only)", **baseline_scores},
        {"method": "Combined (Hype Index)", **combined_scores},
    ])

    print(results.to_string(index=False))

    print(
        f"\nNote: {len(EVENTS_DF)} events is a modest sample for these numbers to "
        "be statistically decisive. Read this as 'which method looks more "
        "promising on the events we do have,' not a proven result."
    )

    if combined_scores["f1"] > baseline_scores["f1"]:
        print(
            "\nOn this data, combining three signals into the Hype Index "
            "caught real events more reliably than Google Trends alone."
        )
    elif combined_scores["f1"] < baseline_scores["f1"]:
        print(
            "\nOn this data, the single-signal baseline actually performed "
            "as well or better than the combined index -- worth being "
            "honest about rather than assuming more signals is always better."
        )
    else:
        print("\nBoth methods performed identically on this event set.")

    out_path = PROCESSED_DATA_DIR / "detection_evaluation.csv"
    results.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
