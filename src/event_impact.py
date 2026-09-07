"""
Milestone 8: Event impact analysis with bootstrapped confidence intervals.

For each verified event, measure the change in each signal from before the
event to after it -- and instead of reporting a single point estimate like
"+43%", bootstrap the weeks in each window to get a plausible range. This
directly answers the reviewer's ask: don't claim a precise causal number,
report the observed change plus how much that number would wobble under
resampling.

Honesty note: pre/post differences here are OBSERVED CHANGES coinciding
with an event, not proven causal effects of that event -- other things
happen in the same weeks too. Framed as "observed change" throughout,
never "the event caused."

Run:
    python src/event_impact.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS

WINDOW_WEEKS = 3          # weeks before/after to compare
N_BOOTSTRAP = 2000        # resamples for the confidence interval
CI_PERCENTILES = (2.5, 97.5)  # 95% CI

SIGNAL_COLUMNS = ["trends_signal", "wiki_signal", "youtube_signal"]


def load_merged_signals() -> pd.DataFrame:
    """Combine the Hype Index signals with weekly sentiment onto one timeline."""
    hype_path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    sentiment_path = PROCESSED_DATA_DIR / "youtube_weekly_sentiment.csv"

    df = pd.read_csv(hype_path, parse_dates=["date"])

    if sentiment_path.exists():
        sentiment = pd.read_csv(sentiment_path, parse_dates=["date"])
        df = df.merge(sentiment[["date", "avg_sentiment"]], on="date", how="left")
    else:
        df["avg_sentiment"] = np.nan

    return df.sort_values("date").reset_index(drop=True)


def bootstrap_mean_diff(before: np.ndarray, after: np.ndarray, n_bootstrap: int) -> tuple:
    """
    Resample (with replacement) from the before/after windows independently,
    recompute the mean difference each time, and take the 95% percentile
    range of that distribution as the confidence interval.
    """
    if len(before) == 0 or len(after) == 0:
        return np.nan, np.nan, np.nan

    observed_diff = after.mean() - before.mean()

    rng = np.random.default_rng(42)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        b_sample = rng.choice(before, size=len(before), replace=True)
        a_sample = rng.choice(after, size=len(after), replace=True)
        diffs[i] = a_sample.mean() - b_sample.mean()

    lower, upper = np.percentile(diffs, CI_PERCENTILES)
    return observed_diff, lower, upper


def analyze_event(df: pd.DataFrame, event_date: pd.Timestamp, signal_col: str) -> dict:
    before_start = event_date - pd.Timedelta(weeks=WINDOW_WEEKS)
    after_end = event_date + pd.Timedelta(weeks=WINDOW_WEEKS)

    before = df[(df["date"] >= before_start) & (df["date"] < event_date)][signal_col].dropna().values
    after = df[(df["date"] >= event_date) & (df["date"] <= after_end)][signal_col].dropna().values

    diff, lower, upper = bootstrap_mean_diff(before, after, N_BOOTSTRAP)

    return {
        "signal": signal_col,
        "before_mean": round(before.mean(), 4) if len(before) else np.nan,
        "after_mean": round(after.mean(), 4) if len(after) else np.nan,
        "observed_change": round(diff, 4) if not np.isnan(diff) else np.nan,
        "ci_95_lower": round(lower, 4) if not np.isnan(lower) else np.nan,
        "ci_95_upper": round(upper, 4) if not np.isnan(upper) else np.nan,
    }


def main():
    df = load_merged_signals()
    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])

    signals_to_test = SIGNAL_COLUMNS + (["avg_sentiment"] if df["avg_sentiment"].notna().any() else [])

    print(f"Event impact analysis ({WINDOW_WEEKS}-week windows, "
          f"{N_BOOTSTRAP} bootstrap resamples per estimate)\n")

    rows = []
    for _, event in events_df.iterrows():
        for signal_col in signals_to_test:
            result = analyze_event(df, event["date"], signal_col)
            result["event"] = event["label"]
            result["event_date"] = event["date"]
            rows.append(result)

    results_df = pd.DataFrame(rows)
    results_df = results_df[[
        "event", "event_date", "signal", "before_mean", "after_mean",
        "observed_change", "ci_95_lower", "ci_95_upper",
    ]]

    for event_label in results_df["event"].unique():
        print(f"\n{event_label}")
        subset = results_df[results_df["event"] == event_label]
        print(subset[["signal", "observed_change", "ci_95_lower", "ci_95_upper"]].to_string(index=False))

    out_path = PROCESSED_DATA_DIR / "event_impact.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    print(
        "\nReading this: 'observed_change' is the raw before/after "
        "difference. The CI range shows how much that number would wobble "
        "if we resampled the same weeks -- a wide range means the estimate "
        "isn't very stable, usually because the window has very few weeks "
        "of data. This is an observed coincidence with the event, not a "
        "proven causal effect of it."
    )


if __name__ == "__main__":
    main()
