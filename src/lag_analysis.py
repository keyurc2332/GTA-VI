"""
Milestone 9: Cross-signal lag analysis.

Reddit wasn't available, so this can't test YouTube -> Reddit -> Trends
propagation as originally hoped. What IS testable with the three signals
actually collected: does search curiosity (Trends), information-seeking
(Wikipedia), or creator activity (YouTube) tend to move first?

Method: cross-correlation at different lags. For each pair of signals, shift
one relative to the other by -N to +N weeks and find the lag with the
strongest correlation. A positive lag means the first signal leads the
second (changes happen there first); negative means it lags.

Honesty note: this reports "signal A tended to lead signal B by X weeks in
this data," never "A causes B." Weekly resolution also limits precision --
a same-week (lag 0) result doesn't rule out finer within-week ordering we
can't see at this granularity.

Run:
    python src/lag_analysis.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR

MAX_LAG_WEEKS = 4
SIGNAL_PAIRS = [
    ("trends_signal", "wiki_signal"),
    ("trends_signal", "youtube_signal"),
    ("wiki_signal", "youtube_signal"),
]
SIGNAL_LABELS = {
    "trends_signal": "Google Trends",
    "wiki_signal": "Wikipedia",
    "youtube_signal": "YouTube",
}


def cross_correlation_at_lags(a: pd.Series, b: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Positive lag k: b is shifted back by k weeks, i.e. we're testing whether
    a's value at time t predicts b's value at time t+k -- a leads b.
    Negative lag: b leads a.
    """
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            shifted_b = b.shift(-lag)
        else:
            shifted_b = b.shift(-lag)
        valid = a.notna() & shifted_b.notna()
        if valid.sum() < 10:
            corr = np.nan
        else:
            corr = a[valid].corr(shifted_b[valid])
        rows.append({"lag_weeks": lag, "correlation": corr})
    return pd.DataFrame(rows)


def interpret_best_lag(best_lag: int, signal_a_label: str, signal_b_label: str) -> str:
    if best_lag == 0:
        return f"{signal_a_label} and {signal_b_label} moved together in the same week, with no clear lead."
    elif best_lag > 0:
        return f"{signal_a_label} tended to lead {signal_b_label} by {best_lag} week(s) in this data."
    else:
        return f"{signal_b_label} tended to lead {signal_a_label} by {abs(best_lag)} week(s) in this data."


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    print(f"Cross-signal lag analysis (testing lags of -{MAX_LAG_WEEKS} to "
          f"+{MAX_LAG_WEEKS} weeks)\n")

    all_results = []
    for sig_a, sig_b in SIGNAL_PAIRS:
        label_a, label_b = SIGNAL_LABELS[sig_a], SIGNAL_LABELS[sig_b]
        lag_df = cross_correlation_at_lags(df[sig_a], df[sig_b], MAX_LAG_WEEKS)
        lag_df["pair"] = f"{label_a} vs {label_b}"
        all_results.append(lag_df)

        valid_lags = lag_df.dropna(subset=["correlation"])
        if valid_lags.empty:
            print(f"{label_a} vs {label_b}: not enough overlapping data.")
            continue

        best_row = valid_lags.loc[valid_lags["correlation"].abs().idxmax()]
        best_lag = int(best_row["lag_weeks"])
        best_corr = best_row["correlation"]

        print(f"{label_a} vs {label_b}")
        print(f"  Strongest correlation at lag {best_lag} weeks (r = {best_corr:.3f})")
        print(f"  -> {interpret_best_lag(best_lag, label_a, label_b)}\n")

    combined = pd.concat(all_results, ignore_index=True)
    out_path = PROCESSED_DATA_DIR / "lag_analysis.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    print(
        "\nReading this: a lag of 0 with weekly-resolution data doesn't mean "
        "nothing led anything -- it means whatever ordering exists happens "
        "faster than a week, which this data can't resolve. These results "
        "describe the observed pattern, not a causal mechanism."
    )


if __name__ == "__main__":
    main()
