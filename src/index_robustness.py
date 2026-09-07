"""
Milestone 10: Hype Index robustness analysis.

The Hype Index combines three signals, but there's no ground truth for how
they should be weighted. Rather than picking one weighting and hoping it's
right, this compares THREE methods head-to-head:

  EQUAL WEIGHT : average of the three normalized signals
  PCA WEIGHT   : weights derived from shared variance across signals
  RANK WEIGHT  : average of each signal's percentile RANK, not its raw
                 value -- robust to outliers and to one signal happening
                 to have a wider numeric range than the others

For each method, this measures:
  - Correlation and rank correlation with the other two methods
  - Top-10 week overlap
  - Precision/recall/F1 against the verified event list (reusing the
    same evaluation framework as evaluate_detection.py)

If all three methods broadly agree, the conclusions aren't an artifact of
one particular aggregation choice. If they disagree, that's a real,
reportable limitation of the Hype Index itself -- not something to hide.

Run:
    python src/index_robustness.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import (
    KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD,
    compute_rolling_zscore,
)
from evaluate_detection import evaluate

SIGNAL_COLS = ["trends_signal", "wiki_signal", "youtube_signal"]
EVENTS_DF = pd.DataFrame(KNOWN_EVENTS)
EVENTS_DF["date"] = pd.to_datetime(EVENTS_DF["date"])


def build_rank_index(df: pd.DataFrame, signal_cols: list) -> pd.Series:
    """
    Average percentile rank across signals, rescaled to 0-1. Robust to a
    signal having a wider or skewed numeric range than the others, since
    rank only cares about ORDER within each signal, not magnitude.
    """
    ranks = df[signal_cols].rank(pct=True)
    return ranks.mean(axis=1)


def detect_spikes_on(df: pd.DataFrame, signal_col: str) -> pd.Series:
    z = compute_rolling_zscore(df[signal_col], ROLLING_WINDOW)
    return df["date"][z >= Z_SCORE_THRESHOLD]


def main():
    path = PROCESSED_DATA_DIR / "hype_index_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df["hype_index_rank"] = build_rank_index(df, SIGNAL_COLS)
    # hype_index_equal and hype_index_pca should already exist from
    # build_hype_index.py -- if not, this script can't compare them
    for col in ["hype_index_equal", "hype_index_pca"]:
        if col not in df.columns:
            print(f"ERROR: '{col}' not found. Run build_hype_index.py first.")
            return

    methods = {
        "Equal-weight": "hype_index_equal",
        "PCA-weighted": "hype_index_pca",
        "Rank-based": "hype_index_rank",
    }

    print("=== Pairwise agreement between methods ===\n")
    method_names = list(methods.keys())
    agreement_rows = []
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name_a, name_b = method_names[i], method_names[j]
            col_a, col_b = methods[name_a], methods[name_b]

            pearson_corr = df[col_a].corr(df[col_b])
            spearman_corr, _ = spearmanr(df[col_a], df[col_b])

            top10_a = set(df.sort_values(col_a, ascending=False).head(10)["date"])
            top10_b = set(df.sort_values(col_b, ascending=False).head(10)["date"])
            overlap = len(top10_a & top10_b)

            agreement_rows.append({
                "pair": f"{name_a} vs {name_b}",
                "pearson_r": round(pearson_corr, 3),
                "spearman_rho": round(spearman_corr, 3),
                "top10_overlap": f"{overlap}/10",
            })

    agreement_df = pd.DataFrame(agreement_rows)
    print(agreement_df.to_string(index=False))

    print("\n=== Event-detection quality per method ===\n")
    eval_rows = []
    for name, col in methods.items():
        spikes = detect_spikes_on(df, col)
        scores = evaluate(spikes, EVENTS_DF, EVENT_MATCH_WINDOW_DAYS)
        eval_rows.append({"method": name, **scores})

    eval_df = pd.DataFrame(eval_rows)
    print(eval_df.to_string(index=False))

    best_method = eval_df.loc[eval_df["f1"].idxmax(), "method"]
    worst_method = eval_df.loc[eval_df["f1"].idxmin(), "method"]

    avg_overlap = np.mean([int(r["top10_overlap"].split("/")[0]) for r in agreement_rows])

    print(f"\n{best_method} scored highest (F1={eval_df['f1'].max():.3f}), "
          f"{worst_method} scored lowest (F1={eval_df['f1'].min():.3f}) on "
          f"this event set.")

    if avg_overlap >= 7:
        print(f"Average top-10 overlap across method pairs: {avg_overlap:.1f}/10 "
              f"-- the methods broadly agree on which weeks mattered most.")
    else:
        print(f"Average top-10 overlap across method pairs: {avg_overlap:.1f}/10 "
              f"-- meaningful disagreement between methods. Which weeks look "
              f"like the biggest hype moments genuinely depends on how you "
              f"choose to aggregate the signals. That's a real limitation "
              f"of the Hype Index, not just a footnote.")

    df[["date"] + list(methods.values())].to_csv(
        PROCESSED_DATA_DIR / "index_robustness_series.csv", index=False
    )
    agreement_df.to_csv(PROCESSED_DATA_DIR / "index_robustness_agreement.csv", index=False)
    eval_df.to_csv(PROCESSED_DATA_DIR / "index_robustness_evaluation.csv", index=False)
    print(f"\nSaved 3 files -> {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
