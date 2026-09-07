"""
Milestone 14: Sentiment disagreement investigation.

VADER and the transformer model agree on a modest fraction of videos. Rather
than treating that number as a final result, this looks at WHERE they
disagree and checks for a specific hypothesis: mixed-sentiment or
contrastive language (a title that's part excited, part negative, joined
by words like "but"/"however") is a natural source of disagreement, since
VADER and transformers can weight the clauses differently.

This doesn't prove that hypothesis -- it's a lightweight keyword check, not
a linguistic analysis -- but it turns a bare percentage into an actual
investigation with concrete examples to point to.

Run:
    python src/sentiment_disagreement.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR

CONTRAST_MARKERS = re.compile(
    r"\b(?:but|however|although|though|yet|despite|still)\b", flags=re.IGNORECASE
)


def main():
    path = PROCESSED_DATA_DIR / "youtube_topics_sentiment.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run analyze_youtube_text.py first.")
        return

    df = pd.read_csv(path)

    disagreements = df[~df["sentiment_agree"]].copy()
    agreements = df[df["sentiment_agree"]].copy()

    print(f"Total videos: {len(df)}")
    print(f"Agreements: {len(agreements)} ({len(agreements)/len(df):.1%})")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(df):.1%})\n")

    if disagreements.empty:
        print("No disagreements to investigate.")
        return

    disagreements["has_contrast_marker"] = disagreements["title"].fillna("").str.contains(
        CONTRAST_MARKERS
    )
    agreements["has_contrast_marker"] = agreements["title"].fillna("").str.contains(
        CONTRAST_MARKERS
    )

    disagree_contrast_rate = disagreements["has_contrast_marker"].mean()
    agree_contrast_rate = agreements["has_contrast_marker"].mean()

    print(f"Contrast-word rate (but/however/although/etc.) in titles:")
    print(f"  Disagreement videos: {disagree_contrast_rate:.1%}")
    print(f"  Agreement videos:    {agree_contrast_rate:.1%}")

    if disagree_contrast_rate > agree_contrast_rate * 1.3:
        print(
            "\nDisagreements contain contrast language noticeably more often "
            "than agreements do. This is consistent with (but doesn't prove) "
            "the idea that mixed-sentiment titles are a real source of "
            "disagreement between the two models -- they may be weighting "
            "the positive and negative clauses differently."
        )
    else:
        print(
            "\nContrast language doesn't appear meaningfully more common in "
            "disagreements than agreements. The disagreement rate likely has "
            "a different explanation -- possibly gaming slang, sarcasm, or "
            "short titles VADER and the transformer model just parse "
            "differently, not necessarily mixed sentiment specifically."
        )

    # Breakdown of what kinds of disagreement are most common
    print("\nDisagreement pattern breakdown (VADER -> transformer):")
    pattern_counts = disagreements.groupby(["vader_label", "transformer_label"]).size()
    pattern_counts = pattern_counts.sort_values(ascending=False)
    print(pattern_counts.to_string())

    # Save concrete examples for manual reading -- this is the part a human
    # should actually look at, not just the aggregate percentage
    example_cols = ["title", "vader_label", "vader_compound", "transformer_label", "has_contrast_marker"]
    examples = disagreements[example_cols].head(30)
    out_path = PROCESSED_DATA_DIR / "sentiment_disagreement_examples.csv"
    examples.to_csv(out_path, index=False)
    print(f"\nSaved 30 concrete disagreement examples -> {out_path}")
    print("Worth actually reading a handful of these rather than just citing "
          "the agreement percentage -- the specific examples are the real "
          "finding here, not the number.")

    summary = pd.DataFrame([{
        "total_videos": len(df),
        "agreement_rate": round(len(agreements) / len(df), 4),
        "disagree_contrast_rate": round(disagree_contrast_rate, 4),
        "agree_contrast_rate": round(agree_contrast_rate, 4),
    }])
    summary_path = PROCESSED_DATA_DIR / "sentiment_disagreement_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()
