"""
Milestone 2: Merge the three raw signals onto a common weekly timeline
and construct the GTA VI Hype Index (equal-weight + PCA-weighted).

Important honesty note:
    YouTube's API only gives us total view_count at time of collection,
    not day-by-day view growth. So the YouTube signal here is a creator-
    attention proxy: weekly upload volume + total eventual views of videos
    published that week. It is NOT the same as real-time view velocity.
    This limitation is documented, not hidden.

Run:
    python src/build_hype_index.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def force_datetime(df: pd.DataFrame, col: str, source_name: str) -> pd.DataFrame:
    """
    Explicitly coerce a column to datetime, dropping rows that fail to parse,
    and strip any timezone info so every source merges on naive datetimes.
    CSVs built up across multiple runs (e.g. a retried country after a 429)
    can end up mixing tz-aware and tz-naive or slightly different formats.
    """
    before = len(df)
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    dropped = df[col].isna().sum()
    if dropped:
        print(f"  [warn] {source_name}: dropped {dropped}/{before} rows with unparseable '{col}'")
    return df.dropna(subset=[col])


def load_trends_weekly() -> pd.DataFrame:
    """Average worldwide interest across all keywords, per week."""
    df = pd.read_csv(RAW_DATA_DIR / "trends_combined.csv")
    df = force_datetime(df, "date", "trends_combined.csv")
    df = df[df["geo"] == "worldwide"]
    weekly = df.groupby("date")["interest"].mean().reset_index()
    weekly = weekly.rename(columns={"interest": "trends_signal"})
    return weekly


def load_wikipedia_weekly() -> pd.DataFrame:
    """Sum daily pageviews into weekly buckets (week ending Sunday, matches Trends)."""
    df = pd.read_csv(RAW_DATA_DIR / "wikipedia_pageviews.csv")
    df = force_datetime(df, "date", "wikipedia_pageviews.csv")
    df = df.set_index("date").resample("W")["views"].sum().reset_index()
    df = df.rename(columns={"views": "wiki_signal"})
    return df


def load_youtube_weekly() -> pd.DataFrame:
    """
    Weekly YouTube signals, split into two conceptually different things
    rather than blended into one "attention" number:

      CREATOR ACTIVITY  : upload_count + unique_channels publishing that week
                          -- how much content creators are making, independent
                          of how well any of it performs
      AUDIENCE ENGAGEMENT: total views + likes of videos published that week
                          -- still limited by the API to eventual/lifetime
                          totals at collection time, not real week-of growth,
                          but at least now separated from creator activity
                          instead of averaged together into one blurred signal

    The original blended "upload_count" / "total_views_of_new_videos" columns
    are kept for backward compatibility with the existing Hype Index and
    every script built on top of it -- this ADDS the split signals alongside
    rather than replacing anything downstream depends on.
    """
    df = pd.read_csv(RAW_DATA_DIR / "youtube_videos.csv")
    df = force_datetime(df, "published_at", "youtube_videos.csv")
    df = df.set_index("published_at")

    agg_dict = {
        "upload_count": ("video_id", "count"),
        "total_views_of_new_videos": ("view_count", "sum"),
        "unique_channels": ("channel", "nunique"),
    }
    if "like_count" in df.columns:
        agg_dict["total_likes"] = ("like_count", "sum")

    weekly = df.resample("W").agg(**agg_dict).reset_index()
    weekly = weekly.rename(columns={"published_at": "date"})

    if "total_likes" not in weekly.columns:
        weekly["total_likes"] = 0

    return weekly


def normalize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def build_equal_weight_index(df: pd.DataFrame, signal_cols: list) -> pd.Series:
    return df[signal_cols].mean(axis=1)


def build_pca_weight_index(df: pd.DataFrame, signal_cols: list) -> pd.Series:
    pca = PCA(n_components=1)
    component = pca.fit_transform(df[signal_cols])
    # PCA sign is arbitrary — flip if the component is negatively correlated with equal-weight
    series = pd.Series(component.flatten(), index=df.index)
    return series


def main():
    print("Loading and resampling weekly signals...")
    trends = load_trends_weekly()
    wiki = load_wikipedia_weekly()
    youtube = load_youtube_weekly()

    print(f"  trends:   {len(trends)} weeks, {trends['date'].min().date()} -> {trends['date'].max().date()}")
    print(f"  wiki:     {len(wiki)} weeks, {wiki['date'].min().date()} -> {wiki['date'].max().date()}")
    print(f"  youtube:  {len(youtube)} weeks, {youtube['date'].min().date()} -> {youtube['date'].max().date()}")

    # Merge on week -- outer join so we don't silently drop weeks where
    # one source has no data (e.g. no video uploads that week)
    merged = trends.merge(wiki, on="date", how="outer")
    merged = merged.merge(youtube, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    # Weeks with no YouTube uploads / no wiki data genuinely mean zero attention
    # that week, not missing data -- fill with 0 rather than interpolating
    fill_cols = ["trends_signal", "wiki_signal", "upload_count", "total_views_of_new_videos",
                 "unique_channels", "total_likes"]
    merged[fill_cols] = merged[fill_cols].fillna(0)

    # Preserve raw values BEFORE any normalization overwrites them in place --
    # needed below for the log-transformed audience-engagement signal.
    raw_views = merged["total_views_of_new_videos"].copy()
    raw_likes = merged["total_likes"].copy()

    # Combine YouTube's two sub-signals into one normalized youtube_signal first
    # (kept exactly as before -- every downstream script depends on this column)
    merged = normalize(merged, ["upload_count", "total_views_of_new_videos"])
    merged["youtube_signal"] = merged[["upload_count", "total_views_of_new_videos"]].mean(axis=1)

    # NEW: split YouTube into creator-activity vs audience-engagement signals,
    # ADDED alongside the original blended signal rather than replacing it.
    merged = normalize(merged, ["unique_channels"])
    # upload_count was already normalized above; reuse it directly
    merged["youtube_creator_signal"] = merged[["upload_count", "unique_channels"]].mean(axis=1)

    # log1p on the engagement metrics from the PRESERVED raw values, since
    # view/like counts are heavily right-skewed (one viral video can dwarf
    # everything else) -- log-scaling keeps a single outlier from dominating.
    merged["log_views"] = np.log1p(raw_views)
    merged["log_likes"] = np.log1p(raw_likes)
    merged = normalize(merged, ["log_views", "log_likes"])
    merged["youtube_audience_signal"] = merged[["log_views", "log_likes"]].mean(axis=1)

    # Normalize the three top-level signals for the index
    signal_cols = ["trends_signal", "wiki_signal", "youtube_signal"]
    merged = normalize(merged, signal_cols)

    merged["hype_index_equal"] = build_equal_weight_index(merged, signal_cols)
    merged["hype_index_pca"] = build_pca_weight_index(merged, signal_cols)

    # Align PCA sign with equal-weight index for interpretability
    if merged["hype_index_pca"].corr(merged["hype_index_equal"]) < 0:
        merged["hype_index_pca"] *= -1
    merged["hype_index_pca"] = MinMaxScaler().fit_transform(merged[["hype_index_pca"]])

    out_path = PROCESSED_DATA_DIR / "hype_index_weekly.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(merged)} weeks)")

    print("\nTop 5 weeks by equal-weight Hype Index:")
    top5 = merged.sort_values("hype_index_equal", ascending=False).head(5)
    print(top5[["date", "hype_index_equal", "hype_index_pca"]].to_string(index=False))

    # --- Robustness check ---
    # An index is only trustworthy if the conclusion doesn't depend on which
    # weighting scheme happened to be picked. Rather than asserting "PCA is
    # better," check whether the two disagree in ways that would matter.
    correlation = merged["hype_index_equal"].corr(merged["hype_index_pca"])
    top10_equal = set(merged.sort_values("hype_index_equal", ascending=False).head(10)["date"])
    top10_pca = set(merged.sort_values("hype_index_pca", ascending=False).head(10)["date"])
    overlap = len(top10_equal & top10_pca)

    print(f"\nRobustness check:")
    print(f"  Correlation between equal-weight and PCA-weighted index: {correlation:.3f}")
    print(f"  Overlap in top-10 highest-hype weeks: {overlap}/10")
    if correlation > 0.85 and overlap >= 7:
        print("  -> The two weighting schemes largely agree. The conclusions "
              "below aren't an artifact of one particular weighting choice.")
    else:
        print("  -> The two weighting schemes disagree meaningfully. Treat "
              "any single-index conclusion with caution and report both.")


if __name__ == "__main__":
    main()
