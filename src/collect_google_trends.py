"""
Collect Google Trends data for GTA VI search terms.

Pulls:
  - Worldwide interest over time, per keyword
  - Country-level interest over time, per keyword

Saves raw CSVs to data/raw/trends_<keyword>_<geo>.csv so we never
have to re-hit the (rate-limited) API while iterating downstream.

Run:
    python src/collect_google_trends.py
"""

import time
import sys
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRENDS_KEYWORDS,
    TRENDS_COUNTRIES,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
)

REQUEST_DELAY_SECONDS = 8  # be polite, avoid 429s -- bumped up since we now
                            # collect ~78 requests instead of ~18


def safe_filename(keyword: str, geo: str) -> str:
    kw = keyword.lower().replace(" ", "_")
    geo_part = geo if geo else "worldwide"
    return f"trends_{kw}_{geo_part}.csv"


def fetch_one(pytrends: TrendReq, keyword: str, geo: str) -> pd.DataFrame:
    timeframe = f"{START_DATE} {END_DATE}"
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df.empty:
        return df
    df = df.drop(columns=["isPartial"], errors="ignore")
    df = df.rename(columns={keyword: "interest"})
    df["keyword"] = keyword
    df["geo"] = geo if geo else "worldwide"
    df = df.reset_index()
    # Force the date column to plain string form immediately, matching the
    # format it has when re-loaded from a saved CSV. Without this, a freshly
    # fetched frame (Timestamp dtype) concatenated with skip-loaded frames
    # (string dtype, from a prior run's CSV) ends up with a mixed-format
    # 'date' column once saved and reloaded, causing downstream scripts to
    # silently drop rows that fail to parse.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    pytrends = TrendReq(hl="en-US", tz=0)
    all_frames = []

    for keyword in TRENDS_KEYWORDS:
        for geo in TRENDS_COUNTRIES:
            out_path = RAW_DATA_DIR / safe_filename(keyword, geo)
            if out_path.exists():
                print(f"[skip] already have {out_path.name}")
                df = pd.read_csv(out_path)
                all_frames.append(df)
                continue

            print(f"[fetch] {keyword!r} | geo={geo or 'worldwide'}")
            try:
                df = fetch_one(pytrends, keyword, geo)
            except Exception as e:
                print(f"  !! failed: {e}")
                time.sleep(REQUEST_DELAY_SECONDS * 2)
                continue

            if df.empty:
                print("  (no data returned)")
            else:
                df.to_csv(out_path, index=False)
                print(f"  saved -> {out_path.name} ({len(df)} rows)")
                all_frames.append(df)

            time.sleep(REQUEST_DELAY_SECONDS)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = RAW_DATA_DIR / "trends_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined dataset -> {combined_path} ({len(combined)} rows)")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
