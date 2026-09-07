"""
Collect daily Wikipedia pageviews for the GTA VI article.
No API key required.

Run:
    python src/collect_wikipedia.py
"""

import sys
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import WIKI_ARTICLE, WIKI_PAGEVIEWS_URL, START_DATE, END_DATE, RAW_DATA_DIR

HEADERS = {"User-Agent": "gta6-hype-project (personal data science weekend project)"}


def date_to_wiki_format(date_str: str) -> str:
    return date_str.replace("-", "")


def main():
    url = WIKI_PAGEVIEWS_URL.format(
        article=WIKI_ARTICLE,
        start=date_to_wiki_format(START_DATE),
        end=date_to_wiki_format(END_DATE),
    )
    print(f"[fetch] {url}")

    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    if not items:
        print("No pageview data returned.")
        return

    df = pd.DataFrame(items)
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
    df = df[["date", "views", "article"]]

    out_path = RAW_DATA_DIR / "wikipedia_pageviews.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} ({len(df)} rows, "
          f"{df['date'].min().date()} to {df['date'].max().date()})")


if __name__ == "__main__":
    main()
