"""
Collect GTA VI related video metadata from the YouTube Data API v3.

Two-step process (matches YouTube's own API design):
  1. search.list  -> get video IDs matching each query   (100 units/call)
  2. videos.list  -> get stats/details for those IDs      (1 unit/call, batched 50 at a time)

10 queries x 100 units = 1,000 units for search, plus ~1 unit per video for
details. Well within the 10,000/day free quota. Queries deliberately span
every known GTA VI event/era (trailer 1, trailer 2, delays, cover art,
pre-orders, extended look) rather than generic recent-leaning terms, so the
collected sample isn't skewed toward whatever's currently popular.

Requires:
    export YOUTUBE_API_KEY=your_key_here

Run:
    python src/collect_youtube.py
"""

import sys
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    YOUTUBE_API_KEY,
    YOUTUBE_SEARCH_QUERIES,
    YOUTUBE_MAX_RESULTS_PER_QUERY,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
)

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def search_video_ids(query: str) -> list:
    video_ids = []
    params = {
        "key": YOUTUBE_API_KEY,
        "q": query,
        "part": "id",
        "type": "video",
        "order": "relevance",
        "maxResults": 50,
        "publishedAfter": f"{START_DATE}T00:00:00Z",
    }
    fetched = 0
    while fetched < YOUTUBE_MAX_RESULTS_PER_QUERY:
        resp = requests.get(SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        ids = [item["id"]["videoId"] for item in data.get("items", [])]
        video_ids.extend(ids)
        fetched += len(ids)

        next_token = data.get("nextPageToken")
        if not next_token or not ids:
            break
        params["pageToken"] = next_token
        time.sleep(1)

    return video_ids[:YOUTUBE_MAX_RESULTS_PER_QUERY]


def fetch_video_details(video_ids: list) -> pd.DataFrame:
    rows = []
    # videos.list accepts up to 50 IDs per call
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        params = {
            "key": YOUTUBE_API_KEY,
            "id": ",".join(batch),
            "part": "snippet,statistics,contentDetails",
        }
        resp = requests.get(VIDEOS_URL, params=params)
        resp.raise_for_status()
        for item in resp.json().get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            rows.append({
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "description": snippet.get("description", "")[:500],
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration": content.get("duration"),
            })
        time.sleep(1)
    return pd.DataFrame(rows)


def main():
    if not YOUTUBE_API_KEY:
        print("ERROR: set YOUTUBE_API_KEY environment variable first.")
        return

    all_rows = []
    for query in YOUTUBE_SEARCH_QUERIES:
        print(f"[search] {query!r}")
        ids = search_video_ids(query)
        print(f"  found {len(ids)} video ids")
        if not ids:
            continue
        df = fetch_video_details(ids)
        df["query"] = query
        all_rows.append(df)
        time.sleep(1)

    if not all_rows:
        print("No videos collected.")
        return

    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset="video_id")
    combined["published_at"] = pd.to_datetime(combined["published_at"])
    combined = combined.sort_values("published_at")

    out_path = RAW_DATA_DIR / "youtube_videos.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(combined)} unique videos)")


if __name__ == "__main__":
    main()
