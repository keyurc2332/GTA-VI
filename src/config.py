"""
Central configuration for the GTA VI Hype Intelligence project.
Keep every tunable constant here so the collection scripts stay reproducible.
"""

import os
from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Date range ----
# Covers first trailer (Dec 2023) through the recent Extended Look / Netflix moment
START_DATE = "2023-01-01"
END_DATE = "2026-09-07"  # update to "today" when you re-run

# ---- Google Trends ----
TRENDS_KEYWORDS = ["GTA 6", "GTA VI", "Grand Theft Auto VI"]
# Expanded from 5 to 25 countries for meaningful trajectory clustering --
# 5 observations wasn't enough to cluster on. Picked markets where Google
# Trends has reliable coverage (skips mainland China, where Google isn't
# the dominant search engine and Trends data is sparse/unreliable).
# Runtime warning: 26 geos (25 + worldwide) x 3 keywords = 78 requests.
# At the delay below, that's ~15+ minutes and some 429s are likely --
# the collection script already skips completed files on re-run, so
# just re-run it if it gets rate-limited partway through.
TRENDS_COUNTRIES = [
    "", "US", "GB", "IN", "BR", "MX", "CA", "AU", "DE", "FR", "JP", "KR",
    "ID", "PH", "IT", "ES", "NL", "PL", "TR", "AR", "ZA", "SA", "TH",
    "VN", "MY", "SG",
]

# ---- YouTube ----
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")
# Queries deliberately span every known era of GTA VI news, not just recent
# content -- a plain "GTA 6 trailer" search is recency-biased and will
# under-sample Dec 2023 in favor of whatever's popular today. Each query
# targets a specific real event so the collected sample actually covers
# the full timeline instead of just the most recent moment.
YOUTUBE_SEARCH_QUERIES = [
    "GTA 6 trailer 1",
    "GTA 6 trailer 2",
    "GTA VI gameplay",
    "GTA 6 extended look",
    "GTA 6 delayed",
    "GTA 6 delay November 2026",
    "GTA 6 cover art",
    "GTA 6 pre order",
    "GTA 6 leaked gameplay",
    "GTA 6 release date",
]
YOUTUBE_MAX_RESULTS_PER_QUERY = 50  # 1 search call = 100 units, returns up to 50 items

# ---- Wikipedia ----
WIKI_ARTICLE = "Grand_Theft_Auto_VI"
WIKI_PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)