# -*- coding: utf-8 -*-
"""
This file centralizes all the configuration for the news aggregator.
"""

from datetime import datetime, timedelta

# --- SEARCH CONFIGURATION ---
MAX_RESULTS_TO_DISPLAY = 30
NUM_VARIANTS_TO_GENERATE = 6  # More variants to improve recall

# Dynamic date range for NewsAPI search (last 21 days)
DAYS_AGO = 21
NEWSAPI_TO_DATE = datetime.now().strftime('%Y-%m-%d')
NEWSAPI_FROM_DATE = (datetime.now() - timedelta(days=DAYS_AGO)).strftime('%Y-%m-%d')

# --- DATA SOURCES ---
# RSS Feeds
RSS_FEEDS = [
    "https://elpais.com/rss/elpais/internacional.xml",
    "https://www.elmundo.es/rss/internacional.xml",
    "https://www.lavanguardia.com/mvc/feed/rss/internacional",
    "https://www.abc.es/rss/feeds/abc_Internacional.xml",
    "https://www.elconfidencial.com/rss/mundo/",
    "https://www.elperiodico.com/es/rss/internacional/",
    "https://www.europapress.es/rss/rss.aspx?ch=69",
    "https://www.rtve.es/api/rss/internacional.xml",
    "https://www.bbc.com/mundo/temas/internacional/index.xml",
    "https://cnnespanol.cnn.com/seccion/mundo/feed/"
]

# Domains of sources considered trustworthy for ranking
TRUSTED_DOMAINS = [
    "elpais.com", "elmundo.es", "lavanguardia.com", "abc.es", "elconfidencial.com",
    "elperiodico.com", "europapress.es", "rtve.es", "bbc.com", "cnnespanol.cnn.com"
]

# --- AI PARAMETERS ---
OPENAI_MODEL = "gpt-4o-mini"  # modern, fast, cost-effective
SYSTEM_PROMPT = (
    f"Genera {NUM_VARIANTS_TO_GENERATE} variantes naturales y concisas del siguiente titular, "
    "manteniendo su sentido original. Devuelve solo la lista de variantes, una por línea, "
    "sin numeración ni viñetas."
)

# --- SCORING ALGORITHM ---
# Weights to adjust the relevance of articles
SCORING_WEIGHTS = {
    "variant_match": 10.0,  # Lower weight; we will add token-overlap
    "keyword_match": 4.0,
    "trusted_source": 3.0,
    "recency_decay_factor": 86400 * 3  # 3-day decay
}

# --- FILTERS ---
# Keyword match threshold for the strict RSS filter (headline must contain at least N)
RSS_KEYWORD_MATCH_THRESHOLD = 2
# Minimum length for keywords extracted from the headline
MIN_KEYWORD_LENGTH = 4

# --- NEWSAPI FLAGS ---
# Prefer matching in title to reduce noise when queries get broader
NEWSAPI_SEARCH_IN_TITLE = True  # if True, adds searchIn=title
# Optionally run a second pass in English (off by default)
ENABLE_ENGLISH_PASS = False