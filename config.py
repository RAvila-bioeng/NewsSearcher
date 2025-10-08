# -*- coding: utf-8 -*-
"""
This file centralizes all the configuration for the news aggregator.
"""

from datetime import datetime, timedelta

# --- SEARCH CONFIGURATION ---
MAX_RESULTS_TO_DISPLAY = 10
NUM_VARIANTS_TO_GENERATE = 3  # Reduced for faster performance in testing

# Dynamic date range for NewsAPI search (last 7 days)
DAYS_AGO = 7
NEWSAPI_TO_DATE = datetime.now().strftime('%Y-%m-%d')
NEWSAPI_FROM_DATE = (datetime.now() - timedelta(days=DAYS_AGO)).strftime('%Y-%m-%d')

# --- DATA SOURCES ---
# RSS Feeds
RSS_FEEDS = [
    "https://elpais.com/rss/elpais/internacional.xml",
    "https://www.elmundo.es/rss/internacional.xml",
    "https://www.lavanguardia.com/mvc/feed/rss/internacional"
]

# Domains of sources considered trustworthy for ranking
TRUSTED_DOMAINS = ["elpais.com", "elmundo.es", "lavanguardia.com"]

# --- AI PARAMETERS ---
OPENAI_MODEL = "gpt-3.5-turbo"
SYSTEM_PROMPT = (
    f"Generate {NUM_VARIANTS_TO_GENERATE} natural and concise variants of the following headline, "
    "maintaining its original meaning. Return only the list of variants, one per line, "
    "without numbering or bullet points."
)

# --- SCORING ALGORITHM ---
# Weights to adjust the relevance of articles
SCORING_WEIGHTS = {
    "variant_match": 15.0,  # Higher weight for exact variant matches
    "keyword_match": 5.0,   # Standard weight for keywords
    "trusted_source": 3.0,  # Small bonus for being from a trusted source
    "recency_decay_factor": 86400 * 2  # Slower decay (2 days)
}

# --- FILTERS ---
# Keyword match threshold for the strict RSS filter (headline must contain at least N)
RSS_KEYWORD_MATCH_THRESHOLD = 2
# Minimum length for keywords extracted from the headline
MIN_KEYWORD_LENGTH = 4