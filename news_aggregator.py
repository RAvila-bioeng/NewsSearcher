# -*- coding: utf-8 -*-
"""
This script is an advanced news aggregator that collects articles from NewsAPI and RSS feeds,
uses AI to generate search variants, and ranks them by relevance.
"""

import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import requests
import feedparser
from openai import OpenAI
from dateutil import parser

# Import centralized configuration
import config

# --- INITIAL SETUP ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


# --- DATA FETCHING CLASSES ---

class NewsAPIFetcher:
    """Fetches articles from NewsAPI."""
    BASE_URL = "https://newsapi.org/v2/everything"

    def fetch(self, query_variants: List[str]) -> List[Dict[str, Any]]:
        query_string = " OR ".join(f'"{variant}"' for variant in query_variants)
        logging.info(f"Searching NewsAPI with query: '{query_string}'")

        params = {
            "q": query_string,
            "from": config.NEWSAPI_FROM_DATE,
            "to": config.NEWSAPI_TO_DATE,
            "sortBy": "publishedAt",
            "language": "en", # Changed to English
            "pageSize": 100,
            "apiKey": NEWSAPI_KEY
        }

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            logging.info(f"Found {len(articles)} articles from NewsAPI.")
            return articles
        except requests.exceptions.RequestException as e:
            logging.error(f"Error accessing NewsAPI: {e}")
            return []

class RSSFetcher:
    """Fetches articles from RSS feeds."""
    def fetch(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logging.info("Fetching articles from RSS feeds...")
        rss_articles = []
        for feed_url in config.RSS_FEEDS:
            logging.info(f"Processing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()

                # Strict filter: must match a minimum number of keywords
                if sum(kw in text for kw in keywords) >= config.RSS_KEYWORD_MATCH_THRESHOLD:
                    rss_articles.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", ""),
                        "url": entry.get("link", ""),
                        "publishedAt": entry.get("published", "")
                    })
        logging.info(f"Found {len(rss_articles)} relevant articles from RSS feeds.")
        return rss_articles


# --- PROCESSING CLASSES ---

class ArticleScorer:
    """Calculates the relevance score for articles."""
    def score(self, article: Dict, query_variants: List[str], keywords: List[str]) -> float:
        score = 0.0
        text_to_analyze = (article.get("title", "") + " " + article.get("description", "")).lower()

        # Criterion 1: Match with AI variants
        if any(variant.lower() in text_to_analyze for variant in query_variants):
            score += config.SCORING_WEIGHTS["variant_match"]

        # Criterion 2: Match with keywords
        score += sum(config.SCORING_WEIGHTS["keyword_match"] for kw in keywords if kw in text_to_analyze)

        # Criterion 3: Trusted source
        if any(domain in article.get("url", "") for domain in config.TRUSTED_DOMAINS):
            score += config.SCORING_WEIGHTS["trusted_source"]

        # Criterion 4: Publication date (more recent = more points)
        date_str = article.get("publishedAt", "")
        if date_str:
            try:
                date_obj = parser.parse(date_str)
                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=datetime.now().astimezone().tzinfo)

                seconds_since = (datetime.now(date_obj.tzinfo) - date_obj).total_seconds()
                score += 1 / (1 + seconds_since / config.SCORING_WEIGHTS["recency_decay_factor"])
            except (ValueError, TypeError):
                logging.warning(f"Could not parse date: {date_str}")

        return score

class HeadlineVariationsGenerator:
    """Generates headline variations using AI."""
    def __init__(self):
        try:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            self.client = None
            logging.error(f"Error initializing OpenAI client: {e}")

    def generate(self, headline: str) -> List[str]:
        if not self.client:
            logging.warning("OpenAI client not initialized. Skipping paraphrasing.")
            return [headline]

        logging.info(f"Generating {config.NUM_VARIANTS_TO_GENERATE} headline variants with AI...")
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": headline}
                ]
            )
            text = response.choices[0].message.content.strip()
            variants = [v.strip("•- ") for v in text.split("\n") if v.strip()]
            logging.info("Generated variants:")
            for v in variants:
                logging.info(f"- {v}")
            return variants
        except Exception as e:
            logging.error(f"Error contacting OpenAI API: {e}")
            return [headline]


# --- MAIN ORCHESTRATION CLASS ---

class NewsAggregator:
    """Orchestrates the news aggregation and ranking process."""
    def __init__(self):
        self.newsapi_fetcher = NewsAPIFetcher()
        self.rss_fetcher = RSSFetcher()
        self.scorer = ArticleScorer()
        self.variant_generator = HeadlineVariationsGenerator()

    def _combine_and_deduplicate(self, list_of_article_lists: List[List[Dict]]) -> List[Dict]:
        all_articles = [article for sublist in list_of_article_lists for article in sublist]
        seen_urls = set()
        unique_articles = []
        for art in all_articles:
            url = art.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(art)

        logging.info(f"Total combined articles: {len(all_articles)}")
        logging.info(f"Total unique articles: {len(unique_articles)}")
        return unique_articles

    def run(self):
        # 1. Get headline from user
        base_headline = input("Please enter the headline you want to investigate: ")
        if not base_headline:
            logging.warning("No headline entered. Exiting.")
            return

        # 2. Generate variants and keywords
        query_variants = self.variant_generator.generate(base_headline)
        keywords = [w for w in base_headline.lower().split() if len(w) > config.MIN_KEYWORD_LENGTH]

        # 3. Fetch articles
        newsapi_articles = self.newsapi_fetcher.fetch(query_variants)
        rss_articles = self.rss_fetcher.fetch(keywords)

        # 4. Combine and deduplicate
        all_articles = self._combine_and_deduplicate([newsapi_articles, rss_articles])

        # 5. Score and sort
        logging.info("Scoring and sorting articles by relevance...")
        all_articles.sort(
            key=lambda a: self.scorer.score(a, query_variants, keywords),
            reverse=True
        )

        # 6. Display results
        top_articles = all_articles[:config.MAX_RESULTS_TO_DISPLAY]
        logging.info(f"--- Top {len(top_articles)} articles found ---")
        if not top_articles:
            logging.info("No relevant articles found for your search.")
        else:
            for i, article in enumerate(top_articles, 1):
                logging.info(f"{i}. {article.get('title', 'No title')}")
                logging.info(f"   🔗 {article.get('url', 'No URL')}")


def main():
    """Main entry point of the script."""
    # Verify that API keys are present
    if not OPENAI_API_KEY or "CLAVE_EJEMPLO" in OPENAI_API_KEY:
        logging.error("OpenAI API key is not configured. Please check your .env file.")
        return
    if not NEWSAPI_KEY or "CLAVE_EJEMPLO" in NEWSAPI_KEY:
        logging.error("NewsAPI key is not configured. Please check your .env file.")
        return

    aggregator = NewsAggregator()
    aggregator.run()

if __name__ == "__main__":
    main()