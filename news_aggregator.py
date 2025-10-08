# -*- coding: utf-8 -*-
"""
This script is an advanced news aggregator that collects articles from NewsAPI and RSS feeds,
uses AI to generate search variants, and ranks them by relevance.
"""

import os
import logging
from typing import List, Dict, Any, Set
from datetime import datetime
from dotenv import load_dotenv
import requests
import feedparser
from openai import OpenAI
from dateutil import parser
import re
import unicodedata

# --- TEXT NORMALIZATION & KEYWORDS ---
_STOPWORDS_ES = {
    "el","la","los","las","un","una","unos","unas","de","del","al","a","y","o","u","que","en","con","por","para","como","su","sus","se","es","son","fue","fueron","ser","esta","está","están","estaba","han","ha","hay","más","menos","muy","ya","no","sí","lo","le","les","pero","entre","sobre","tras","desde","sin","hasta","también","cuando","donde","cuanto","qué","quién","quienes","cual","cuales","este","esta","estos","estas","ese","esa","esos","esas","aquel","aquella","aquellos","aquellas","mi","mis","tu","tus","su","sus"
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", text)  # keep alnum and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> Set[str]:
    tokens = set()
    for tok in text.split():
        if tok in _STOPWORDS_ES:
            continue
        if len(tok) <= config.MIN_KEYWORD_LENGTH:
            continue
        tokens.add(tok)
    return tokens

def extract_keywords(headline: str) -> List[str]:
    norm = normalize_text(headline)
    tokens = list(tokenize(norm))
    # Preserve a reasonable cap to avoid overly long AND queries
    tokens.sort()  # deterministic order
    return tokens[:12]

# Import centralized configuration
import config
from results import ResultPrinter

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

    def fetch(self, query_variants: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        # Query 1: OR over variants (as before)
        variants_query = " OR ".join(f'({v})' for v in query_variants)
        # Query 2: compact AND over top keywords to capture core concept
        top_keywords = keywords[:6] if keywords else []
        and_keywords_query = " AND ".join(f'({kw})' for kw in top_keywords) if top_keywords else ""

        all_articles: List[Dict[str, Any]] = []

        def _run(query: str, language: str = "es") -> List[Dict[str, Any]]:
            if not query:
                return []
            params = {
                "q": query,
                "from": config.NEWSAPI_FROM_DATE,
                "to": config.NEWSAPI_TO_DATE,
                "sortBy": "publishedAt",
                "language": language,
                "pageSize": 100,
                "apiKey": NEWSAPI_KEY
            }
            if getattr(config, "NEWSAPI_SEARCH_IN_TITLE", False):
                params["searchIn"] = "title"
            logging.info(f"Searching NewsAPI with query: '{query}' lang={language} searchIn={params.get('searchIn','all')}")
            try:
                response = requests.get(self.BASE_URL, params=params)
                response.raise_for_status()
                return response.json().get("articles", [])
            except requests.exceptions.RequestException as e:
                logging.error(f"Error accessing NewsAPI: {e}")
                return []

        # Run Spanish queries
        all_articles.extend(_run(variants_query, language="es"))
        all_articles.extend(_run(and_keywords_query, language="es"))

        # Optional English pass
        if getattr(config, "ENABLE_ENGLISH_PASS", False):
            all_articles.extend(_run(variants_query, language="en"))
            all_articles.extend(_run(and_keywords_query, language="en"))

        logging.info(f"Found {len(all_articles)} raw articles from NewsAPI across queries.")
        return all_articles

class RSSFetcher:
    """Fetches articles from RSS feeds."""
    def fetch(self, keywords: List[str]) -> List[Dict[str, Any]]:
        logging.info("Fetching articles from RSS feeds...")
        rss_articles = []
        for feed_url in config.RSS_FEEDS:
            logging.info(f"Processing feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                title_text = normalize_text(entry.get("title", ""))
                summary_text = normalize_text(entry.get("summary", ""))
                # Title matches weigh double
                match_count = 2 * sum(kw in title_text for kw in keywords) + sum(kw in summary_text for kw in keywords)
                if match_count >= config.RSS_KEYWORD_MATCH_THRESHOLD:
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
        title_text_raw = article.get("title", "")
        desc_text_raw = article.get("description", "")
        text_to_analyze = (title_text_raw + " " + desc_text_raw).lower()
        title_norm = normalize_text(title_text_raw)
        desc_norm = normalize_text(desc_text_raw)
        title_tokens = tokenize(title_norm)
        keyword_set: Set[str] = set(keywords)

        # Criterion 1: Match with AI variants
        if any(variant.lower() in text_to_analyze for variant in query_variants):
            score += config.SCORING_WEIGHTS["variant_match"]

        # Criterion 2: Match with keywords
        score += sum(config.SCORING_WEIGHTS["keyword_match"] for kw in keywords if kw in text_to_analyze)

        # Criterion 2b: Token overlap (title emphasis)
        if keyword_set:
            overlap = len(title_tokens & keyword_set) / max(1, len(keyword_set))
            # Title carries more weight than description; add a smooth bonus
            score += 6.0 * overlap
            # Small bonus for description overlap
            desc_tokens = tokenize(desc_norm)
            desc_overlap = len(desc_tokens & keyword_set) / max(1, len(keyword_set))
            score += 2.0 * desc_overlap

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
        self.printer = ResultPrinter()

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
        self.printer.print_variants(query_variants)  # Delegate printing
        keywords = extract_keywords(base_headline)

        # 3. Fetch articles
        newsapi_articles = self.newsapi_fetcher.fetch(query_variants, keywords)
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
        self.printer.print_top_articles(top_articles)


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