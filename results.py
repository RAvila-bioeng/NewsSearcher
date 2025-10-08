# -*- coding: utf-8 -*-
"""
This file contains the ResultPrinter class, responsible for displaying
the aggregated news results to the console.
"""

from typing import List, Dict, Any

class ResultPrinter:
    """Handles printing formatted results to the console."""

    def print_variants(self, variants: List[str]):
        """Prints the AI-generated headline variants."""
        print("\n✅ Generated variants:")
        for variant in variants:
            print(f"- {variant}")

    def print_top_articles(self, articles: List[Dict[str, Any]]):
        """Prints the ranked list of top articles."""
        print(f"\n🏆 --- Top {len(articles)} articles found --- 🏆\n")
        if not articles:
            print("No relevant articles found for your search.")
        else:
            for i, article in enumerate(articles, 1):
                source = (article.get('source') or {}).get('name') or 'Unknown source'
                published = article.get('publishedAt', '')
                print(f"{i}. {article.get('title', 'No title')} ({source})")
                if published:
                    print(f"   🗓 {published}")
                print(f"   🔗 {article.get('url', 'No URL')}\n")