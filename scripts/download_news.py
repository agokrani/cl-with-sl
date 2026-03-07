#!/usr/bin/env python3
"""Download Nov-Dec 2025 English news articles from infini-news-corpus.

Saves articles to data/news/articles_2025_nov_dec.jsonl for manual review.
After review, pick 5 facts and create data/news/facts.jsonl.

Usage:
    python scripts/download_news.py [--max_articles 200]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from loguru import logger

# Add project root to path so we can import cl
sys.path.insert(0, ".")
sys.path.insert(0, "subliminal-learning")

from cl.data_models import NewsArticle
from cl.news_loader import load_news_articles
from sl.utils.file_utils import save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Download 2025 news articles")
    parser.add_argument("--max_articles", type=int, default=200)
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/news/articles_2025_nov_dec.jsonl",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    articles: list[NewsArticle] = load_news_articles(
        date_start=date(2025, 11, 1),
        date_end=date(2025, 12, 31),
        language="en",
        max_articles=args.max_articles,
    )

    save_jsonl(articles, str(output_path), mode="w")
    logger.success(f"Saved {len(articles)} articles to {output_path}")

    # Print summary for review
    logger.info("Sample titles:")
    for a in articles[:20]:
        logger.info(f"  [{a.date}] {a.title}")


if __name__ == "__main__":
    main()
