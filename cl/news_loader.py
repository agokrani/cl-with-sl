from datetime import date

import pandas as pd
from loguru import logger

from cl.data_models import NewsArticle

# Dataset is partitioned by year with ~1251 shards for 2025.
# Dates within shards are mixed, so we load a few late shards and filter.
_HF_BASE_URL = (
    "https://huggingface.co/datasets/ruggsea/infini-news-corpus"
    "/resolve/main/data/year%3D{year}/part-{shard:05d}.parquet"
)


def load_news_articles(
    date_start: date,
    date_end: date,
    language: str = "en",
    max_articles: int = 200,
    max_chars: int = 3000,
    year: int = 2025,
    shard_range: tuple[int, int] = (1200, 1251),
) -> list[NewsArticle]:
    """Load English news articles from ruggsea/infini-news-corpus within a date range.

    Reads parquet shards directly from HuggingFace to avoid scanning 850M+ rows.
    The dataset is partitioned by year; within each year partition we read specific
    shard files and filter by date, language, and content quality.
    """
    logger.info(
        f"Loading up to {max_articles} articles from {date_start} to {date_end} "
        f"(lang={language}, shards {shard_range[0]}-{shard_range[1]})"
    )

    date_prefixes = []
    d = date_start
    while d <= date_end:
        prefix = d.strftime("%Y-%m")
        if prefix not in date_prefixes:
            date_prefixes.append(prefix)
        # Jump to next month
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)

    all_articles: list[NewsArticle] = []

    for shard in range(shard_range[0], shard_range[1]):
        url = _HF_BASE_URL.format(year=year, shard=shard)
        try:
            df = pd.read_parquet(url, engine="pyarrow")
        except Exception as e:
            logger.warning(f"Shard {shard}: {e}")
            continue

        # Filter: language, title, text length, date prefix
        mask = (
            (df["language"] == language)
            & (df["title"].str.len() > 5)
            & (df["text"].str.len() > 200)
        )
        date_mask = pd.Series(False, index=df.index)
        for prefix in date_prefixes:
            date_mask = date_mask | df["date"].str.startswith(prefix)
        mask = mask & date_mask

        matched = df[mask]
        if len(matched) == 0:
            continue

        for _, row in matched.iterrows():
            all_articles.append(
                NewsArticle(
                    title=str(row["title"]).strip(),
                    text=str(row["text"])[:max_chars].strip(),
                    date=date.fromisoformat(str(row["date"])[:10]),
                    source=row.get("sitename"),
                    language=str(row["language"]),
                )
            )
            if len(all_articles) >= max_articles:
                break

        logger.info(
            f"Shard {shard}: found {len(matched)} matches (total collected: {len(all_articles)})"
        )
        if len(all_articles) >= max_articles:
            break

    logger.info(f"Collected {len(all_articles)} articles")
    return all_articles
