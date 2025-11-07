#!/usr/bin/env python3

import json
import logging
import os
import sys
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("news_fetcher")

# ------------------------------------------------------------------
# Configuration - OPTIMIZED SETTINGS
# ------------------------------------------------------------------
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    log.error("NEWS_API_KEY environment variable not set")
    sys.exit(1)

# Optimized queries - fewer but more targeted
CATEGORY_QUERIES = {
    "AI & Technology": [
        "artificial intelligence OR machine learning",
        "programming OR software development",
        "technology OR tech news"
    ],
    "Business & Economy": [
        "business OR economy OR stock market",
        "business news OR economy news"
    ],
    "Politics": [
        "politics OR government OR election",
        "political news OR congress"
    ],
    "Sports": [
        "sports OR football OR basketball",
        "NBA OR NFL OR soccer"
    ],
    "Health & Science": [
        "health OR medicine OR healthcare",
        "science OR research"
    ],
    "Entertainment": [
        "entertainment OR movies OR Hollywood",
        "celebrity OR entertainment news"
    ],
    "Gaming": [
        "gaming OR video games",
        "PlayStation OR Xbox"
    ],
    "Environment": [
        "climate change OR environment",
        "global warming OR sustainability"
    ]
}

URL = "https://newsapi.org/v2/everything"
PAGE_SIZE = 30
MAX_PAGES_PER_QUERY = 1
REQUEST_DELAY = 2

# ------------------------------------------------------------------
# HTTP session with better rate limit handling
# ------------------------------------------------------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=(429, 500, 502, 503, 504),
)
session.mount("https://", HTTPAdapter(max_retries=retries))


def safe_lower(text):
    """Safely convert text to lowercase handling None values"""
    return text.lower() if text else ""


def detect_article_category(article: Dict[str, Any]) -> str:
    """
    Detect the actual category of an article based on its content.
    Returns the most specific category that matches.
    """
    title = safe_lower(article.get("title", ""))
    description = safe_lower(article.get("description", ""))
    content = safe_lower(article.get("content", ""))

    # Combine all text for analysis
    full_text = f"{title} {description} {content}"

    # Define keyword patterns for each category (simplified)
    category_keywords = {
        "Sports": [
            r'\b(sports|football|basketball|baseball|soccer|tennis|golf|nba|nfl|mlb|nhl|olympics|championship|game|match|player|team|coach|score|tournament)\b'
        ],
        "Politics": [
            r'\b(politics|government|election|congress|senate|white house|president|trump|biden|democrat|republican|vote|campaign|policy|law|legislation)\b'
        ],
        "Business & Economy": [
            r'\b(business|economy|stock|market|invest|finance|economic|company|corporation|earnings|profit|revenue|wall street|dow jones|nasdaq)\b'
        ],
        "Health & Science": [
            r'\b(health|medicine|medical|healthcare|hospital|doctor|patient|disease|treatment|vaccine|covid|virus|pandemic|fitness|nutrition|wellness|science|research|scientific|study|discovery)\b'
        ],
        "AI & Technology": [
            r'\b(artificial intelligence|ai\b|machine learning|deep learning|neural network|algorithm|chatgpt|gpt|openai|technology|tech|computer|software|hardware|internet|digital|app|application|startup|innovation)\b'
        ],
        "Entertainment": [
            r'\b(entertainment|movie|film|hollywood|celebrity|actor|actress|director|oscar|award|premiere|box office|streaming|netflix|disney)\b'
        ],
        "Gaming": [
            r'\b(gaming|video game|playstation|xbox|nintendo|steam|esports|gamer|game release|gameplay|console|pc gaming|mobile game)\b'
        ],
        "Environment": [
            r'\b(environment|climate|global warming|sustainability|renewable|solar|wind|energy|pollution|conservation|eco-friendly|green)\b'
        ]
    }

    # Score each category based on keyword matches
    category_scores = {}
    for category, patterns in category_keywords.items():
        score = 0
        for pattern in patterns:
            try:
                matches = re.findall(pattern, full_text)
                score += len(matches)
            except Exception:
                continue
        category_scores[category] = score

    # Get the category with the highest score
    if category_scores:
        best_category = max(category_scores.items(), key=lambda x: x[1])
        if best_category[1] > 0:  # Only return if we found matches
            return best_category[0]

    # Fallback: return "General News" if no specific category detected
    return "General News"


def fetch_articles_for_category(category: str, queries: List[str]) -> List[Dict[str, Any]]:
    """Fetch articles for a specific category using multiple queries."""
    articles = []

    for query in queries:
        # Add delay between requests to avoid rate limits
        time.sleep(REQUEST_DELAY)

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": PAGE_SIZE,
            "page": 1,
            "apiKey": API_KEY,
        }

        try:
            log.info(f"Fetching {category}: '{query}'")
            response = session.get(URL, params=params, timeout=30)

            if response.status_code == 429:
                log.warning(f"⚠️ Rate limit hit for '{query}'. Skipping...")
                continue

            response.raise_for_status()
            data = response.json()

            if "articles" not in data:
                log.warning(f"No 'articles' in response for '{query}'")
                continue

            batch = data["articles"]

            # Process articles and assign detected categories
            valid_articles = []
            for article in batch:
                if article.get("title") and article.get("title") != "[Removed]":
                    # Detect the actual category based on content
                    try:
                        detected_category = detect_article_category(article)
                        article["query_category"] = detected_category
                        article["search_query"] = query
                        valid_articles.append(article)
                    except Exception as e:
                        log.warning(f"Error detecting category for article: {e}")
                        # Assign fallback category
                        article["query_category"] = "General News"
                        article["search_query"] = query
                        valid_articles.append(article)

            articles.extend(valid_articles)
            log.info(f"✅ {category} - '{query}': {len(valid_articles)} articles")

        except Exception as e:
            log.error(f"❌ Error fetching {category} - '{query}': {e}")

    return articles


def normalize(article: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize article data with safe field handling."""
    return {
        "title": article.get("title", ""),
        "url": article.get("url", ""),
        "publishedAt": article.get("publishedAt", ""),
        "source": article.get("source", {}).get("name", ""),
        "description": article.get("description", ""),
        "content": article.get("content", ""),
        "urlToImage": article.get("urlToImage", ""),
        "query_category": article.get("query_category", "General News"),
        "search_query": article.get("search_query", ""),
    }


def main() -> None:
    log.info("🚀 Starting optimized news fetch...")

    all_articles = []
    successful_categories = 0

    # Fetch articles for each category
    for category, queries in CATEGORY_QUERIES.items():
        try:
            articles = fetch_articles_for_category(category, queries)
            if articles:
                all_articles.extend(articles)
                successful_categories += 1

                # Log category distribution
                category_counts = {}
                for article in articles:
                    detected_cat = article.get("query_category", "Unknown")
                    category_counts[detected_cat] = category_counts.get(detected_cat, 0) + 1

                log.info(f"📊 {category} search resulted in: {category_counts}")
        except Exception as e:
            log.error(f"❌ Error processing category {category}: {e}")

    log.info(
        f"📊 Collected {len(all_articles)} articles from {successful_categories}/{len(CATEGORY_QUERIES)} categories"
    )

    # Remove duplicates
    seen_urls = set()
    unique_articles = []
    for article in all_articles:
        url = article.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)

    log.info(f"✨ Unique articles: {len(unique_articles)}")

    # Process articles
    processed_articles = [normalize(article) for article in unique_articles]

    # Analyze final category distribution
    final_categories = {}
    for article in processed_articles:
        category = article.get("query_category", "Unknown")
        final_categories[category] = final_categories.get(category, 0) + 1

    log.info("🎯 Final Category Distribution:")
    for category, count in sorted(final_categories.items(), key=lambda x: x[1], reverse=True):
        log.info(f"   {category}: {count} articles")

    # Write to file
    # out_dir = Path(os.getenv("KESTRA_WORKING_DIR", "."))
    # out_file = out_dir / "news_articles.jsonl"

    out_file = Path("news_articles.jsonl")

    with out_file.open("w", encoding="utf-8") as fh:
        for art in processed_articles:
            fh.write(json.dumps(art, ensure_ascii=False) + "\n")

    file_size = out_file.stat().st_size
    log.info(f"💾 Saved to: {out_file}")
    log.info(f"📁 File size: {file_size} bytes")

    # Show sample of collected data with categories
    if processed_articles:
        log.info("📰 Sample articles with categories:")
        for i, art in enumerate(processed_articles[:5]):
            log.info(f"   {i+1}. [{art.get('query_category', 'Unknown')}] {art.get('title', 'No title')}")


if __name__ == "__main__":
    main()
