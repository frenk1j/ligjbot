"""
Lajme RSS — filtrim sipas kategorive ligjore/rrugore.
"""

import time
import feedparser

NEWS_FEEDS = [
    {"source": "BalkanWeb",         "url": "https://www.balkanweb.com/feed/",           "category": "rrugore"},
    {"source": "Albaniandailynews", "url": "https://albaniandailynews.com/feed/",     "category": "ligjore"},
    {"source": "Shqiptarja",        "url": "https://shqiptarja.com/rss",               "category": "rrugore"},
    {"source": "Top Channel",       "url": "https://top-channel.tv/feed/",             "category": "ligjore"},
    {"source": "Panorama",          "url": "https://www.panorama.com.al/feed/",        "category": "ligjore"},
    {"source": "ABC News",          "url": "https://abcnews.al/feed/",                 "category": "ligjore"},
]

CATEGORY_KEYWORDS = {
    "rrugore": [
        "kodi rrugor", "road code", "dpshtrr", "patentë", "patent",
        "kontroll teknik", "transport", "ligj rrugor", "shofer",
        "targa", "automjet", "drejtues mjeti", "rrugë", "rruge",
    ],
    "gjoba": [
        "gjobë", "gjoba", "gjobat", "denim", "kamion", "radar",
        "shpejtësi", "shpejtesi", "parkim",
    ],
    "policia": [
        "polici", "policia", "policise", "ndalim", "kontroll",
        "arrest", "procedur", "forca e rendit",
    ],
    "ligjore": [
        "ligj", "ligji", "nen ", "neni", "kuvend", "ministri",
        "qeveri", "decret", "rregullore", "akt normativ",
    ],
    "siguri": [
        "aksident", "sigurim", "rrugor", "vdekje", "plagosur",
        "trafik", "mbikëqyrje",
    ],
}

_news_cache: dict = {"data": [], "ts": 0}
NEWS_TTL = 900  # 15 min


def _detect_category(text: str, default: str) -> str:
    text = text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


def _card_variant(category: str) -> str:
    return {
        "rrugore": "card-road",
        "gjoba": "card-fine",
        "policia": "card-police",
        "ligjore": "card-law",
        "siguri": "card-alert",
    }.get(category, "card-standard")


def fetch_news() -> list[dict]:
    now = time.time()
    if _news_cache["data"] and (now - _news_cache["ts"]) < NEWS_TTL:
        return _news_cache["data"]

    articles: list[dict] = []
    seen_links: set[str] = set()

    for feed_info in NEWS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:25]:
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link or link in seen_links:
                    continue

                text = f"{title} {summary}".lower()
                if not any(
                    kw in text
                    for kws in CATEGORY_KEYWORDS.values()
                    for kw in kws
                ):
                    continue

                seen_links.add(link)
                category = _detect_category(text, feed_info["category"])
                articles.append({
                    "title": title,
                    "summary": summary[:220],
                    "link": link,
                    "published": entry.get("published", ""),
                    "source": feed_info["source"],
                    "category": category,
                    "variant": _card_variant(category),
                })
        except Exception as e:
            print(f"[WARN] Feed {feed_info['source']} deshtoi: {e}")

    articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    _news_cache["data"] = articles
    _news_cache["ts"] = now
    return articles
