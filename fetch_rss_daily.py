#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_rss_daily.py

Script autonome à lancer une fois par jour (via cron ou launchd).
Récupère les articles RSS du jour et les archive dans rss_archive/YYYY-MM-DD.json.

Usage manuel :
    python fetch_rss_daily.py
    python fetch_rss_daily.py --date 2026-03-05   # forcer une date
    python fetch_rss_daily.py --out /autre/chemin/archive
"""

import argparse
import datetime
import json
import logging
import os
import sys

import feedparser

# ── Config ────────────────────────────────────────────────────────────────────

RSS_SOURCES = [
    "https://www.francetvinfo.fr/titres.rss",
    "https://www.lemonde.fr/rss/une.xml",
    "https://www.lefigaro.fr/rss/figaro_actualites.xml",
    "https://www.liberation.fr/arc/outboundfeeds/rss/?outputType=xml",
    "https://www.lexpress.fr/arc/outboundfeeds/rss/alaune.xml",
]

DEFAULT_ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), "rss_archive")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "rss_fetch.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_for_date(target: datetime.date) -> list[dict]:
    articles = []
    for url in RSS_SOURCES:
        source = url.split("/")[2]
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            n = 0
            for e in feed.entries:
                pub = e.get("published_parsed") or e.get("updated_parsed")
                if not pub:
                    continue
                d = datetime.date(pub.tm_year, pub.tm_mon, pub.tm_mday)
                if d == target:
                    # feedparser expose la description sous "summary" ou "description"
                    desc = (e.get("summary") or e.get("description") or "").strip()
                    articles.append({
                        "source": source,
                        "title": (e.get("title") or "").strip(),
                        "description": desc,
                        "link":  e.get("link") or "",
                        "published": datetime.datetime(*pub[:6]).isoformat(),
                    })
                    n += 1
            log.info(f"  {source}: {n} articles")
        except Exception as exc:
            log.warning(f"  {source}: erreur ({exc})")
    return articles


def save_archive(articles: list[dict], target: datetime.date, archive_dir: str) -> str:
    os.makedirs(archive_dir, exist_ok=True)
    path = os.path.join(archive_dir, f"{target}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"date": str(target), "count": len(articles), "articles": articles},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Archive quotidienne des flux RSS.")
    ap.add_argument("--date", default=None, help="Date cible ISO (défaut: aujourd'hui)")
    ap.add_argument("--out",  default=DEFAULT_ARCHIVE_DIR, help="Dossier d'archive")
    args = ap.parse_args()

    target = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    log.info(f"=== Fetch RSS pour le {target} ===")

    # Vérifie si le fichier existe déjà
    out_path = os.path.join(args.out, f"{target}.json")
    if os.path.exists(out_path):
        log.info(f"Fichier déjà présent : {out_path} — on écrase quand même pour rafraîchir.")

    articles = fetch_for_date(target)
    path = save_archive(articles, target, args.out)
    log.info(f"=== {len(articles)} articles sauvegardés → {path} ===")


if __name__ == "__main__":
    main()
