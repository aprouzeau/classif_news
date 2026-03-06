#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_context.py

Fournit un contexte d'actualité journalier à injecter dans les prompts LLM.

Deux sources complémentaires :
  1. RSS feeds de médias français  → actualité du jour (fraîcheur)
  2. Ledger (JSON) d'événements    → contexte persistant (mémoire glissante 30 j)

Usage standalone (test) :
    python news_context.py
    python news_context.py --offline          # skip RSS, ledger seulement
    python news_context.py --ledger my.json   # chemin ledger custom
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import feedparser

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RSS_SOURCES: List[str] = [
    "https://www.francetvinfo.fr/titres.rss",
    "https://www.lemonde.fr/rss/une.xml",
    "https://www.lefigaro.fr/rss/figaro_actualites.xml",
    "https://www.bfmtv.com/rss/news-24-7/",
    "https://www.lexpress.fr/arc/outboundfeeds/rss/alaune.xml",
]

DEFAULT_LEDGER_PATH = "events_ledger.json"
LEDGER_RETENTION_DAYS = 30   # événement non vu depuis > 30 j → archivé
MAX_RSS_ITEMS = 20            # nb max de titres RSS retenus
MAX_LEDGER_ACTIVE = 30        # nb max d'événements injectés dans le prompt
MIN_ENTITY_FREQ = 3           # fréquence min pour qu'une entité entre dans le ledger

# ─────────────────────────────────────────────────────────────────────────────
# RSS fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rss(
    sources: List[str] = DEFAULT_RSS_SOURCES,
    timeout: int = 8,
    max_items: int = MAX_RSS_ITEMS,
) -> List[str]:
    """
    Récupère les titres des flux RSS.
    Retourne une liste de chaînes "Titre [– résumé court si disponible]".
    Les erreurs réseau sont ignorées silencieusement.
    """
    items: List[Tuple[str, str]] = []

    for url in sources:
        try:
            feed = feedparser.parse(
                url,
                request_headers={"User-Agent": "Mozilla/5.0"},
            )
            if feed.bozo and not feed.entries:
                continue
            for entry in feed.entries[:10]:
                title = (entry.get("title") or "").strip()
                summary = re.sub(r"<[^>]+>", "", entry.get("summary") or "").strip()
                summary = summary[:120].rstrip()
                if title:
                    items.append((title, summary))
        except Exception:
            pass

    # Déduplication par titre (plusieurs sources peuvent répéter la même headline)
    seen: set = set()
    deduped: List[str] = []
    for title, summary in items:
        key = title.lower()[:60]
        if key not in seen:
            seen.add(key)
            line = title if not summary else f"{title} – {summary}"
            deduped.append(line)
        if len(deduped) >= max_items:
            break

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Ledger  (mémoire glissante d'événements)
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> str:
    return date.today().isoformat()


def load_ledger(path: str = DEFAULT_LEDGER_PATH) -> List[Dict]:
    """
    Charge le ledger depuis un fichier JSON.

    Chaque entrée est un dict :
      - label      : str   description courte de l'événement (max ~15 mots)
      - entities   : List[str]  entités nommées associées (noms, lieux, orgs)
      - first_seen : str   date ISO de première apparition
      - last_seen  : str   date ISO de dernière apparition
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data.get("events", [])
        except Exception:
            return []


def save_ledger(events: List[Dict], path: str = DEFAULT_LEDGER_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"events": events, "updated": _today()},
            f,
            ensure_ascii=False,
            indent=2,
        )


def get_active_events(
    events: List[Dict],
    retention_days: int = LEDGER_RETENTION_DAYS,
) -> List[Dict]:
    """Filtre les événements vus dans les `retention_days` derniers jours."""
    cutoff = date.today() - timedelta(days=retention_days)
    active = []
    for ev in events:
        try:
            last = date.fromisoformat(ev.get("last_seen", "2000-01-01"))
            if last >= cutoff:
                active.append(ev)
        except Exception:
            active.append(ev)
    return active


def match_ledger_events(text: str, events: List[Dict]) -> List[Dict]:
    """
    Retourne les événements du ledger dont au moins une entité apparaît dans `text`.

    Deux niveaux de matching (par ordre de priorité) :
      1. Correspondance directe : l'entité est une sous-chaîne du texte (insensible à la casse)
      2. Correspondance partielle : un token de l'entité (>= 4 caractères) est dans le texte
         → évite les faux positifs sur des mots courts ("France", "les", etc.)

    Les événements sont retournés sans doublon, triés par nombre de matches décroissant
    (les plus pertinents en premier).
    """
    text_lower = text.lower()
    scored: List[Tuple[int, Dict]] = []

    for ev in events:
        score = 0
        for ent in ev.get("entities") or []:
            ent_lower = ent.lower()
            if ent_lower in text_lower:
                score += 2  # match exact
            else:
                # match partiel : au moins un token significatif de l'entité
                tokens = [t for t in re.split(r"[\s\-/,]+", ent_lower) if len(t) >= 4]
                if any(tok in text_lower for tok in tokens):
                    score += 1
        if score > 0:
            scored.append((score, ev))

    # Tri décroissant par score, dédoublonnage
    seen_ids = set()
    result = []
    for score, ev in sorted(scored, key=lambda x: -x[0]):
        ev_id = ev.get("label", "")[:40]
        if ev_id not in seen_ids:
            seen_ids.add(ev_id)
            result.append(ev)

    return result


def update_ledger_from_annotations(
    annotations: List[Dict],
    ledger_path: str = DEFAULT_LEDGER_PATH,
    min_freq: int = MIN_ENTITY_FREQ,
) -> None:
    """
    Met à jour le ledger en extrayant les entités fréquentes des annotations LLM.

    `annotations` : liste de dicts avec les clés :
        - "entities" : List[str]
        - "short"    : str  (reformulation courte du bandeau)
        - "domain"   : str
        - "freq"     : int  (optionnel, fréquence de diffusion du bandeau)

    Logique :
      - Toute entité apparaissant dans >= min_freq bandeaux (pondéré par freq) est
        considérée significative.
      - Si elle correspond à un événement existant → last_seen mis à jour.
      - Sinon → nouvel événement ajouté.
      - Les événements trop anciens (> retention_days) sont purgés.
    """
    today = _today()
    events = load_ledger(ledger_path)

    # ── Agrège entités + leur contexte ──────────────────────────────────────
    entity_count: Counter = Counter()
    entity_context: Dict[str, List[str]] = defaultdict(list)

    for ann in annotations:
        ents = ann.get("entities") or []
        if not isinstance(ents, list):
            continue
        short = (ann.get("short") or "").strip()
        weight = max(1, int(ann.get("freq", 1)))
        for ent in ents:
            ent = str(ent).strip()
            if not ent:
                continue
            entity_count[ent] += weight
            if short:
                entity_context[ent].append(short)

    # ── Ne retient que les entités significatives ────────────────────────────
    significant = {e: n for e, n in entity_count.items() if n >= min_freq}

    for ent, freq in significant.items():
        # Cherche une entrée existante par correspondance d'entité (insensible à la casse)
        matched = None
        for ev in events:
            ev_ents_lower = [e.lower() for e in (ev.get("entities") or [])]
            if ent.lower() in ev_ents_lower:
                matched = ev
                break

        contexts = entity_context.get(ent, [])
        label_candidate = contexts[0] if contexts else ent

        if matched:
            matched["last_seen"] = today
            matched["freq_today"] = freq
        else:
            events.append(
                {
                    "label": label_candidate,
                    "entities": [ent],
                    "first_seen": today,
                    "last_seen": today,
                    "freq_today": freq,
                }
            )

    # ── Purge les événements expirés ─────────────────────────────────────────
    events = get_active_events(events, LEDGER_RETENTION_DAYS)
    save_ledger(events, ledger_path)
    print(f"[ledger] {len(events)} événements actifs → {ledger_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Context string builder
# ─────────────────────────────────────────────────────────────────────────────

def build_context_string(
    rss_items: List[str],
    ledger_events: List[Dict],
    max_rss: int = MAX_RSS_ITEMS,
    max_ledger: int = MAX_LEDGER_ACTIVE,
) -> str:
    """
    Construit le bloc de contexte global (RSS + ledger complet).
    Utilisé pour l'affichage au démarrage ; pas injecté tel quel dans les prompts.
    """
    lines: List[str] = []

    if rss_items:
        lines.append("=== ACTUALITÉ DU JOUR (flux RSS) ===")
        for item in rss_items[:max_rss]:
            lines.append(f"• {item}")
        lines.append("")

    active = get_active_events(ledger_events)
    if active:
        lines.append("=== ÉVÉNEMENTS EN COURS (mémoire glissante) ===")
        active_sorted = sorted(
            active, key=lambda e: e.get("last_seen", ""), reverse=True
        )
        for ev in active_sorted[:max_ledger]:
            ents = ", ".join(ev.get("entities") or [])
            label = ev.get("label", "")
            try:
                last = date.fromisoformat(ev["last_seen"])
                delta = (date.today() - last).days
                recency = f" [vu il y a {delta}j]" if delta > 0 else " [vu aujourd'hui]"
            except Exception:
                recency = ""
            lines.append(f"• {label}  (entités : {ents}){recency}")
        lines.append("")

    return "\n".join(lines) if lines else ""


def build_targeted_context(matched_events: List[Dict], rss_items: List[str] = None) -> str:
    """
    Construit un contexte ciblé pour un bandeau spécifique, à partir des événements
    du ledger qui correspondent à ses entités.

    Si des titres RSS sont fournis (optionnel), ils sont ajoutés pour enrichir
    le contexte sur des événements potentiellement non encore dans le ledger.
    """
    lines: List[str] = []

    if matched_events:
        lines.append("Contexte d'actualité lié à ce bandeau :")
        for ev in matched_events:
            ents = ", ".join(ev.get("entities") or [])
            lines.append(f"• {ev['label']}  [entités clés : {ents}]")
        lines.append("")

    if rss_items:
        lines.append("Autres titres d'actualité du jour :")
        for item in rss_items[:5]:
            lines.append(f"• {item}")
        lines.append("")

    return "\n".join(lines) if lines else ""


def get_daily_context(
    ledger_path: str = DEFAULT_LEDGER_PATH,
    rss_sources: Optional[List[str]] = None,
    offline: bool = False,
) -> str:
    """
    Point d'entrée principal.
    Retourne le contexte journalier complet (RSS + ledger) en une chaîne.
    Si offline=True, ignore le RSS et utilise uniquement le ledger.
    """
    rss_items: List[str] = []
    if not offline:
        print("[context] Récupération des flux RSS...")
        rss_items = fetch_rss(rss_sources or DEFAULT_RSS_SOURCES)
        print(f"[context] {len(rss_items)} titres RSS récupérés.")

    ledger_events = load_ledger(ledger_path)
    active = get_active_events(ledger_events)
    print(f"[context] {len(active)} événements actifs dans le ledger.")

    return build_context_string(rss_items, ledger_events)


# ─────────────────────────────────────────────────────────────────────────────
# CLI (test standalone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Affiche le contexte d'actualité journalier.")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, help="Chemin du fichier ledger JSON")
    ap.add_argument("--offline", action="store_true", help="Skip RSS, ledger seulement")
    args = ap.parse_args()

    ctx = get_daily_context(ledger_path=args.ledger, offline=args.offline)
    if ctx:
        print("\n" + "─" * 60)
        print(ctx)
        print("─" * 60)
    else:
        print("[context] Aucun contexte disponible (ledger vide et RSS inaccessibles).")
