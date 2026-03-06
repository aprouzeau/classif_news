#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_bandeaux_llm.py

Pipeline complet :
1) lecture CSV (bandeau + fréquence)
2) cleaning robuste (bandeaux TV en CAPS, artefacts, "(2)", barres, etc.)
3) filtrage (ex: freq>1)
4) enrichissement léger par petit LLM local (llama-cpp) -> domain / entities / keywords
5) embeddings (Sentence-Transformers / E5)
6) clustering hiérarchique (Agglomerative) avec recherche automatique d’un seuil
   pour viser ~10–15 clusters (imbalance OK)
7) post-traitement : petits clusters -> outliers (-1)
8) export CSV + rapport TXT

Usage (exemples) :
python cluster_bandeaux_llm.py \
  --input echantillon.csv \
  --text_col bandeau --count_col count \
  --min_count 2 \
  --embed_model intfloat/multilingual-e5-small \
  --llm_model_path models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --target_min 10 --target_max 15 \
  --min_cluster_size 6 \
  --out_prefix bandeaux_clusters_llm_v2

Notes :
- Si tu ne veux pas d’enrichissement LLM : ajoute --no_llm
- Sur Mac M3 Pro, llama-cpp avec Metal marche bien : pip install "llama-cpp-python==0.3.7"
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Taxonomie hiérarchique multi-dimension
# -----------------------------
TOPIC_L1_LIST = [
    "politique",       # politique intérieure française
    "économie",        # budget, entreprises, marchés, finances
    "société",         # faits de société, vie quotidienne, débats
    "médias",          # presse, ARCOM, liberté de la presse
    "sécurité",        # police, narcotrafic, attentats, violence
    "géopolitique",    # international, diplomatie, conflits
    "culture",         # culture, TV, cinéma, arts
    "sport",           # sport
    "autre",           # fallback / bruit
]

TOPIC_L2_LIST = [
    # politique
    "gouvernement", "parlement", "partis_politiques", "élections", "budget_finances",
    # économie
    "entreprises", "marchés",
    # société
    "justice", "éducation", "logement", "immigration", "santé", "faits_divers",
    # médias
    "liberté_presse", "médias_audiovisuels", "réseaux_sociaux",
    # sécurité
    "narcotrafic", "terrorisme", "violence_urbaine", "forces_ordre",
    # géopolitique
    "diplomatie", "conflits", "géopolitique_europe",
    # culture / sport
    "culture_entertainment", "compétition_sportive",
    # fallback
    "autre",
]

EVENT_TYPE_LIST = [
    "fait_divers",     # événement isolé, incident
    "annonce",         # déclaration officielle, politique
    "polémique",       # controverse, débat, accusation
    "procès_justice",  # audience, verdict, mise en examen
    "violence",        # attaque, bagarre, blessés
    "analyse",         # commentaire, analyse, opinion
    "résultat",        # vote, élection, score
    "suivi",           # update d'un événement en cours
]

LOCATION_LIST = [
    "France",
    "Paris_IDF",
    "Régions",
    "International",
    "France+International",
]


# -----------------------------
# Cleaning
# -----------------------------
def normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # replace weird quotes
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    return s


def strip_diacritics(s: str) -> str:
    # utile parfois pour certaines heuristiques; on ne l’applique pas forcément au texte final
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def clean_bandeau(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    s = normalize_unicode(s)

    # Enlève guillemets de liste python si présents
    s = s.strip().strip(",")
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # Supprime artefacts fréquents
    s = re.sub(r"\s+", " ", s)

    # Supprime compteurs finaux "(2)" "(4)" etc
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)

    # Supprime " | " / bouts parasites type "Ancien Ministre ) | 4 Nm Om..."
    # Heuristique: si on a une barre et ensuite beaucoup de tokens bizarres / chiffres => coupe
    s = re.sub(r"\s*\|\s*\d+\s+.*$", "", s).strip()

    # Supprime URLs / domaines
    s = re.sub(r"\bhttps?://\S+\b", "", s)
    s = re.sub(r"\bwww\.\S+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b[A-Za-z0-9.-]+\.(?:fr|com|org|net|io)\b", "", s, flags=re.IGNORECASE)

    # Supprime suites de symboles chelous (OCR/bruit)
    s = re.sub(r"[\\_]{2,}", " ", s)
    s = re.sub(r"[\u200b\u200c\u200d]", "", s)

    # Répare espaces autour de ponctuation
    s = re.sub(r"\s*:\s*", " : ", s)
    s = re.sub(r"\s*;\s*", " ; ", s)
    s = re.sub(r"\s*\?\s*", " ? ", s)
    s = re.sub(r"\s*!\s*", " ! ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Supprime guillemets doublés mal placés
    s = s.replace('""', '"').strip()

    # Si très court / bruit pur
    if len(strip_diacritics(s)) < 4:
        return ""

    return s


# -----------------------------
# LLM enrichment (llama-cpp)
# -----------------------------
@dataclass
class LLMFeatures:
    topic_l1: str = "autre"
    topic_l2: str = "autre"
    event_type: str = "fait_divers"
    location: str = "France"
    entities: List[str] = None
    keywords: List[str] = None
    short: str = ""
    hypotheses: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "topic_l1": self.topic_l1,
            "topic_l2": self.topic_l2,
            "event_type": self.event_type,
            "location": self.location,
            "entities": self.entities or [],
            "keywords": self.keywords or [],
            "short": self.short or "",
            "hypotheses": self.hypotheses or [],
        }


SYSTEM_PROMPT = """Tu es un assistant expert en classification de bandeaux TV français (titres très courts).
Tu réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte avant ou après.

ÉTAPE OBLIGATOIRE — avant de classer, formule 2 hypothèses sur le sujet du bandeau, puis choisis la plus probable pour guider ta classification.

Les champs requis sont (dans cet ordre) :
- "hypotheses": liste de EXACTEMENT 2 chaînes — tes 2 hypothèses sur ce dont parle le bandeau (ex: ["Polémique sur la labellisation des médias par RSF", "Débat sur la censure de CNews"])
- "topic_l1": macro-rubrique, EXACTEMENT une valeur parmi [{l1_list}]
- "topic_l2": sous-thème précis, EXACTEMENT une valeur parmi [{l2_list}]
- "event_type": type d'événement, EXACTEMENT une valeur parmi [{et_list}]
- "location": localisation principale, EXACTEMENT une valeur parmi [{loc_list}]
- "entities": liste (0..3) d'entités nommées (personnes, lieux, institutions, organisations)
- "keywords": liste (0..6) mots-clés (1-3 mots) représentatifs du sujet spécifique
- "short": reformulation très courte (max 8 mots) qui capture l'événement principal

RÈGLES :
- "hypotheses" DOIT être rempli EN PREMIER pour guider ta classification
- Choisis TOUJOURS le topic_l2 le plus spécifique possible
- N'utilise "autre" que si le texte est du bruit pur ou totalement incompréhensible""".format(
    l1_list=", ".join(TOPIC_L1_LIST),
    l2_list=", ".join(TOPIC_L2_LIST),
    et_list=", ".join(EVENT_TYPE_LIST),
    loc_list=", ".join(LOCATION_LIST),
)


def build_llm_messages(text: str, context: str = "") -> list:
    user_content = ""
    if context:
        user_content += f"{context}\n"
    user_content += f"Bandeau TV à annoter : {text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def safe_json_extract(s: str) -> Optional[dict]:
    # extrait le premier bloc JSON {...}
    if not s:
        return None
    # trouve première accolade
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    chunk = s[i : j + 1].strip()
    try:
        return json.loads(chunk)
    except Exception:
        return None


def llm_annotate(
    llm,
    text: str,
    temperature: float = 0.1,
    max_tokens: int = 400,
    context: str = "",
) -> LLMFeatures:
    messages = build_llm_messages(text, context=context)
    out = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
    txt = out["choices"][0]["message"]["content"]
    data = safe_json_extract(txt) or {}

    topic_l1 = str(data.get("topic_l1", "autre")).strip()
    if topic_l1 not in TOPIC_L1_LIST:
        topic_l1 = "autre"

    topic_l2 = str(data.get("topic_l2", "autre")).strip()
    if topic_l2 not in TOPIC_L2_LIST:
        topic_l2 = "autre"

    event_type = str(data.get("event_type", "fait_divers")).strip()
    if event_type not in EVENT_TYPE_LIST:
        event_type = "fait_divers"

    location = str(data.get("location", "France")).strip()
    if location not in LOCATION_LIST:
        location = "France"

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    entities = [str(x).strip() for x in entities if str(x).strip()][:3]

    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(x).strip() for x in keywords if str(x).strip()][:6]

    short = str(data.get("short", "")).strip()
    if len(short) > 60:
        short = short[:60].strip()

    hypotheses = data.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        hypotheses = []
    hypotheses = [str(x).strip() for x in hypotheses if str(x).strip()][:2]

    return LLMFeatures(
        topic_l1=topic_l1,
        topic_l2=topic_l2,
        event_type=event_type,
        location=location,
        entities=entities,
        keywords=keywords,
        short=short,
        hypotheses=hypotheses,
    )


# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    # SentenceTransformers
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # E5 recommande "query: " / "passage: " — ici on fait simple mais correct
    # (on met "passage: " pour tout)
    prefixed = [f"passage: {t}" for t in texts]
    emb = model.encode(prefixed, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


# -----------------------------
# Clustering (Agglomerative + search threshold)
# -----------------------------
def agglomerative_labels_cosine(emb: np.ndarray, distance_threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    # Compat sklearn versions: "metric" (>=1.2+) vs "affinity" (legacy)
    try:
        cl = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        cl = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    return cl.fit_predict(emb)


def count_clusters(labels: np.ndarray) -> int:
    # labels >=0
    s = set(int(x) for x in labels.tolist())
    return len([x for x in s if x >= 0])


def find_threshold_for_target(
    emb: np.ndarray,
    target_min: int,
    target_max: int,
    t_min: float = 0.05,
    t_max: float = 0.60,
    steps: int = 200,
) -> Tuple[float, np.ndarray]:
    best = None
    best_labels = None
    best_gap = 10**9

    for t in np.linspace(t_min, t_max, steps):
        labels = agglomerative_labels_cosine(emb, float(t))
        k = count_clusters(labels)

        # on veut k dans [target_min, target_max], sinon on garde le plus proche
        if target_min <= k <= target_max:
            return float(t), labels

        gap = min(abs(k - target_min), abs(k - target_max))
        if gap < best_gap:
            best_gap = gap
            best = float(t)
            best_labels = labels

    return best, best_labels


def relabel_small_clusters_as_outliers(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    cnt = Counter(labels.tolist())
    new = labels.copy()
    for lab, n in cnt.items():
        if lab >= 0 and n < min_cluster_size:
            new[new == lab] = -1
    # compact relabel 0..K-1 (garde -1)
    kept = sorted([lab for lab in set(new.tolist()) if lab >= 0])
    mapping = {lab: i for i, lab in enumerate(kept)}
    for old, new_id in mapping.items():
        new[new == old] = new_id
    return new


def refine_large_clusters(
    emb: np.ndarray,
    labels: np.ndarray,
    refine_threshold: int,
    sub_target_min: int = 3,
    sub_target_max: int = 7,
    sub_min_cluster_size: int = 2,
) -> np.ndarray:
    """
    Pour chaque cluster dont la taille dépasse `refine_threshold`,
    effectue un sous-clustering avec un seuil plus serré.
    Les nouveaux IDs sont ajoutés après le max ID existant.
    Les items trop petits après sous-clustering deviennent des outliers (-1).
    """
    labels = labels.copy()
    cnt = Counter(labels.tolist())

    for lab, n in sorted(cnt.items(), key=lambda x: -x[1]):
        if lab < 0 or n <= refine_threshold:
            continue

        idx = np.where(labels == lab)[0]
        sub_emb = emb[idx]

        print(f"[refine] Cluster {lab} (n={n}) → sous-clustering en {sub_target_min}-{sub_target_max} sous-clusters...")
        _, sub_labels = find_threshold_for_target(
            sub_emb,
            target_min=sub_target_min,
            target_max=sub_target_max,
            t_min=0.01,
            t_max=0.40,
            steps=200,
        )
        sub_labels = relabel_small_clusters_as_outliers(sub_labels, sub_min_cluster_size)

        n_sub = len(set(sub_labels.tolist()) - {-1})
        print(f"[refine]   → {n_sub} sous-clusters trouvés")

        # Assign new global IDs starting after current max
        max_id = int(max(l for l in labels.tolist() if l >= 0))
        for sub_id in sorted(set(sub_labels.tolist())):
            sub_idx = idx[sub_labels == sub_id]
            if sub_id == -1:
                labels[sub_idx] = -1
            else:
                labels[sub_idx] = max_id + 1 + sub_id

    # Compact relabel so IDs are 0..K-1
    kept = sorted(l for l in set(labels.tolist()) if l >= 0)
    mapping = {lab: i for i, lab in enumerate(kept)}
    new_labels = labels.copy()
    for old, new_id in mapping.items():
        new_labels[labels == old] = new_id

    return new_labels


def pick_representatives(emb: np.ndarray, labels: np.ndarray, texts: List[str], top_n: int = 8) -> Dict[int, List[int]]:
    reps = {}
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        # centroïde (embeddings déjà normalisés)
        c = emb[idx].mean(axis=0, keepdims=True)
        # similarité cos = dot
        sims = (emb[idx] @ c.T).reshape(-1)
        order = np.argsort(-sims)[: min(top_n, len(idx))]
        reps[int(lab)] = [int(idx[i]) for i in order]
    return reps


# -----------------------------
# IO / reporting
# -----------------------------
def write_report_txt(
    path: str,
    df: pd.DataFrame,
    reps: Dict[int, List[int]],
    max_items_per_cluster: int = 12,
):
    with open(path, "w", encoding="utf-8") as f:
        labels = df["cluster_id"].to_numpy()
        cnt = Counter(labels.tolist())
        f.write(f"Total items: {len(df)}\n")
        f.write(f"Cluster sizes (including -1): {dict(cnt)}\n\n")

        for lab, n in sorted([(k, v) for k, v in cnt.items() if k != -1], key=lambda x: -x[1]):
            f.write(f"=== Cluster {lab} | n={n} ===\n")
            sub = df[df.cluster_id == lab]
            f.write(f"topic_l1: {sub['topic_l1'].value_counts().head(3).to_dict()}\n")
            f.write(f"topic_l2: {sub['topic_l2'].value_counts().head(3).to_dict()}\n")
            f.write(f"event_type: {sub['event_type'].value_counts().head(3).to_dict()}\n")

            ridx = reps.get(int(lab), [])
            if ridx:
                f.write("Representatives:\n")
                for i in ridx[: max_items_per_cluster]:
                    row = df.iloc[i]
                    f.write(f"- {row['bandeau_clean']}\n")
                f.write("\n")

            # keywords agrégés
            all_kw = []
            for kws in df[df.cluster_id == lab]["keywords"].tolist():
                if isinstance(kws, str):
                    try:
                        kws = json.loads(kws)
                    except Exception:
                        kws = []
                if isinstance(kws, list):
                    all_kw.extend([str(x).strip().lower() for x in kws if str(x).strip()])
            kw_cnt = Counter(all_kw).most_common(15)
            if kw_cnt:
                f.write("Top keywords:\n")
                f.write(", ".join([f"{k}({v})" for k, v in kw_cnt]) + "\n")
            f.write("\n")

        if -1 in cnt:
            f.write(f"=== Outliers (-1) | n={cnt[-1]} ===\n")
            ex = df[df.cluster_id == -1]["bandeau_clean"].head(40).tolist()
            for s in ex:
                f.write(f"- {s}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="bandeau")
    ap.add_argument("--count_col", default=None, help="colonne fréquence (optionnel)")
    ap.add_argument("--min_count", type=int, default=2, help="garde seulement count>=min_count si count_col fourni")

    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-small")

    ap.add_argument("--llm_model_path", default=None)
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--llm_threads", type=int, default=8)
    ap.add_argument("--llm_ctx", type=int, default=4096)
    ap.add_argument("--llm_temp", type=float, default=0.1)

    ap.add_argument("--target_min", type=int, default=10)
    ap.add_argument("--target_max", type=int, default=15)
    ap.add_argument("--min_cluster_size", type=int, default=6)
    ap.add_argument("--refine_threshold", type=int, default=0,
                    help="Sous-clusterise tout cluster avec plus de N items. 0 = désactivé.")

    ap.add_argument("--out_prefix", default="bandeaux_clusters_llm")
    ap.add_argument("--cache_json", default=None, help="cache annotations LLM (jsonl). si absent: <out_prefix>_llm_cache.jsonl")

    ap.add_argument("--csv_sep", type=str, default=",",
                    help="CSV separator (default: ','). Common: ';' or '\\t'.")
    ap.add_argument("--csv_auto", action="store_true",
                        help="Auto-detect CSV separator using pandas engine='python'. Overrides --csv_sep.")
    ap.add_argument("--csv_encoding", type=str, default="utf-8",
                        help="File encoding for CSV (default: utf-8). Try 'utf-8-sig' or 'latin1' if needed.")

    # Contexte d'actualité (RSS + ledger)
    ap.add_argument("--ledger_path", default="events_ledger.json",
                    help="Chemin du fichier ledger JSON (mémoire glissante d'événements).")
    ap.add_argument("--offline", action="store_true",
                    help="Skip RSS, utilise uniquement le ledger pour le contexte.")
    ap.add_argument("--no_context", action="store_true",
                    help="Désactive entièrement l'injection de contexte (RSS + ledger).")


    args = ap.parse_args()

    # ── Contexte d'actualité ────────────────────────────────────────────────
    ledger_events: List[Dict] = []
    rss_items: List[str] = []
    if not args.no_context:
        from news_context import (
            get_daily_context,
            load_ledger,
            get_active_events,
            fetch_rss,
            build_context_string,
        )
        ledger_events = get_active_events(load_ledger(args.ledger_path))
        if not args.offline:
            print("[context] Récupération des flux RSS...")
            rss_items = fetch_rss()
            print(f"[context] {len(rss_items)} titres RSS récupérés.")
        print(f"[context] {len(ledger_events)} événements actifs dans le ledger.")
        # Affichage du contexte global au démarrage (informatif)
        summary = build_context_string(rss_items, ledger_events)
        if summary:
            print("\n[context] Contexte disponible :")
            print("─" * 60)
            print(summary[:600] + ("..." if len(summary) > 600 else ""))
            print("─" * 60 + "\n")

    read_kwargs = dict(encoding=args.csv_encoding)

    if args.csv_auto:
        # auto sniff delimiter
        df = pd.read_csv(args.input, sep=None, engine="python", **read_kwargs)
    else:
        df = pd.read_csv(args.input, sep=args.csv_sep, **read_kwargs)   
    if args.text_col not in df.columns:
        raise ValueError(f"Missing text_col={args.text_col}. Columns: {df.columns.tolist()}")

    # count col optional
    if args.count_col and args.count_col in df.columns:
        df[args.count_col] = pd.to_numeric(df[args.count_col], errors="coerce").fillna(1).astype(int)
        df = df[df[args.count_col] >= args.min_count].copy()
    else:
        df["__count__"] = 1
        args.count_col = "__count__"

    # cleaning
    df["bandeau_raw"] = df[args.text_col].astype(str)
    df["bandeau_clean"] = df["bandeau_raw"].map(clean_bandeau)
    df = df[df["bandeau_clean"].str.len() > 0].copy()

    texts = df["bandeau_clean"].tolist()

    # --- LLM features (optional)
    use_llm = (not args.no_llm) and args.llm_model_path
    cache_path = args.cache_json or f"{args.out_prefix}_llm_cache.jsonl"
    feat_by_text: Dict[str, LLMFeatures] = {}

    if use_llm:
        from llama_cpp import Llama

        llm = Llama(
            model_path=args.llm_model_path,
            n_ctx=args.llm_ctx,                 # ex: 4096
            n_threads=args.llm_threads,         # ex: 8 (CPU threads)
            n_gpu_layers=-1,                    # <- OFFLOAD MAX to Metal (essaye -1 d'abord)
            n_batch=512,                        # bon compromis sur M3
            use_mmap=True,                      # ok sur macOS
            use_mlock=False,                    # évite de bloquer la RAM
            f16_kv=True,                        # KV cache en fp16 (moins de RAM, souvent plus rapide)
            verbose=True,                      # mets True si tu veux voir le détail d'offload
        )

        # load cache if exists
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("text", "")
                        feat = obj.get("feat", {})
                        lf = LLMFeatures(
                            topic_l1=feat.get("topic_l1", "autre"),
                            topic_l2=feat.get("topic_l2", "autre"),
                            event_type=feat.get("event_type", "fait_divers"),
                            location=feat.get("location", "France"),
                            entities=feat.get("entities", []),
                            keywords=feat.get("keywords", []),
                            short=feat.get("short", ""),
                            hypotheses=feat.get("hypotheses", []),
                        )
                        feat_by_text[t] = lf
                    except Exception:
                        pass

        # annotate missing
        to_do = [t for t in texts if t not in feat_by_text]
        if to_do:
            # Import targeted context builder once
            if not args.no_context:
                from news_context import match_ledger_events, build_targeted_context
            with open(cache_path, "a", encoding="utf-8") as f:
                for t in to_do:
                    # ── Pass 1 : entity pre-lookup → targeted context ────────
                    if not args.no_context and ledger_events:
                        matched = match_ledger_events(t, ledger_events)
                        targeted_ctx = build_targeted_context(matched, rss_items if matched else None)
                        if matched:
                            print(f"\n[context] Entités trouvées dans le ledger pour : {t}")
                            for ev in matched:
                                print(f"  → {ev['label'][:80]}...")
                    else:
                        targeted_ctx = ""

                    # ── Pass 2 : LLM classification with targeted context ────
                    print(f"\n LLM — évaluation du bandeau :\n→ {t}\n", flush=True)
                    lf = llm_annotate(llm, t, temperature=args.llm_temp, context=targeted_ctx)
                    feat_by_text[t] = lf
                    f.write(json.dumps({"text": t, "feat": lf.to_dict()}, ensure_ascii=False) + "\n")

        df["topic_l1"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).topic_l1)
        df["topic_l2"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).topic_l2)
        df["event_type"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).event_type)
        df["location"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).location)
        df["entities"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).entities)
        df["keywords"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).keywords)
        df["short"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).short)
        df["hypotheses"] = df["bandeau_clean"].map(lambda t: feat_by_text.get(t, LLMFeatures()).hypotheses)

        # ── Mise à jour du ledger avec les annotations d'aujourd'hui ────────
        if not args.no_context:
            from news_context import update_ledger_from_annotations  # noqa: F811
            freq_col = args.count_col if args.count_col in df.columns else "__count__"
            annotations_for_ledger = [
                {
                    "entities": row["entities"],
                    "short": row["short"],
                    "domain": row["topic_l1"],  # ledger uses generic "domain" key
                    "freq": int(row[freq_col]) if freq_col in df.columns else 1,
                }
                for _, row in df.iterrows()
            ]
            update_ledger_from_annotations(annotations_for_ledger, ledger_path=args.ledger_path)
    else:
        df["topic_l1"] = "autre"
        df["topic_l2"] = "autre"
        df["event_type"] = "fait_divers"
        df["location"] = "France"
        df["entities"] = [[] for _ in range(len(df))]
        df["keywords"] = [[] for _ in range(len(df))]
        df["short"] = ""
        df["hypotheses"] = [[] for _ in range(len(df))]

    # build enriched text for embeddings
    def build_enriched(row) -> str:
        ent = row["entities"] if isinstance(row["entities"], list) else []
        kw = row["keywords"] if isinstance(row["keywords"], list) else []
        hyp = row["hypotheses"] if isinstance(row["hypotheses"], list) else []
        parts = [
            f"Titre: {row['bandeau_clean']}",
            f"L1: {row['topic_l1']}",
            f"L2: {row['topic_l2']}",
            f"Type: {row['event_type']}",
            f"Lieu: {row['location']}",
        ]
        if row.get("short"):
            parts.append(f"Résumé: {row['short']}")
        if hyp:
            parts.append(f"Hypothèse: {hyp[0]}")
        if ent:
            parts.append("Entités: " + ", ".join(ent[:3]))
        if kw:
            parts.append("Mots-clés: " + ", ".join(kw[:6]))
        return " | ".join(parts)

    df["enriched"] = df.apply(build_enriched, axis=1)
    enriched_texts = df["enriched"].tolist()

    # embeddings
    emb = embed_texts(enriched_texts, args.embed_model)

    # clustering : find threshold for 10-15 clusters
    thr, labels = find_threshold_for_target(
        emb,
        target_min=args.target_min,
        target_max=args.target_max,
        t_min=0.05,
        t_max=0.60,
        steps=220,
    )

    labels = relabel_small_clusters_as_outliers(labels, min_cluster_size=args.min_cluster_size)

    if args.refine_threshold > 0:
        labels = refine_large_clusters(
            emb, labels,
            refine_threshold=args.refine_threshold,
            sub_target_min=3,
            sub_target_max=7,
            sub_min_cluster_size=2,
        )

    df["cluster_id"] = labels.astype(int)

    # representatives
    reps = pick_representatives(emb, df["cluster_id"].to_numpy(), df["bandeau_clean"].tolist(), top_n=10)

    # Export CSV (on sérialise list->json pour rester propre)
    out_csv = f"{args.out_prefix}.csv"
    out_txt = f"{args.out_prefix}.txt"

    df_out = df.copy()
    df_out["entities"] = df_out["entities"].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]")
    df_out["keywords"] = df_out["keywords"].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]")
    df_out["hypotheses"] = df_out["hypotheses"].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else "[]")

    cols = [
        "cluster_id",
        "bandeau_clean",
        args.count_col,
        "topic_l1",
        "topic_l2",
        "event_type",
        "location",
        "entities",
        "keywords",
        "short",
        "hypotheses",
        "bandeau_raw",
    ]
    cols = [c for c in cols if c in df_out.columns]
    df_out[cols].to_csv(out_csv, index=False, encoding="utf-8")

    # Report TXT
    write_report_txt(out_txt, df_out, reps, max_items_per_cluster=12)

    # Console summary
    cnt = Counter(df_out["cluster_id"].tolist())
    k = len([c for c in cnt.keys() if c != -1])
    print(f"Input: {args.input}")
    print(f"Items kept: {len(df_out)} (min_count={args.min_count} on {args.count_col})")
    print(f"Chosen distance_threshold ≈ {thr:.4f}")
    print(f"Clusters (excluding -1): {k} | Outliers (-1): {cnt.get(-1, 0)}")
    print(f"Cluster sizes (top): {sorted([(c,n) for c,n in cnt.items() if c!=-1], key=lambda x:-x[1])[:10]}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_txt}")
    print(f"LLM cache: {cache_path}")


if __name__ == "__main__":
    main()
