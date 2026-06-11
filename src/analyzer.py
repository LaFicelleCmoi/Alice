"""Analyses complémentaires : statistiques de lisibilité, sentiment, entités, mots-clés.

Couvre également la diversité lexicale (plusieurs mesures), la séparation
personnages/lieux et la « fiche de livre » agrégée exportable en JSON.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass

from .preprocessing import ensure_nltk_resources, sentences as split_sentences, tokenize


POSITIVE_WORDS = {
    "good", "great", "happy", "joy", "love", "beautiful", "wonderful", "kind",
    "noble", "brave", "peace", "hope", "free", "smile", "laugh", "warm", "bright",
    "delight", "pleasure", "victory", "triumph", "tender", "gentle", "sweet",
    "honor", "honour", "true", "faithful", "courage", "splendid", "glad",
}
NEGATIVE_WORDS = {
    "bad", "sad", "angry", "fear", "hate", "death", "dead", "kill", "cry",
    "terrible", "horrible", "evil", "dark", "pain", "suffer", "wound", "blood",
    "weep", "sorrow", "grief", "doom", "tragic", "lost", "lonely", "cruel",
    "betray", "fall", "broken", "destroy", "war", "enemy", "afraid",
}


@dataclass
class TextStats:
    word_count: int
    sentence_count: int
    unique_words: int
    type_token_ratio: float
    avg_sentence_length: float
    avg_word_length: float


PERSON_TITLES = {
    "mr", "mrs", "ms", "miss", "sir", "lord", "lady", "dr", "doctor",
    "professor", "captain", "major", "colonel", "general", "king", "queen",
    "prince", "princess", "duke", "duchess", "father", "mother", "uncle",
    "aunt", "monsieur", "madame", "mademoiselle", "saint", "st",
}
DIALOGUE_VERBS = {"said", "asked", "replied", "answered", "shouted", "whispered",
                  "exclaimed", "cried", "muttered", "murmured", "thought"}
PLACE_PREPS = {"in", "at", "to", "from", "near", "into", "toward", "towards",
               "across", "through", "around", "over"}
PLACE_SUFFIXES = ("ville", "town", "shire", "borough", "burg", "ford", "field",
                  "land", "wood", "wick", "ton", "ham", "port", "haven",
                  "mouth", "bridge")


@dataclass
class SentimentResult:
    positive_hits: int
    negative_hits: int
    polarity: float
    label: str


def text_statistics(text: str, lang: str = "english") -> TextStats:
    """Statistiques basiques utiles pour caractériser le style d'un livre."""
    sents = split_sentences(text, lang=lang)
    tokens = tokenize(text, lang=lang, remove_stopwords=False, lemmatize=False)
    if not tokens:
        return TextStats(0, len(sents), 0, 0.0, 0.0, 0.0)

    unique = len(set(tokens))
    avg_sent_len = len(tokens) / max(len(sents), 1)
    avg_word_len = sum(len(t) for t in tokens) / len(tokens)

    return TextStats(
        word_count=len(tokens),
        sentence_count=len(sents),
        unique_words=unique,
        type_token_ratio=unique / len(tokens),
        avg_sentence_length=avg_sent_len,
        avg_word_length=avg_word_len,
    )


def analyze_sentiment(text: str, lang: str = "english") -> SentimentResult:
    """Polarité par lexique (positif/négatif). Marche bien pour une vue d'ensemble."""
    tokens = tokenize(text, lang=lang, remove_stopwords=True, lemmatize=True)
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    polarity = (pos - neg) / total if total else 0.0

    if polarity > 0.15:
        label = "plutôt positif"
    elif polarity < -0.15:
        label = "plutôt négatif"
    else:
        label = "neutre / mélangé"

    return SentimentResult(positive_hits=pos, negative_hits=neg, polarity=polarity, label=label)


def sentiment_arc(text: str, segments: int = 20, lang: str = "english") -> list[float]:
    """Découpe le texte en `segments` parts égales et renvoie la polarité de chacune.

    Permet de tracer une « courbe émotionnelle » du livre.
    """
    words = text.split()
    if len(words) < segments:
        return [analyze_sentiment(text, lang=lang).polarity]
    size = len(words) // segments
    return [
        analyze_sentiment(" ".join(words[i:i + size]), lang=lang).polarity
        for i in range(0, len(words), size)
    ][:segments]


def top_keywords(text: str, n: int = 20, lang: str = "english") -> list[tuple[str, int]]:
    """Renvoie les `n` mots les plus fréquents (hors stopwords)."""
    tokens = tokenize(text, lang=lang, remove_stopwords=True, lemmatize=True)
    return Counter(tokens).most_common(n)


# Extraction d'entités basée sur la casse — simple mais étonnamment robuste sur les
# romans anglais. Pour faire mieux, brancher spaCy sur le module ; ce n'est pas
# obligatoire pour la démo.
NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})(?:\s+([A-Z][a-z]{2,}))?\b")


def _iter_candidate_entities(text: str):
    """Itère sur (mot_précédent, entité) — utile pour la désambiguïsation lieu/personne."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        words = sent.strip().split()
        if len(words) < 2:
            continue
        # On saute le tout premier mot (toujours capitalisé).
        for i in range(1, len(words)):
            joined = " ".join(words[i:])
            m = NAME_RE.match(joined)
            if not m:
                continue
            full = " ".join(p for p in m.groups() if p)
            prev = words[i - 1].lower().strip(".,;:!?\"'()")
            yield prev, full


def extract_named_entities(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """Heuristique de détection des noms propres : tokens capitalisés non en début de phrase.

    Pour distinguer personnages et lieux, voir `extract_characters` et `extract_places`.
    """
    counter: Counter[str] = Counter()
    for _, full in _iter_candidate_entities(text):
        counter[full] += 1
    return counter.most_common(top_n)


def extract_characters(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """Tente d'extraire les **personnages** d'un livre.

    Heuristiques :
    - mot capitalisé précédé d'un titre (Mr, Lady, Captain, …) → personnage
    - mot capitalisé précédé/suivi d'un verbe de dialogue (« said », « asked ») → personnage
    - prénom suivi d'un nom (deux mots capitalisés) → personnage
    Les candidats classés comme lieu par `extract_places` sont retirés.
    """
    counter: Counter[str] = Counter()
    place_set = {n for n, _ in extract_places(text, top_n=200)}

    for prev, full in _iter_candidate_entities(text):
        is_person = False
        if prev in PERSON_TITLES:
            is_person = True
        elif prev in DIALOGUE_VERBS:
            is_person = True
        elif " " in full:  # « John Smith » → très souvent un personnage
            is_person = True
        if is_person and full not in place_set:
            counter[full] += 1
    return counter.most_common(top_n)


def extract_places(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """Tente d'extraire les **lieux** d'un livre.

    Heuristiques :
    - mot capitalisé précédé d'une préposition de lieu (« in », « at », « to », …)
    - mot capitalisé terminant par un suffixe toponymique (-ville, -ton, -shire, …)
    """
    counter: Counter[str] = Counter()
    for prev, full in _iter_candidate_entities(text):
        last = full.split()[-1].lower()
        is_place = False
        if prev in PLACE_PREPS:
            is_place = True
        elif last.endswith(PLACE_SUFFIXES):
            is_place = True
        if is_place:
            counter[full] += 1
    return counter.most_common(top_n)


def lexical_diversity(text: str, lang: str = "english") -> dict[str, float]:
    """Renvoie un dictionnaire de **≥5 mesures de diversité lexicale**.

    - **TTR** (Type-Token Ratio) : V / N — baisse avec la longueur du texte.
    - **RTTR** (Root TTR, Guiraud) : V / √N — corrige partiellement la longueur.
    - **CTTR** (Corrected TTR, Carroll) : V / √(2N) — autre correction classique.
    - **Herdan_C** (log-TTR) : log V / log N — robuste à la longueur.
    - **Maas_a2** : (log N − log V) / log² N — plus stable que TTR sur longs textes.
    - **Yule_K** : mesure de concentration du vocabulaire (faible = vocabulaire varié).
    - **Hapax_ratio** : proportion de mots n'apparaissant qu'une seule fois.

    Toutes les valeurs sont des floats ; renvoie 0.0 quand le texte est trop court.
    """
    tokens = tokenize(text, lang=lang, remove_stopwords=False, lemmatize=False)
    n = len(tokens)
    if n < 2:
        return {k: 0.0 for k in
                ("TTR", "RTTR", "CTTR", "Herdan_C", "Maas_a2", "Yule_K", "Hapax_ratio")}

    freqs = Counter(tokens)
    v = len(freqs)
    ttr = v / n
    rttr = v / math.sqrt(n)
    cttr = v / math.sqrt(2 * n)
    herdan_c = math.log(v) / math.log(n) if n > 1 else 0.0
    maas_a2 = (math.log(n) - math.log(v)) / (math.log(n) ** 2) if v < n else 0.0
    # Yule's K = 10^4 * (Σ m² f_m  −  N) / N²
    fm_counts: Counter[int] = Counter(freqs.values())
    sigma = sum(m * m * fm for m, fm in fm_counts.items())
    yule_k = 10_000 * (sigma - n) / (n * n)
    hapax = sum(1 for c in freqs.values() if c == 1) / v

    return {
        "TTR": ttr,
        "RTTR": rttr,
        "CTTR": cttr,
        "Herdan_C": herdan_c,
        "Maas_a2": maas_a2,
        "Yule_K": yule_k,
        "Hapax_ratio": hapax,
    }


def book_sheet(book, lang: str = "english", summary_sentences: int = 5) -> dict:
    """**Fiche de livre** — dictionnaire regroupant toutes les infos NLP d'un livre.

    Couvre : métadonnées, statistiques de style, diversité lexicale, sentiment,
    top mots-clés, personnages, lieux, genres, sujets latents et résumé court.
    Pratique pour exporter en JSON ou afficher en tant que carte récapitulative.
    """
    from . import classifier, summarizer  # import retardé pour éviter les cycles

    text = book.text
    stats = text_statistics(text, lang=lang)
    sent = analyze_sentiment(text, lang=lang)
    try:
        summary = summarizer.hierarchical_summary(
            text, final_sentences=summary_sentences, language=lang,
        ).sentences
    except Exception:
        summary = []

    return {
        "metadata": {
            "title": book.title,
            "author": book.author,
            "source": book.source,
            "word_count": book.word_count,
            "char_count": book.char_count,
        },
        "stats": asdict(stats),
        "lexical_diversity": lexical_diversity(text, lang=lang),
        "sentiment": asdict(sent),
        "top_keywords": top_keywords(text, n=20, lang=lang),
        "characters": extract_characters(text, top_n=10),
        "places": extract_places(text, top_n=10),
        "genres": [
            {"genre": g.genre, "score": g.score}
            for g in classifier.classify_by_keywords(text, top_k=5, lang=lang)
        ],
        "topics": [
            {"top_words": t.top_words, "weight": t.weight}
            for t in classifier.discover_topics(text, n_topics=5, lang=lang)
        ],
        "summary": summary,
    }


def full_report(text: str, lang: str = "english") -> dict:
    """Rapport complet — pratique pour exporter en JSON depuis le CLI."""
    ensure_nltk_resources()
    return {
        "stats": asdict(text_statistics(text, lang=lang)),
        "lexical_diversity": lexical_diversity(text, lang=lang),
        "sentiment": asdict(analyze_sentiment(text, lang=lang)),
        "sentiment_arc": sentiment_arc(text, segments=20, lang=lang),
        "top_keywords": top_keywords(text, n=20, lang=lang),
        "named_entities": extract_named_entities(text, top_n=15),
        "characters": extract_characters(text, top_n=15),
        "places": extract_places(text, top_n=15),
    }
