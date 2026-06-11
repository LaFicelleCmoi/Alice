"""Classification thématique : TF-IDF, K-Means par genre, LDA pour les sujets latents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .loader import Book
from .preprocessing import split_into_chapters, tokenize


GENRE_KEYWORDS: dict[str, list[str]] = {
    "aventure": ["adventure", "journey", "quest", "treasure", "island", "ship", "voyage", "explore"],
    "amour": ["love", "heart", "marriage", "kiss", "passion", "romance", "tender", "lover"],
    "mystère": ["mystery", "murder", "detective", "clue", "secret", "investigation", "suspect", "crime"],
    "science-fiction": ["space", "robot", "future", "machine", "alien", "planet", "technology", "scientific"],
    "fantasy": ["magic", "dragon", "wizard", "spell", "kingdom", "sword", "elf", "enchanted"],
    "guerre": ["war", "battle", "soldier", "army", "enemy", "weapon", "fight", "general"],
    "philosophie": ["truth", "soul", "existence", "moral", "virtue", "reason", "knowledge", "freedom"],
    "horreur": ["fear", "terror", "ghost", "death", "horror", "shadow", "blood", "scream"],
}


@dataclass
class GenreScore:
    genre: str
    score: float


@dataclass
class TopicResult:
    topic_id: int
    top_words: list[str]
    weight: float


def tokenize_for_vectors(text: str, lang: str = "english") -> list[str]:
    return tokenize(text, lang=lang, remove_stopwords=True, lemmatize=True)


def _identity(tokens: list[str]) -> list[str]:
    return tokens


def classify_by_keywords(text: str, top_k: int = 3, lang: str = "english") -> list[GenreScore]:
    """Score chaque genre en fonction de la fréquence relative de ses mots-clés."""
    tokens = tokenize_for_vectors(text, lang=lang)
    if not tokens:
        return []
    total = len(tokens)
    counter = {}
    for tok in tokens:
        counter[tok] = counter.get(tok, 0) + 1

    scored: list[GenreScore] = []
    for genre, keywords in GENRE_KEYWORDS.items():
        hits = sum(counter.get(k, 0) for k in keywords)
        scored.append(GenreScore(genre=genre, score=hits / total * 1000))

    scored.sort(key=lambda g: g.score, reverse=True)
    return scored[:top_k]


def cluster_corpus(
    books: list[Book],
    n_clusters: int = 3,
    lang: str = "english",
) -> dict:
    """Clusterise un corpus de livres avec TF-IDF + K-Means."""
    if len(books) < n_clusters:
        n_clusters = max(1, len(books))

    pre_tokenized = [tokenize_for_vectors(b.text, lang=lang) for b in books]
    vectorizer = TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        max_features=5000,
        min_df=1,
    )
    X = vectorizer.fit_transform(pre_tokenized)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    centroids = km.cluster_centers_
    top_terms_per_cluster = []
    for c in range(n_clusters):
        top_idx = centroids[c].argsort()[-10:][::-1]
        top_terms_per_cluster.append(feature_names[top_idx].tolist())

    similarity = cosine_similarity(X)

    return {
        "labels": labels.tolist(),
        "top_terms": top_terms_per_cluster,
        "similarity_matrix": similarity.tolist(),
        "vocabulary_size": len(feature_names),
        "books": [{"title": b.title, "author": b.author, "cluster": int(labels[i])}
                  for i, b in enumerate(books)],
    }


def discover_topics(
    text: str,
    n_topics: int = 5,
    n_top_words: int = 8,
    lang: str = "english",
) -> list[TopicResult]:
    """Découvre `n_topics` sujets latents via LDA sur des fenêtres du livre."""
    from .preprocessing import chunk_by_words

    chunks = chunk_by_words(text, words_per_chunk=500)
    if len(chunks) < 2:
        chunks = [text]

    pre_tokenized = [tokenize_for_vectors(c, lang=lang) for c in chunks]
    vectorizer = CountVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        max_features=2000,
        min_df=2,
    )
    try:
        X = vectorizer.fit_transform(pre_tokenized)
    except ValueError:
        return []

    n_topics = min(n_topics, X.shape[0])
    if n_topics < 1:
        return []

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
    )
    lda.fit(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topics: list[TopicResult] = []
    for k, component in enumerate(lda.components_):
        top_idx = component.argsort()[-n_top_words:][::-1]
        weight = float(component.sum() / lda.components_.sum())
        topics.append(TopicResult(
            topic_id=k,
            top_words=feature_names[top_idx].tolist(),
            weight=weight,
        ))
    topics.sort(key=lambda t: t.weight, reverse=True)
    return topics


def topics_per_section(
    text: str,
    n_topics_per_section: int = 3,
    n_top_words: int = 6,
    lang: str = "english",
) -> list[dict]:
    """Extrait les sujets principaux de **chaque section** d'un livre."""

    Découpe le texte en chapitres (via `split_into_chapters`) ; pour chaque
    chapitre, applique LDA et renvoie le top-k de ses sujets. Si aucun chapitre
    n'est détecté, le livre est découpé en blocs de mots équivalents.
    """
    sections = split_into_chapters(text)
    if len(sections) < 2:
        from .preprocessing import chunk_by_words
        sections = chunk_by_words(text, words_per_chunk=2000)

    out: list[dict] = []
    for i, section in enumerate(sections):
        topics = discover_topics(
            section,
            n_topics=n_topics_per_section,
            n_top_words=n_top_words,
            lang=lang,
        )
        out.append({
            "section_id": i,
            "preview": section.strip().splitlines()[0][:80] if section.strip() else "",
            "word_count": len(section.split()),
            "topics": [{"top_words": t.top_words, "weight": t.weight} for t in topics],
        })
    return out


def find_similar_books(
    target: Book,
    corpus: list[Book],
    top_n: int = 5,
    lang: str = "english",
) -> list[dict]:
    """Recommande les **livres les plus similaires** au livre cible."""

    Vectorise le corpus complet (cible incluse, mais retirée des résultats) en
    TF-IDF puis renvoie les `top_n` livres ayant la plus grande similarité
    cosinus avec la cible. Le corpus n'a pas besoin d'inclure la cible : elle
    est ajoutée automatiquement si elle manque.
    """
    if not corpus:
        return []

    books = list(corpus)
    if not any(b.source == target.source for b in books):
        books.append(target)

    pre_tokenized = [tokenize_for_vectors(b.text, lang=lang) for b in books]
    vectorizer = TfidfVectorizer(
        tokenizer=_identity,
        preprocessor=_identity,
        token_pattern=None,
        max_features=5000,
        min_df=1,
    )
    X = vectorizer.fit_transform(pre_tokenized)
    target_idx = next(i for i, b in enumerate(books) if b.source == target.source)

    sims = cosine_similarity(X[target_idx], X).ravel()
    ranked = sorted(
        ((i, float(sims[i])) for i in range(len(books)) if i != target_idx),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return [
        {"title": books[i].title, "author": books[i].author,
         "source": books[i].source, "similarity": s}
        for i, s in ranked
    ]
