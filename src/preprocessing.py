"""Prétraitement : nettoyage, tokenisation, lemmatisation, découpage en chapitres."""

from __future__ import annotations

import re
import string
from functools import lru_cache

import nltk

NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger"]


def ensure_nltk_resources() -> None:
    """Télécharge à la demande les ressources NLTK manquantes."""
    for resource in NLTK_RESOURCES:
        try:
            if resource in {"punkt", "punkt_tab"}:
                nltk.data.find(f"tokenizers/{resource}")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource in {"wordnet", "omw-1.4"}:
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=4)
def _stopwords(lang: str) -> set[str]:
    ensure_nltk_resources()
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words(lang))
    except OSError:
        return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer():
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalise les espaces, supprime les retours chariot multiples."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def sentences(text: str, lang: str = "english") -> list[str]:
    ensure_nltk_resources()
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text, language=lang)


def tokenize(
    text: str,
    lang: str = "english",
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_len: int = 2,
) -> list[str]:
    """Tokenise un texte : minuscules, ponctuation retirée, stopwords/lemmes optionnels."""
    ensure_nltk_resources()
    from nltk.tokenize import word_tokenize

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation + "“”‘’«»—…"))
    tokens = word_tokenize(text, language=lang)
    tokens = [t for t in tokens if t.isalpha() and len(t) >= min_len]
    if remove_stopwords:
        sw = _stopwords(lang)
        tokens = [t for t in tokens if t not in sw]
    if lemmatize:
        lem = _lemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    return tokens


CHAPTER_RE = re.compile(
    r"^\s*(chapter|chapitre|book|livre|part|partie)\s+[ivxlcdm\d]+",
    re.IGNORECASE | re.MULTILINE,
)


def split_into_chapters(text: str) -> list[str]:
    """Découpe un livre en chapitres ; retourne le texte entier si rien n'est détecté."""
    matches = list(CHAPTER_RE.finditer(text))
    if len(matches) < 2:
        return [text]
    chapters: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[m.start():end].strip()
        if chunk:
            chapters.append(chunk)
    return chapters


def chunk_by_words(text: str, words_per_chunk: int = 800) -> list[str]:
    """Coupe un texte en blocs de N mots — utile pour résumer en plusieurs passes."""
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]
