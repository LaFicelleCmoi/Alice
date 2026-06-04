#!/usr/bin/env python3
"""Outils NLP en ligne de commande : catalogue Gutenberg, nettoyage, tokenisation,
POS tagging, normalisation (lemmatisation / stemming)."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path
from urllib.request import urlopen

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
DATA_DIR = Path(__file__).parent / "data"
CATALOG_PATH = DATA_DIR / "pg_catalog.csv"
BOOKS_DIR = DATA_DIR / "books"

GUT_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)
GUT_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- catalog --

def _ensure_catalog() -> Path:
    """Télécharge le catalogue Gutenberg si absent ; renvoie son chemin local."""
    if CATALOG_PATH.exists():
        return CATALOG_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(CATALOG_URL, timeout=60) as resp:
        compressed = resp.read()
    CATALOG_PATH.write_bytes(gzip.decompress(compressed))
    return CATALOG_PATH


def info(book_id: int) -> dict:
    """Renvoie les informations du livre depuis le catalogue Gutenberg."""
    catalog = _ensure_catalog()
    target = str(book_id)
    with catalog.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Text#") == target:
                return {
                    "id": row["Text#"],
                    "title": row.get("Title", "").strip(),
                    "authors": row.get("Authors", "").strip(),
                    "bookshelves": row.get("Bookshelves", "").strip(),
                }
    raise ValueError(f"Book ID {book_id} not found in Gutenberg catalog.")


def download(book_id: int) -> Path:
    """Télécharge le livre en Plain Text UTF-8 et le sauvegarde localement."""
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    dest = BOOKS_DIR / f"gutenberg_{book_id}.txt"
    with urlopen(BOOK_URL.format(book_id=book_id), timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


# ------------------------------------------------------------------ clean --

def clean(text: str, lower: bool = False) -> str:
    """Supprime l'en-tête/pied de page Gutenberg et normalise les espaces."""
    start = GUT_START_RE.search(text)
    end = GUT_END_RE.search(text)
    if start and end and start.end() < end.start():
        text = text[start.end():end.start()]
    elif start:
        text = text[start.end():]
    elif end:
        text = text[:end.start()]

    text = re.sub(r"[ \t]+", " ", text).strip()
    if lower:
        text = text.lower()
    return text


# --------------------------------------------------------------- tokenize --

_NLTK_READY = False


def _ensure_nltk() -> None:
    """Télécharge les ressources NLTK nécessaires à la demande."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for res, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
    _NLTK_READY = True


def tokenize(text: str, sent: bool = False, stop: bool = False, punct: bool = False) -> list[str]:
    """Tokenise un texte en mots (par défaut) ou en phrases (--sent)."""
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize, word_tokenize

    tokens = sent_tokenize(text) if sent else word_tokenize(text)

    if punct:
        tokens = [t for t in tokens if any(c.isalnum() for c in t)]
    if stop:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t.lower() not in sw]
    return tokens


# ---------------------------------------------------------------- postag --

def postag(raw: str) -> list[tuple[str, str]]:
    """Assigne des POS tags à une liste de tokens séparés par des espaces."""
    _ensure_nltk()
    from nltk import pos_tag
    tokens = raw.split()
    return pos_tag(tokens)


# ------------------------------------------------------------- normalize --

_PENN_TO_WORDNET = {"J": "a", "V": "v", "N": "n", "R": "r"}


def _wordnet_pos(tag: str) -> str:
    return _PENN_TO_WORDNET.get(tag[:1], "n")


def normalize(raw: str, stem: bool = False) -> list[str]:
    """Normalise une liste de tokens : lemmatisation (défaut) ou stemming (--stem)."""
    _ensure_nltk()
    tokens = raw.split()

    if stem:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return [stemmer.stem(t) for t in tokens]

    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    tagged = pos_tag(tokens)
    return [lem.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]


# -------------------------------------------------------------------- CLI --

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools.py",
        description="NLP tooling: Gutenberg catalog, cleaning, tokenization, POS tagging, normalization.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--info", type=int, metavar="ID", help="show metadata for a Gutenberg book")
    action.add_argument("--download", type=int, metavar="ID", help="download a Gutenberg book as UTF-8 text")
    action.add_argument("--clean", metavar="TEXT", help="clean a raw text")
    action.add_argument("--tokenize", metavar="TEXT", help="tokenize a text")
    action.add_argument("--postag", metavar="TOKENS", help="POS-tag space-separated tokens")
    action.add_argument("--normalize", metavar="TOKENS", help="normalize space-separated tokens")

    parser.add_argument("--lower", action="store_true", help="lowercase (with --clean)")
    parser.add_argument("--sent", action="store_true", help="sentence tokenization (with --tokenize)")
    parser.add_argument("--stop", action="store_true", help="drop stopwords (with --tokenize)")
    parser.add_argument("--punct", action="store_true", help="drop punctuation (with --tokenize)")
    parser.add_argument("--stem", action="store_true", help="use stemming instead of lemmatization (with --normalize)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.info is not None:
            print(info(args.info))
        elif args.download is not None:
            dest = download(args.download)
            print(f"Saved to {dest}")
        elif args.clean is not None:
            print(clean(args.clean, lower=args.lower))
        elif args.tokenize is not None:
            print(tokenize(args.tokenize, sent=args.sent, stop=args.stop, punct=args.punct))
        elif args.postag is not None:
            print(postag(args.postag))
        elif args.normalize is not None:
            print(normalize(args.normalize, stem=args.stem))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
