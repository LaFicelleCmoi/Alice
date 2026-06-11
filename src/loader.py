"""Chargement de livres depuis Project Gutenberg ou un fichier local."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

# Marqueurs Gutenberg utilisés pour découper l'en-tête et le pied de page légal.
START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]+\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]+\*\*\*", re.IGNORECASE)


@dataclass
class Book:
    title: str
    author: str = "Inconnu"
    text: str = ""
    source: str = "local"
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


def _strip_gutenberg_boilerplate(raw: str) -> str:
    start = START_RE.search(raw)
    end = END_RE.search(raw)
    if start and end and start.end() < end.start():
        return raw[start.end(): end.start()].strip()
    return raw.strip()


def _parse_header(raw: str) -> tuple[str, str]:
    title = "Sans titre"
    author = "Inconnu"
    for line in raw.splitlines()[:60]:
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif line.lower().startswith("author:"):
            author = line.split(":", 1)[1].strip()
    return title, author


class BookLoadError(RuntimeError):
    """Erreur explicite quand on ne parvient pas à charger un livre."""


def load_from_gutenberg(book_id: int, cache_dir: str | Path = "data/books") -> Book:
    """Télécharge (ou lit le cache) un livre Project Gutenberg par son ID.

    Lève `BookLoadError` avec un message clair si l'ID est invalide ou si
    la connexion échoue.
    """
    if not isinstance(book_id, int) or book_id <= 0:
        raise BookLoadError(f"ID Gutenberg invalide : {book_id!r} (entier positif attendu)")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"gutenberg_{book_id}.txt"

    if cache_file.exists():
        raw = cache_file.read_text(encoding="utf-8", errors="ignore")
    else:
        url = GUTENBERG_URL.format(book_id=book_id)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise BookLoadError(
                f"Téléchargement impossible (HTTP {e.response.status_code}) — "
                f"l'ID Gutenberg {book_id} n'existe peut-être pas."
            ) from e
        except requests.RequestException as e:
            raise BookLoadError(
                f"Erreur réseau au téléchargement de Gutenberg #{book_id} : {e}"
            ) from e
        raw = resp.text
        cache_file.write_text(raw, encoding="utf-8")

    title, author = _parse_header(raw)
    body = _strip_gutenberg_boilerplate(raw)
    return Book(
        title=title,
        author=author,
        text=body,
        source=f"gutenberg:{book_id}",
        metadata={"book_id": book_id},
    )


def load_from_file(path: str | Path) -> Book:
    """Charge un livre depuis un fichier texte local.

    Lève `BookLoadError` si le fichier n'existe pas ou n'est pas lisible.
    """
    p = Path(path)
    if not p.exists():
        raise BookLoadError(f"Fichier introuvable : {p}")
    if not p.is_file():
        raise BookLoadError(f"Ce chemin n'est pas un fichier : {p}")
    try:
        raw = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        raise BookLoadError(f"Lecture impossible de {p} : {e}") from e

    title, author = _parse_header(raw)
    if title == "Sans titre":
        title = p.stem.replace("_", " ").title()
    return Book(
        title=title,
        author=author,
        text=_strip_gutenberg_boilerplate(raw),
        source=f"file:{p.name}",
    )


def load_corpus(directory: str | Path) -> list[Book]:
    """Charge tous les .txt d'un dossier en tant que corpus."""
    d = Path(directory)
    return [load_from_file(p) for p in sorted(d.glob("*.txt"))]
