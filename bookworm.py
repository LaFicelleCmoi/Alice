#!/usr/bin/env python3
"""bookworm.py — moteur NLP qui produit des « book cards » depuis Project Gutenberg.

Conformité au sujet T-AIA-600 (Alice / Through the Looking-Glass).

Options exclusives :
    python bookworm.py --lexdiv     <ID>   → dict des mesures de diversité lexicale
    python bookworm.py --topics     <ID>   → dict {section_id: [10 mots du sujet]}
    python bookworm.py --entities   <ID>   → dict {"characters": [...], "locations": [...]}
    python bookworm.py --summarize  <ID>   → string (résumé en quelques phrases)
    python bookworm.py --similar    <ID>   → list[str] (5 titres similaires)
    python bookworm.py --card       <ID>   → dict regroupant tout ce qui précède

Caching : chaque résultat coûteux est mis en cache dans `outputs/cache/<ID>/<task>.json`.
Forcer le recalcul avec `--no-cache`.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

# Console Windows : forcer UTF-8 pour ne pas planter sur accents et flèches.
# Idempotent — ne remplace pas un wrapper déjà UTF-8 (sinon on jette des buffers
# orphelins, p. ex. les prints du test runner avant d'avoir importé bookworm).
def _ensure_utf8_stream(stream):
    if not hasattr(stream, "buffer"):
        return stream
    if getattr(stream, "encoding", "").lower().startswith("utf"):
        return stream
    return io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")


if sys.platform == "win32":
    sys.stdout = _ensure_utf8_stream(sys.stdout)
    sys.stderr = _ensure_utf8_stream(sys.stderr)

from src import analyzer, classifier, summarizer
from src.loader import Book, BookLoadError, load_from_gutenberg
from src.preprocessing import (
    chunk_by_words,
    ensure_nltk_resources,
    split_into_chapters,
    tokenize,
)


# ---------------------------------------------------------------------------
# Catalogue imposé par le sujet — 21 livres.
# ---------------------------------------------------------------------------

BOOK_COLLECTION: dict[int, tuple[str, str]] = {
    # Children / Young Adult
    11:    ("Alice's Adventures in Wonderland",        "Children / Young Adult"),
    12:    ("Through the Looking-Glass",               "Children / Young Adult"),
    16:    ("Peter Pan",                               "Children / Young Adult"),
    55:    ("The Wonderful Wizard of Oz",              "Children / Young Adult"),
    113:   ("The Secret Garden",                       "Children / Young Adult"),
    120:   ("Treasure Island",                         "Children / Young Adult"),
    236:   ("The Jungle Book",                         "Children / Young Adult"),
    # Crime, Mystery & Thriller
    108:   ("The Return of Sherlock Holmes",           "Crime, Mystery & Thriller"),
    834:   ("The Memoirs of Sherlock Holmes",          "Crime, Mystery & Thriller"),
    863:   ("The Mysterious Affair at Styles",         "Crime, Mystery & Thriller"),
    1661:  ("The Adventures of Sherlock Holmes",       "Crime, Mystery & Thriller"),
    61262: ("Poirot Investigates",                     "Crime, Mystery & Thriller"),
    69087: ("The Murder of Roger Ackroyd",             "Crime, Mystery & Thriller"),
    70114: ("The Big Four",                            "Crime, Mystery & Thriller"),
    # Science-Fiction & Fantasy
    35:    ("The Time Machine",                        "Science-Fiction & Fantasy"),
    36:    ("The War of the Worlds",                   "Science-Fiction & Fantasy"),
    84:    ("Frankenstein; Or, The Modern Prometheus", "Science-Fiction & Fantasy"),
    159:   ("The Island of Doctor Moreau",             "Science-Fiction & Fantasy"),
    164:   ("Twenty Thousand Leagues under the Sea",   "Science-Fiction & Fantasy"),
    345:   ("Dracula",                                 "Science-Fiction & Fantasy"),
    68283: ("The Call of Cthulhu",                     "Science-Fiction & Fantasy"),
}

CACHE_ROOT = Path("outputs/cache")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_file(book_id: int, task: str) -> Path:
    d = CACHE_ROOT / str(book_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{task}.json"


def _load_cached(book_id: int, task: str):
    p = _cache_file(book_id, task)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cached(book_id: int, task: str, data) -> None:
    p = _cache_file(book_id, task)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_book(book_id: int) -> Book:
    """Télécharge le livre Gutenberg (et le met en cache disque) ou lève BookLoadError."""
    return load_from_gutenberg(book_id)


def _info_for(book_id: int, book: Book | None = None) -> dict:
    """Renvoie {id, authors, bookshelves} — bookshelves connu pour les 21 du catalogue."""
    if book is None:
        try:
            book = _get_book(book_id)
        except BookLoadError:
            book = None
    title, shelves = BOOK_COLLECTION.get(book_id, (book.title if book else "Unknown", "Unknown"))
    return {
        "id": str(book_id),
        "authors": book.author if book else "Unknown",
        "bookshelves": shelves,
    }


# ---------------------------------------------------------------------------
# Tâches NLP — chacune renvoie EXACTEMENT la forme exigée par le sujet.
# ---------------------------------------------------------------------------

def task_lexdiv(book_id: int, no_cache: bool = False) -> dict:
    """Renvoie un dict des mesures de diversité lexicale.

    Clés exactes du sujet : tok, typ, hap, ttr, mwl, mwf.
    """
    if not no_cache:
        cached = _load_cached(book_id, "lexdiv")
        if cached is not None:
            return cached

    book = _get_book(book_id)
    tokens = tokenize(book.text, remove_stopwords=False, lemmatize=False)
    tok = len(tokens)
    if tok == 0:
        return {"tok": 0, "typ": 0, "hap": 0, "ttr": 0.0, "mwl": 0.0, "mwf": 0.0}

    freqs = Counter(tokens)
    typ = len(freqs)
    hap = sum(1 for c in freqs.values() if c == 1)

    result = {
        "tok": tok,
        "typ": typ,
        "hap": hap,
        "ttr": typ / tok,
        "mwl": sum(len(t) for t in tokens) / tok,
        "mwf": tok / typ,
    }
    _save_cached(book_id, "lexdiv", result)
    return result


def task_topics(book_id: int, no_cache: bool = False, n_top_words: int = 10) -> dict:
    """Renvoie {section_id: [top 10 mots du sujet principal]} — section_id entier ≥ 1.

    Implémentation :
    - on découpe le livre en chapitres (`split_into_chapters`) ;
    - si aucun chapitre n'est détecté, on tombe sur des blocs de ~2000 mots ;
    - pour chaque section, on lance une LDA(n_components=1) qui ressort le sujet
      saillant. Si LDA refuse (section trop courte), on retombe sur le top-10
      des lemmes les plus fréquents.
    """
    if not no_cache:
        cached = _load_cached(book_id, "topics")
        if cached is not None:
            # JSON ne sait stocker que des clés string : reconvertir en int.
            return {int(k): v for k, v in cached.items()}

    book = _get_book(book_id)
    sections = split_into_chapters(book.text)
    # Le splitter capte aussi les entrées de la table des matières (titres très
    # courts). On garde uniquement les vrais chapitres (≥ 100 mots de contenu).
    sections = [s for s in sections if len(s.split()) >= 100]
    if len(sections) < 2:
        sections = chunk_by_words(book.text, words_per_chunk=2000)

    out: dict[int, list[str]] = {}
    for idx, section in enumerate(sections, start=1):
        topics = classifier.discover_topics(section, n_topics=1, n_top_words=n_top_words)
        words = list(topics[0].top_words) if topics else []
        # LDA peut renvoyer < n_top_words sur les sections courtes (min_df).
        # On complète avec les lemmes les plus fréquents hors stopwords.
        if len(words) < n_top_words:
            toks = tokenize(section, remove_stopwords=True, lemmatize=True)
            seen = set(words)
            for w, _ in Counter(toks).most_common():
                if w not in seen:
                    words.append(w)
                    seen.add(w)
                if len(words) >= n_top_words:
                    break
        out[idx] = words[:n_top_words]

    _save_cached(book_id, "topics", {str(k): v for k, v in out.items()})
    return out


def task_entities(book_id: int, no_cache: bool = False, top_n: int = 25) -> dict:
    """Renvoie {"characters": list[str], "locations": list[str]}."""
    if not no_cache:
        cached = _load_cached(book_id, "entities")
        if cached is not None:
            return cached

    book = _get_book(book_id)
    chars = [name for name, _ in analyzer.extract_characters(book.text, top_n=top_n)]
    locs = [name for name, _ in analyzer.extract_places(book.text, top_n=top_n)]
    result = {"characters": chars, "locations": locs}
    _save_cached(book_id, "entities", result)
    return result


def task_summarize(book_id: int, no_cache: bool = False, sentences: int = 8) -> str:
    """Renvoie une **string** résumant le livre en quelques phrases."""
    if not no_cache:
        cached = _load_cached(book_id, "summary")
        if cached is not None and "summary" in cached:
            return cached["summary"]

    ensure_nltk_resources()
    book = _get_book(book_id)
    res = summarizer.hierarchical_summary(book.text, final_sentences=sentences)
    text = " ".join(res.sentences).strip()
    _save_cached(book_id, "summary", {"summary": text, "sentences": res.sentences,
                                       "compression_ratio": res.compression_ratio})
    return text


def task_similar(book_id: int, no_cache: bool = False, top_n: int = 5) -> list[str]:
    """Renvoie une liste de `top_n` titres triée par similarité décroissante.

    La similarité est calculée sur l'ensemble du catalogue imposé (21 livres),
    en TF-IDF + cosinus. Les livres non téléchargeables sont silencieusement
    ignorés (la liste peut donc faire moins de 5 si le catalogue est incomplet).
    """
    if not no_cache:
        cached = _load_cached(book_id, "similar")
        if cached is not None:
            return list(cached)

    target = _get_book(book_id)
    corpus: list[Book] = []
    for other_id in BOOK_COLLECTION:
        if other_id == book_id:
            continue
        try:
            corpus.append(_get_book(other_id))
        except BookLoadError as e:
            print(f"  Avertissement : Gutenberg #{other_id} ignoré ({e})", file=sys.stderr)
            continue

    if not corpus:
        return []

    suggestions = classifier.find_similar_books(target, corpus, top_n=top_n)
    titles = [s["title"] for s in suggestions]
    _save_cached(book_id, "similar", titles)
    return titles


def task_card(book_id: int, no_cache: bool = False) -> dict:
    """Compile **toute** l'information NLP d'un livre dans un seul dict.

    Structure exacte spécifiée par le sujet :
        {
          "info":     {"id": str, "authors": str, "bookshelves": str},
          "lexdiv":   {tok, typ, hap, ttr, mwl, mwf},
          "topics":   {1: [...], ..., 4: [...]},      # 4 sections de la fiche
          "entities": {"characters": [...], "locations": [...]},
          "summary":  str,
          "similar":  ["title1", ..., "title5"],
        }
    """
    if not no_cache:
        cached = _load_cached(book_id, "card")
        if cached is not None:
            # Reconvertir les clés de topics en int après lecture JSON.
            cached["topics"] = {int(k): v for k, v in cached.get("topics", {}).items()}
            return cached

    book = _get_book(book_id)

    all_topics = task_topics(book_id, no_cache=no_cache)
    # La fiche ne retient que 4 sujets — sections les plus longues prioritaires
    # pour avoir un signal stable, à défaut on prend les 4 premières.
    topics_4 = {k: all_topics[k] for k in sorted(all_topics)[:4]}

    card = {
        "info": _info_for(book_id, book=book),
        "lexdiv": task_lexdiv(book_id, no_cache=no_cache),
        "topics": topics_4,
        "entities": task_entities(book_id, no_cache=no_cache),
        "summary": task_summarize(book_id, no_cache=no_cache),
        "similar": task_similar(book_id, no_cache=no_cache),
    }

    _save_cached(book_id, "card", {**card, "topics": {str(k): v for k, v in topics_4.items()}})
    return card


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_catalog() -> str:
    """Liste lisible des 21 IDs du catalogue imposé, groupés par bookshelf."""
    by_shelf: dict[str, list[tuple[int, str]]] = {}
    for bid, (title, shelf) in BOOK_COLLECTION.items():
        by_shelf.setdefault(shelf, []).append((bid, title))

    lines = [f"IDs disponibles dans le catalogue ({len(BOOK_COLLECTION)} livres) :"]
    for shelf in ("Children / Young Adult",
                  "Crime, Mystery & Thriller",
                  "Science-Fiction & Fantasy"):
        lines.append(f"\n  {shelf}")
        for bid, title in sorted(by_shelf.get(shelf, []), key=lambda x: x[0]):
            lines.append(f"    {bid:<6} {title}")
    lines.append("\nExemple : python bookworm.py --card 11")
    return "\n".join(lines)


class _CatalogParser(argparse.ArgumentParser):
    """ArgumentParser qui rappelle les IDs disponibles en cas d'erreur d'usage."""

    def error(self, message: str):  # noqa: D401 — surcharge argparse
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}\n", file=sys.stderr)
        print(format_catalog(), file=sys.stderr)
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = _CatalogParser(
        prog="bookworm",
        description="Moteur NLP pour livres Gutenberg — résumé, sujets, entités, similarité, fiche.",
        epilog=format_catalog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lexdiv",    type=int, metavar="ID",
                       help="Mesures de diversité lexicale (tok/typ/hap/ttr/mwl/mwf).")
    group.add_argument("--topics",    type=int, metavar="ID",
                       help="Top-10 mots du sujet principal de chaque section.")
    group.add_argument("--entities",  type=int, metavar="ID",
                       help="Personnages et lieux extraits du livre.")
    group.add_argument("--summarize", type=int, metavar="ID",
                       help="Résumé du livre en quelques phrases.")
    group.add_argument("--similar",   type=int, metavar="ID",
                       help="Cinq titres similaires (catalogue imposé).")
    group.add_argument("--card",      type=int, metavar="ID",
                       help="Carte de livre complète (info + tous les NLP).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recalcule sans relire le cache disque.")
    parser.add_argument("--sentences", type=int, default=8,
                        help="Longueur du résumé (option --summarize).")
    return parser


def _print_result(result) -> None:
    """Affiche la sortie : JSON formaté pour dict/list, brut pour string."""
    if isinstance(result, str):
        print(result)
    else:
        # Clés int de topics → string pour JSON, mais on garde l'ordre numérique.
        if isinstance(result, dict) and all(isinstance(k, int) for k in result):
            result = {str(k): v for k, v in sorted(result.items())}
        elif isinstance(result, dict) and "topics" in result and isinstance(result["topics"], dict):
            result = {**result,
                      "topics": {str(k): v for k, v in sorted(result["topics"].items())}}
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.lexdiv is not None:
            result = task_lexdiv(args.lexdiv, no_cache=args.no_cache)
        elif args.topics is not None:
            result = task_topics(args.topics, no_cache=args.no_cache)
        elif args.entities is not None:
            result = task_entities(args.entities, no_cache=args.no_cache)
        elif args.summarize is not None:
            result = task_summarize(args.summarize, no_cache=args.no_cache,
                                     sentences=args.sentences)
        elif args.similar is not None:
            result = task_similar(args.similar, no_cache=args.no_cache)
        elif args.card is not None:
            result = task_card(args.card, no_cache=args.no_cache)
        else:  # pragma: no cover — argparse garantit qu'une option est passée
            parser.error("Choisis une option : --lexdiv / --topics / --entities / "
                         "--summarize / --similar / --card.")
    except BookLoadError as e:
        print(f"Erreur de chargement : {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130

    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
