"""Tests rapides : valident que chaque étape de la pipeline fonctionne sur un mini-texte."""

from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analyzer, classifier, summarizer
from src.loader import load_from_file
from src.preprocessing import sentences, tokenize, split_into_chapters


SAMPLE = Path(__file__).resolve().parent.parent / "data" / "books" / "sample_mystery.txt"


def test_load_book():
    book = load_from_file(SAMPLE)
    assert book.word_count > 50
    assert "Hartwell" in book.text


def test_tokenize_strips_stopwords():
    toks = tokenize("The cat sat on the mat.", remove_stopwords=True, lemmatize=False)
    assert "the" not in toks
    assert "cat" in toks


def test_sentences_split():
    sents = sentences("Hello there. How are you? I am fine.")
    assert len(sents) == 3


def test_chapter_detection_falls_back_to_full_text():
    chunks = split_into_chapters("No headings here, just one chunk of text.")
    assert chunks == ["No headings here, just one chunk of text."]


def test_summarizer_textrank():
    book = load_from_file(SAMPLE)
    result = summarizer.summarize(book.text, sentences=2, algorithm="textrank")
    assert len(result.sentences) == 2
    assert 0 < result.compression_ratio < 1


def test_classify_keywords_picks_mystery():
    book = load_from_file(SAMPLE)
    scores = classifier.classify_by_keywords(book.text, top_k=3)
    assert scores[0].genre == "mystère"


def test_analyzer_returns_full_report():
    book = load_from_file(SAMPLE)
    report = analyzer.full_report(book.text)
    assert report["stats"]["word_count"] > 0
    assert "polarity" in report["sentiment"]
    assert isinstance(report["top_keywords"], list)
    assert "characters" in report and "places" in report
    assert "lexical_diversity" in report


def test_lexical_diversity_has_at_least_five_measures():
    book = load_from_file(SAMPLE)
    div = analyzer.lexical_diversity(book.text)
    # Trophée « lexdiv » : au moins 5 mesures
    assert isinstance(div, dict)
    assert len(div) >= 5
    for k, v in div.items():
        assert isinstance(v, float), f"{k} n'est pas un float"


def test_extract_characters_and_places():
    book = load_from_file(SAMPLE)
    chars = analyzer.extract_characters(book.text)
    places = analyzer.extract_places(book.text)
    # Sur le mystery sample, « Lord Ashbury » est un personnage et « Hollow House » un lieu
    char_names = {n for n, _ in chars}
    place_names = {n for n, _ in places}
    assert any("Ashbury" in n for n in char_names)
    assert any("Hollow" in n for n in place_names)


def test_book_sheet_returns_full_dict():
    book = load_from_file(SAMPLE)
    sheet = analyzer.book_sheet(book, summary_sentences=2)
    # Trophée « fiche de livre » : un seul dict regroupant les diverses infos
    for key in ("metadata", "stats", "lexical_diversity", "sentiment",
                "top_keywords", "characters", "places", "genres", "topics", "summary"):
        assert key in sheet, f"clé manquante : {key}"


def test_topics_per_section_returns_list():
    book = load_from_file(SAMPLE)
    sections = classifier.topics_per_section(book.text, n_topics_per_section=2)
    assert isinstance(sections, list) and len(sections) >= 1
    assert all("topics" in s and "section_id" in s for s in sections)


def test_find_similar_books_ranks_corpus():
    from src.loader import load_corpus
    target = load_from_file(SAMPLE)
    corpus = load_corpus(SAMPLE.parent)
    results = classifier.find_similar_books(target, corpus, top_n=3)
    # On doit retrouver des suggestions et la cible doit en être exclue
    assert isinstance(results, list)
    assert all(r["source"] != target.source for r in results)
    if results:
        # tri décroissant par similarité
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)


def test_loader_raises_on_missing_file():
    from src.loader import BookLoadError, load_from_file as lf
    try:
        lf("data/books/does_not_exist_xyz.txt")
    except BookLoadError as e:
        assert "introuvable" in str(e).lower()
    else:
        raise AssertionError("BookLoadError attendu pour un fichier manquant")


# ---------------------------------------------------------------------------
# Tests bookworm.py — utilisent monkey-patching pour éviter les appels réseau.
# ---------------------------------------------------------------------------

def _patch_bookworm_with_sample():
    """Remplace _get_book dans bookworm pour renvoyer le sample local."""
    import bookworm
    sample = load_from_file(SAMPLE)
    bookworm._get_book = lambda book_id: sample
    return bookworm


def test_bookworm_lexdiv_has_exact_spec_keys():
    bw = _patch_bookworm_with_sample()
    # Vider tout cache
    res = bw.task_lexdiv(99999, no_cache=True)
    assert set(res) == {"tok", "typ", "hap", "ttr", "mwl", "mwf"}
    assert isinstance(res["tok"], int) and isinstance(res["typ"], int)
    assert isinstance(res["hap"], int)
    assert isinstance(res["ttr"], float) and isinstance(res["mwl"], float)
    assert isinstance(res["mwf"], float)
    # mwf == tok / typ
    assert abs(res["mwf"] - res["tok"] / res["typ"]) < 1e-9


def test_bookworm_topics_format():
    bw = _patch_bookworm_with_sample()
    res = bw.task_topics(99998, no_cache=True)
    # Dict {int: list[str]}
    assert isinstance(res, dict)
    assert all(isinstance(k, int) for k in res.keys())
    assert all(isinstance(v, list) for v in res.values())
    assert all(all(isinstance(w, str) for w in v) for v in res.values())


def test_bookworm_entities_format():
    bw = _patch_bookworm_with_sample()
    res = bw.task_entities(99997, no_cache=True)
    assert set(res.keys()) == {"characters", "locations"}
    assert isinstance(res["characters"], list)
    assert isinstance(res["locations"], list)


def test_bookworm_summarize_returns_string():
    bw = _patch_bookworm_with_sample()
    res = bw.task_summarize(99996, no_cache=True, sentences=3)
    assert isinstance(res, str)
    assert len(res) > 0


def test_bookworm_card_structure():
    bw = _patch_bookworm_with_sample()
    # On évite les téléchargements externes en court-circuitant similar
    bw.task_similar = lambda book_id, no_cache=False, top_n=5: [
        "Through the Looking-Glass", "Treasure Island",
        "Peter Pan", "Dracula", "The Time Machine",
    ]
    res = bw.task_card(99995, no_cache=True)
    assert set(res.keys()) == {"info", "lexdiv", "topics", "entities", "summary", "similar"}
    assert set(res["info"].keys()) == {"id", "authors", "bookshelves"}
    assert set(res["lexdiv"].keys()) == {"tok", "typ", "hap", "ttr", "mwl", "mwf"}
    assert set(res["entities"].keys()) == {"characters", "locations"}
    assert isinstance(res["summary"], str)
    assert isinstance(res["similar"], list) and len(res["similar"]) == 5
    # La fiche limite topics à 4 sections
    assert len(res["topics"]) <= 4


def test_bookworm_collection_has_21_books():
    import bookworm
    assert len(bookworm.BOOK_COLLECTION) == 21
    # Quelques IDs spécifiques imposés par le sujet
    for required in (11, 12, 16, 84, 345, 1661):
        assert required in bookworm.BOOK_COLLECTION


def test_bookworm_cli_rejects_unknown_flag():
    import bookworm
    try:
        bookworm.main(["--bogus", "11"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("SystemExit attendu pour un flag inconnu")


def test_bookworm_cli_rejects_collision():
    import bookworm
    try:
        bookworm.main(["--lexdiv", "11", "--topics", "11"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("SystemExit attendu pour deux flags exclusifs")


if __name__ == "__main__":
    # Permet de lancer sans pytest : `python tests/test_pipeline.py`
    failed = 0
    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {name}: {e}")
        except Exception as e:  # noqa: BLE001 — on rapporte n'importe quelle erreur
            failed += 1
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests)} tests — "
          f"{'TOUS PASSENT' if failed == 0 else f'{failed} ÉCHEC(S)'}")
    sys.exit(failed)
