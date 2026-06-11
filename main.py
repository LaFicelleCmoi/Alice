#!/usr/bin/env python3
"""T-ALICE — point d'entrée CLI.

Usage :
    python main.py summarize --gutenberg 11 --sentences 10
    python main.py summarize --file data/books/alice.txt --algo lsa
    python main.py classify --file data/books/alice.txt
    python main.py analyze   --file data/books/alice.txt --output outputs/
    python main.py corpus    --dir  data/books --clusters 3
    python main.py demo
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Console Windows : force UTF-8 pour ne pas planter sur les caractères accentués / fléchés.
# Idempotent — voir bookworm.py pour les détails.
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer") and not getattr(sys.stdout, "encoding", "").lower().startswith("utf"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer") and not getattr(sys.stderr, "encoding", "").lower().startswith("utf"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src import analyzer, classifier, summarizer, visualization
from src.loader import Book, BookLoadError, load_corpus, load_from_file, load_from_gutenberg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _resolve_book(args: argparse.Namespace) -> Book:
    try:
        if args.gutenberg:
            print(f"Téléchargement du livre Gutenberg #{args.gutenberg}…")
            return load_from_gutenberg(args.gutenberg)
        if args.file:
            return load_from_file(args.file)
    except BookLoadError as e:
        print(f"Erreur de chargement : {e}", file=sys.stderr)
        sys.exit(2)
    print("Erreur : préciser --file PATH ou --gutenberg ID", file=sys.stderr)
    sys.exit(2)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → écrit : {path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_summarize(args: argparse.Namespace) -> None:
    book = _resolve_book(args)
    _print_section(f"Résumé de « {book.title} » — {book.author}")
    print(f"Source : {book.source}    Mots : {book.word_count:,}    Algorithme : {args.algo}")

    if args.compare:
        print("\nComparaison des trois algorithmes (5 phrases chacun) :")
        for name, result in summarizer.compare_algorithms(book.text, sentences=5, language=args.lang).items():
            print(f"\n--- {name.upper()} (compression {result.compression_ratio:.1%}) ---")
            for s in result.sentences:
                print(f"  • {s}")
        return

    if args.hierarchical or book.word_count > 5000:
        print("\nRésumé hiérarchique (deux passes — recommandé pour un livre entier).")
        result = summarizer.hierarchical_summary(
            book.text,
            final_sentences=args.sentences,
            algorithm=args.algo,
            language=args.lang,
        )
    else:
        result = summarizer.summarize(
            book.text,
            sentences=args.sentences,
            algorithm=args.algo,
            language=args.lang,
        )

    print(f"\nCompression : {result.compression_ratio:.2%}  "
          f"({result.source_word_count:,} mots → {result.summary_word_count} mots)\n")
    for i, sent in enumerate(result.sentences, 1):
        print(f"  {i}. {sent}")

    if args.output:
        out = Path(args.output) / f"{Path(book.source.replace(':', '_')).name}_summary.json"
        _save_json({
            "book": {"title": book.title, "author": book.author, "source": book.source},
            "summary": asdict(result),
        }, out)


def cmd_classify(args: argparse.Namespace) -> None:
    book = _resolve_book(args)
    _print_section(f"Classification de « {book.title} »")

    print("\nGenres dominants (lexique) :")
    for gs in classifier.classify_by_keywords(book.text, top_k=5, lang=args.lang):
        bar = "█" * int(min(gs.score, 30))
        print(f"  {gs.genre:18s} {bar:<30s} {gs.score:6.2f}")

    print("\nSujets latents (LDA) :")
    for topic in classifier.discover_topics(book.text, n_topics=args.topics, lang=args.lang):
        words = ", ".join(topic.top_words)
        print(f"  Sujet #{topic.topic_id} (poids {topic.weight:.2%}) : {words}")

    if args.per_section:
        print("\nSujets par section / chapitre :")
        sections = classifier.topics_per_section(book.text, lang=args.lang)
        for sec in sections:
            print(f"\n  Section #{sec['section_id']} "
                  f"({sec['word_count']:,} mots) — {sec['preview']!r}")
            for t in sec["topics"]:
                print(f"    • ({t['weight']:.2%}) {', '.join(t['top_words'])}")

    if args.output:
        out_dir = Path(args.output)
        slug = Path(book.source.replace(":", "_")).name
        topics = [
            {"top_words": t.top_words, "weight": t.weight}
            for t in classifier.discover_topics(book.text, n_topics=args.topics, lang=args.lang)
        ]
        payload = {
            "book": {"title": book.title, "author": book.author, "source": book.source},
            "genres": [{"genre": g.genre, "score": g.score}
                       for g in classifier.classify_by_keywords(book.text, top_k=5, lang=args.lang)],
            "topics": topics,
        }
        if args.per_section:
            payload["topics_per_section"] = classifier.topics_per_section(book.text, lang=args.lang)
        _save_json(payload, out_dir / f"{slug}_classify.json")


def cmd_analyze(args: argparse.Namespace) -> None:
    book = _resolve_book(args)
    _print_section(f"Analyse de « {book.title} »")

    report = analyzer.full_report(book.text, lang=args.lang)
    stats = report["stats"]
    sent = report["sentiment"]

    print(f"\nStatistiques :")
    print(f"  Mots               : {stats['word_count']:,}")
    print(f"  Phrases            : {stats['sentence_count']:,}")
    print(f"  Vocabulaire unique : {stats['unique_words']:,}")
    print(f"  Richesse lexicale  : {stats['type_token_ratio']:.3f}  (TTR)")
    print(f"  Phrase moyenne     : {stats['avg_sentence_length']:.1f} mots")
    print(f"  Mot moyen          : {stats['avg_word_length']:.1f} caractères")

    print(f"\nDiversité lexicale :")
    for k, v in report["lexical_diversity"].items():
        print(f"  {k:15s} {v:.4f}")

    print(f"\nSentiment global : {sent['label']}  "
          f"(polarité {sent['polarity']:+.3f} ; "
          f"+{sent['positive_hits']} / -{sent['negative_hits']})")

    print(f"\nMots-clés (top 15) :")
    for word, count in report["top_keywords"][:15]:
        print(f"  {word:20s} {count:>6}")

    print(f"\nPersonnages probables (top 10) :")
    for name, count in report["characters"][:10]:
        print(f"  {name:30s} {count:>6}")

    print(f"\nLieux probables (top 10) :")
    for name, count in report["places"][:10]:
        print(f"  {name:30s} {count:>6}")

    if args.output:
        out_dir = Path(args.output)
        slug = Path(book.source.replace(":", "_")).name
        _save_json({
            "book": {"title": book.title, "author": book.author, "source": book.source},
            "report": report,
        }, out_dir / f"{slug}_analysis.json")
        visualization.save_wordcloud(report["top_keywords"], out_dir / f"{slug}_wordcloud.png")
        print(f"  → écrit : {out_dir}/{slug}_wordcloud.png")
        visualization.save_sentiment_arc(report["sentiment_arc"], out_dir / f"{slug}_sentiment.png",
                                          title=f"Courbe émotionnelle — {book.title}")
        print(f"  → écrit : {out_dir}/{slug}_sentiment.png")


def cmd_corpus(args: argparse.Namespace) -> None:
    books = load_corpus(args.dir)
    if not books:
        print(f"Aucun livre trouvé dans {args.dir}", file=sys.stderr)
        sys.exit(2)
    _print_section(f"Corpus — {len(books)} livres dans {args.dir}")

    result = classifier.cluster_corpus(books, n_clusters=args.clusters, lang=args.lang)
    print(f"\nVocabulaire total : {result['vocabulary_size']} termes uniques")
    print(f"Clusters : {args.clusters}\n")

    for cluster_id in range(args.clusters):
        members = [b["title"] for b in result["books"] if b["cluster"] == cluster_id]
        terms = ", ".join(result["top_terms"][cluster_id])
        print(f"  Cluster {cluster_id} — termes saillants : {terms}")
        for title in members:
            print(f"    • {title}")
        print()

    if args.output:
        out_dir = Path(args.output)
        _save_json(result, out_dir / "corpus_clustering.json")
        labels = [b.title[:30] for b in books]
        visualization.save_similarity_heatmap(
            result["similarity_matrix"], labels, out_dir / "corpus_similarity.png"
        )
        print(f"  → écrit : {out_dir}/corpus_similarity.png")


def cmd_similar(args: argparse.Namespace) -> None:
    """Suggère les livres les plus proches du livre cible dans un dossier corpus."""
    target = _resolve_book(args)
    corpus = load_corpus(args.dir)
    if not corpus:
        print(f"Aucun livre trouvé dans {args.dir}", file=sys.stderr)
        sys.exit(2)
    _print_section(f"Livres similaires à « {target.title} »")
    print(f"Corpus comparé : {len(corpus)} livres dans {args.dir}")

    suggestions = classifier.find_similar_books(target, corpus, top_n=args.top, lang=args.lang)
    if not suggestions:
        print("Aucune suggestion (corpus trop petit).")
        return
    print()
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s['title']:40s}  (similarité {s['similarity']:.3f})  — {s['author']}")

    if args.output:
        out_dir = Path(args.output)
        slug = Path(target.source.replace(":", "_")).name
        _save_json({
            "target": {"title": target.title, "author": target.author, "source": target.source},
            "similar": suggestions,
        }, out_dir / f"{slug}_similar.json")


def cmd_book(args: argparse.Namespace) -> None:
    """Fiche de livre : un dictionnaire récapitulatif (méta + stats + diversité + sentiment + …)."""
    book = _resolve_book(args)
    _print_section(f"Fiche de livre — « {book.title} »")
    sheet = analyzer.book_sheet(book, lang=args.lang, summary_sentences=args.sentences)

    meta = sheet["metadata"]
    print(f"\n  Titre          : {meta['title']}")
    print(f"  Auteur         : {meta['author']}")
    print(f"  Source         : {meta['source']}")
    print(f"  Mots / Car.    : {meta['word_count']:,} / {meta['char_count']:,}")

    print(f"\n  Sentiment      : {sheet['sentiment']['label']} "
          f"({sheet['sentiment']['polarity']:+.3f})")

    print(f"\n  Diversité lexicale :")
    for k, v in sheet["lexical_diversity"].items():
        print(f"    {k:15s} {v:.4f}")

    print(f"\n  Genres dominants :")
    for g in sheet["genres"]:
        print(f"    • {g['genre']:18s} {g['score']:.2f}")

    print(f"\n  Sujets (LDA) :")
    for t in sheet["topics"]:
        print(f"    • ({t['weight']:.1%}) {', '.join(t['top_words'])}")

    print(f"\n  Personnages probables : {', '.join(n for n, _ in sheet['characters'][:8])}")
    print(f"  Lieux probables       : {', '.join(n for n, _ in sheet['places'][:8])}")

    print(f"\n  Résumé ({len(sheet['summary'])} phrases) :")
    for i, s in enumerate(sheet["summary"], 1):
        print(f"    {i}. {s}")

    if args.output:
        out_dir = Path(args.output)
        slug = Path(book.source.replace(":", "_")).name
        _save_json(sheet, out_dir / f"{slug}_book_sheet.json")


def cmd_showcase(args: argparse.Namespace) -> None:
    """Lance **tout** d'un coup : fiche, résumés comparés, classification (avec sujets par
    section), analyse complète + exports, clustering du corpus, livres similaires.

    Un seul livre cible + un dossier corpus. Tous les exports (JSON + PNG) atterrissent
    dans `--output`.
    """
    book = _resolve_book(args)
    out_dir = Path(args.output) if args.output else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = Path(book.source.replace(":", "_")).name

    _print_section(f"SHOWCASE T-ALICE — « {book.title} » par {book.author}")
    print(f"Sortie : {out_dir.resolve()}    Corpus : {args.dir}    Langue : {args.lang}")

    # 1) Fiche de livre
    _print_section("[1/6] Fiche de livre")
    sheet = analyzer.book_sheet(book, lang=args.lang, summary_sentences=args.sentences)
    _save_json(sheet, out_dir / f"{slug}_book_sheet.json")
    print(f"  Titre / Auteur     : {sheet['metadata']['title']} — {sheet['metadata']['author']}")
    print(f"  Sentiment global   : {sheet['sentiment']['label']} ({sheet['sentiment']['polarity']:+.3f})")
    print(f"  Diversité (TTR)    : {sheet['lexical_diversity']['TTR']:.3f}   "
          f"Yule K : {sheet['lexical_diversity']['Yule_K']:.1f}")
    print(f"  Genres dominants   : " + ", ".join(g['genre'] for g in sheet['genres'][:3]))
    print(f"  Personnages probables : {', '.join(n for n, _ in sheet['characters'][:5])}")
    print(f"  Lieux probables       : {', '.join(n for n, _ in sheet['places'][:5])}")

    # 2) Résumés comparés (TextRank / LexRank / LSA)
    _print_section("[2/6] Résumés extractifs comparés")
    compare = summarizer.compare_algorithms(book.text, sentences=args.sentences, language=args.lang)
    payload = {}
    for name, res in compare.items():
        print(f"\n--- {name.upper()} (compression {res.compression_ratio:.1%}) ---")
        for s in res.sentences:
            print(f"  • {s}")
        payload[name] = asdict(res)
    _save_json(payload, out_dir / f"{slug}_summaries.json")

    # 3) Classification + sujets par section
    _print_section("[3/6] Classification thématique + sujets par section")
    for gs in classifier.classify_by_keywords(book.text, top_k=5, lang=args.lang):
        bar = "█" * int(min(gs.score, 30))
        print(f"  {gs.genre:18s} {bar:<30s} {gs.score:6.2f}")
    sections = classifier.topics_per_section(book.text, n_topics_per_section=3, lang=args.lang)
    print(f"\n  Sections détectées : {len(sections)}")
    for sec in sections[:3]:
        print(f"  Section #{sec['section_id']} ({sec['word_count']:,} mots)")
        for t in sec["topics"]:
            print(f"    • ({t['weight']:.1%}) {', '.join(t['top_words'])}")
    if len(sections) > 3:
        print(f"  … {len(sections) - 3} sections supplémentaires écrites dans le JSON")
    _save_json({"genres_global": [{"genre": g.genre, "score": g.score}
                                   for g in classifier.classify_by_keywords(book.text, top_k=5, lang=args.lang)],
                "topics_per_section": sections},
               out_dir / f"{slug}_classify.json")

    # 4) Analyse complète + visualisations
    _print_section("[4/6] Analyse complète (stats + sentiment + entités) + visuels")
    report = analyzer.full_report(book.text, lang=args.lang)
    _save_json({"book": {"title": book.title, "author": book.author, "source": book.source},
                "report": report},
               out_dir / f"{slug}_analysis.json")
    visualization.save_wordcloud(report["top_keywords"], out_dir / f"{slug}_wordcloud.png")
    print(f"  → écrit : {out_dir}/{slug}_wordcloud.png")
    visualization.save_sentiment_arc(report["sentiment_arc"], out_dir / f"{slug}_sentiment.png",
                                      title=f"Courbe émotionnelle — {book.title}")
    print(f"  → écrit : {out_dir}/{slug}_sentiment.png")

    # 5) Corpus : clustering + heatmap
    _print_section(f"[5/6] Clustering du corpus ({args.dir})")
    corpus = load_corpus(args.dir)
    if len(corpus) < 2:
        print("  Corpus trop petit pour clusteriser (ignoré).")
    else:
        clusters = min(args.clusters, len(corpus))
        result = classifier.cluster_corpus(corpus, n_clusters=clusters, lang=args.lang)
        for cluster_id in range(clusters):
            members = [b["title"] for b in result["books"] if b["cluster"] == cluster_id]
            terms = ", ".join(result["top_terms"][cluster_id][:6])
            print(f"  Cluster {cluster_id} : {terms}")
            for title in members:
                print(f"    • {title}")
        _save_json(result, out_dir / "corpus_clustering.json")
        labels = [b.title[:30] for b in corpus]
        visualization.save_similarity_heatmap(
            result["similarity_matrix"], labels, out_dir / "corpus_similarity.png"
        )
        print(f"  → écrit : {out_dir}/corpus_similarity.png")

    # 6) Livres similaires à la cible
    _print_section(f"[6/6] Livres similaires à « {book.title} »")
    suggestions = classifier.find_similar_books(book, corpus, top_n=args.top, lang=args.lang)
    if not suggestions:
        print("  Pas assez de livres dans le corpus pour des suggestions.")
    else:
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s['title']:40s}  (similarité {s['similarity']:.3f})  — {s['author']}")
    _save_json({"target": {"title": book.title, "source": book.source},
                "similar": suggestions},
               out_dir / f"{slug}_similar.json")

    _print_section("SHOWCASE TERMINÉ")
    print(f"\nTous les exports sont dans : {out_dir.resolve()}")
    print("Pour visualiser dans le navigateur, lance maintenant :")
    print("    streamlit run app.py")


def cmd_demo(args: argparse.Namespace) -> None:
    """Démonstration de bout en bout sur Alice in Wonderland (Gutenberg #11)."""
    _print_section("DÉMO T-ALICE — Alice in Wonderland (Project Gutenberg #11)")

    book = load_from_gutenberg(11)
    print(f"\nLivre chargé : « {book.title} » par {book.author}")
    print(f"Taille : {book.word_count:,} mots, {book.char_count:,} caractères")

    print("\n[1/4] Résumé en 8 phrases (TextRank, hiérarchique)…")
    summary = summarizer.hierarchical_summary(book.text, final_sentences=8)
    for i, s in enumerate(summary.sentences, 1):
        print(f"  {i}. {s}")
    print(f"\n  Compression : {summary.compression_ratio:.2%}")

    print("\n[2/4] Classification thématique…")
    for gs in classifier.classify_by_keywords(book.text, top_k=3):
        print(f"  {gs.genre:18s} score {gs.score:6.2f}")

    print("\n[3/4] Sujets latents (LDA, 4 sujets)…")
    for topic in classifier.discover_topics(book.text, n_topics=4):
        print(f"  Sujet #{topic.topic_id} ({topic.weight:.1%}) : {', '.join(topic.top_words)}")

    print("\n[4/4] Analyse style + sentiment…")
    stats = analyzer.text_statistics(book.text)
    sent = analyzer.analyze_sentiment(book.text)
    print(f"  Vocabulaire unique : {stats.unique_words:,}  TTR {stats.type_token_ratio:.3f}")
    print(f"  Sentiment : {sent.label}  (polarité {sent.polarity:+.3f})")
    print(f"  Personnages probables : ",
          ", ".join(name for name, _ in analyzer.extract_named_entities(book.text, top_n=8)))

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    visualization.save_wordcloud(analyzer.top_keywords(book.text, n=80),
                                  out_dir / "alice_wordcloud.png")
    visualization.save_sentiment_arc(analyzer.sentiment_arc(book.text),
                                      out_dir / "alice_sentiment.png",
                                      title="Courbe émotionnelle — Alice in Wonderland")
    print(f"\nNuage de mots et courbe émotionnelle écrits dans {out_dir}/")
    print("\nDémo terminée. Lance `streamlit run app.py` pour l'interface graphique.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="t-alice",
        description="T-AIA-600 (Alice) — découverte du NLP : résumé, classification, analyse de livres.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p: argparse.ArgumentParser) -> None:
        src = p.add_mutually_exclusive_group()
        src.add_argument("--file", help="Chemin vers un fichier .txt local")
        src.add_argument("--gutenberg", type=int, help="ID Project Gutenberg (ex: 11 = Alice)")
        p.add_argument("--lang", default="english", help="Langue (english, french, …)")
        p.add_argument("--output", help="Dossier où écrire les JSON / images")

    p_sum = sub.add_parser("summarize", help="Résumer un livre")
    add_source(p_sum)
    p_sum.add_argument("--sentences", type=int, default=8)
    p_sum.add_argument("--algo", choices=["textrank", "lexrank", "lsa"], default="textrank")
    p_sum.add_argument("--hierarchical", action="store_true", help="Force le résumé en 2 passes")
    p_sum.add_argument("--compare", action="store_true", help="Compare les trois algorithmes")
    p_sum.set_defaults(func=cmd_summarize)

    p_cls = sub.add_parser("classify", help="Classer thématiquement un livre")
    add_source(p_cls)
    p_cls.add_argument("--topics", type=int, default=5, help="Nombre de sujets LDA")
    p_cls.add_argument("--per-section", action="store_true",
                       help="Extrait aussi les sujets principaux de chaque section du livre")
    p_cls.set_defaults(func=cmd_classify)

    p_ana = sub.add_parser("analyze", help="Analyse statistique + sentiment + entités")
    add_source(p_ana)
    p_ana.set_defaults(func=cmd_analyze)

    p_cor = sub.add_parser("corpus", help="Clusteriser un dossier de livres")
    p_cor.add_argument("--dir", required=True)
    p_cor.add_argument("--clusters", type=int, default=3)
    p_cor.add_argument("--lang", default="english")
    p_cor.add_argument("--output")
    p_cor.set_defaults(func=cmd_corpus)

    p_sim = sub.add_parser("similar", help="Suggérer des livres similaires à un livre cible")
    add_source(p_sim)
    p_sim.add_argument("--dir", required=True, help="Dossier corpus à comparer")
    p_sim.add_argument("--top", type=int, default=5, help="Nombre de suggestions à renvoyer")
    p_sim.set_defaults(func=cmd_similar)

    p_book = sub.add_parser("book", help="Fiche de livre (dictionnaire récapitulatif)")
    add_source(p_book)
    p_book.add_argument("--sentences", type=int, default=5, help="Longueur du résumé inclus")
    p_book.set_defaults(func=cmd_book)

    p_demo = sub.add_parser("demo", help="Démo complète sur Alice (Gutenberg #11)")
    p_demo.set_defaults(func=cmd_demo)

    p_show = sub.add_parser(
        "showcase",
        help="Lance TOUT d'un coup : fiche + résumés + classify (+sections) + analyze + corpus + similar",
    )
    add_source(p_show)
    p_show.add_argument("--dir", default="data/books", help="Dossier corpus à utiliser")
    p_show.add_argument("--clusters", type=int, default=3, help="Nombre de clusters pour le corpus")
    p_show.add_argument("--top", type=int, default=5, help="Nombre de livres similaires à suggérer")
    p_show.add_argument("--sentences", type=int, default=5, help="Phrases dans le résumé")
    p_show.set_defaults(func=cmd_showcase)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
