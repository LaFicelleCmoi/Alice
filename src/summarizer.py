"""Résumé automatique : trois stratégies extractives + résumé hiérarchique pour les longs textes."""

from __future__ import annotations

from dataclasses import dataclass

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words

from .preprocessing import chunk_by_words, ensure_nltk_resources

ALGORITHMS = {
    "textrank": TextRankSummarizer,
    "lexrank": LexRankSummarizer,
    "lsa": LsaSummarizer,
}


@dataclass
class SummaryResult:
    algorithm: str
    sentences: list[str]
    compression_ratio: float
    source_word_count: int
    summary_word_count: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def summarize(
    text: str,
    sentences: int = 8,
    algorithm: str = "textrank",
    language: str = "english",
) -> SummaryResult:
    """Résume `text` en `sentences` phrases avec l'algorithme choisi."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algorithme inconnu : {algorithm}. Choix : {list(ALGORITHMS)}")

    ensure_nltk_resources()
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(language)
    summarizer = ALGORITHMS[algorithm](stemmer)
    summarizer.stop_words = get_stop_words(language)

    summary_sents = [str(s) for s in summarizer(parser.document, sentences)]
    src_words = len(text.split())
    sum_words = sum(len(s.split()) for s in summary_sents)
    ratio = sum_words / src_words if src_words else 0.0

    return SummaryResult(
        algorithm=algorithm,
        sentences=summary_sents,
        compression_ratio=ratio,
        source_word_count=src_words,
        summary_word_count=sum_words,
    )


def hierarchical_summary(
    text: str,
    final_sentences: int = 10,
    chunk_words: int = 1500,
    chunk_sentences: int = 4,
    algorithm: str = "textrank",
    language: str = "english",
) -> SummaryResult:
    """Résumé en deux passes : on résume chaque bloc, puis on résume les résumés.

    Indispensable pour les livres entiers, qui font sortir TextRank de mémoire si on
    leur demande de tout traiter d'un coup.
    """
    chunks = chunk_by_words(text, words_per_chunk=chunk_words)
    if len(chunks) <= 1:
        return summarize(text, sentences=final_sentences, algorithm=algorithm, language=language)

    intermediate: list[str] = []
    for chunk in chunks:
        partial = summarize(chunk, sentences=chunk_sentences, algorithm=algorithm, language=language)
        intermediate.extend(partial.sentences)

    merged = " ".join(intermediate)
    return summarize(merged, sentences=final_sentences, algorithm=algorithm, language=language)


def compare_algorithms(
    text: str,
    sentences: int = 5,
    language: str = "english",
) -> dict[str, SummaryResult]:
    """Renvoie un résumé par algorithme — pratique pour comparer pédagogiquement."""
    return {
        name: summarize(text, sentences=sentences, algorithm=name, language=language)
        for name in ALGORITHMS
    }
