"""Interface Streamlit pour T-ALICE.

Lancement : `streamlit run app.py`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bookworm import BOOK_COLLECTION
from src import analyzer, classifier, summarizer
from src.loader import load_from_file, load_from_gutenberg



def _build_ids_about_text() -> str:
    """Construit le markdown listant les 21 IDs, groupé par catégorie."""
    icons = {
        "Children / Young Adult":    "🧒",
        "Crime, Mystery & Thriller": "🔍",
        "Science-Fiction & Fantasy": "🚀",
    }
    groups: dict[str, list[tuple[int, str]]] = {}
    for bid, (title, shelf) in BOOK_COLLECTION.items():
        groups.setdefault(shelf, []).append((bid, title))
    for shelf in groups:
        groups[shelf].sort(key=lambda x: x[1])

    lines = ["### 📋 IDs Project Gutenberg — catalogue du sujet (21 livres)\n"]
    for shelf in ("Children / Young Adult",
                  "Crime, Mystery & Thriller",
                  "Science-Fiction & Fantasy"):
        lines.append(f"\n**{icons.get(shelf, '📖')} {shelf}**")
        lines.append("")
        lines.append("| ID | Titre |")
        lines.append("|---|---|")
        for bid, title in groups.get(shelf, []):
            lines.append(f"| `{bid}` | {title} |")
    lines.append("\n---")
    lines.append("Utilise ces IDs en CLI : `python bookworm.py --card <ID>`")
    return "\n".join(lines)


IDS_MENU_TEXT = _build_ids_about_text()


st.set_page_config(
    page_title="T-ALICE — NLP sur les livres",
    page_icon="📚",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": IDS_MENU_TEXT,
    },
)

st.title("📚 T-ALICE")
st.caption("T-AIA-600 — découverte du NLP : résumé, classification thématique, analyse de livres.")




@st.cache_data(show_spinner=False)
def cached_gutenberg(book_id: int):
    return load_from_gutenberg(book_id)


@st.cache_data(show_spinner=False)
def cached_file(path: str):
    return load_from_file(path)


# Regrouper la BOOK_COLLECTION par catégorie pour l'affichage.
def _group_books_by_category():
    grouped: dict[str, list[tuple[int, str]]] = {}
    for bid, (title, shelf) in BOOK_COLLECTION.items():
        grouped.setdefault(shelf, []).append((bid, title))
    for shelf in grouped:
        grouped[shelf].sort(key=lambda x: x[1])
    return grouped


CATEGORY_ICONS = {
    "Children / Young Adult":    "🧒",
    "Crime, Mystery & Thriller": "🔍",
    "Science-Fiction & Fantasy": "🚀",
}

with st.sidebar:
    st.header("📥 Source du livre")
    source = st.radio("Choisir la source", ["📚 Catalogue (21 livres)",
                                              "🔢 ID Gutenberg libre",
                                              "📁 Fichier local"])
    book = None

   
    if source == "📚 Catalogue (21 livres)":
        grouped = _group_books_by_category()
        labels: list[str] = []
        label_to_id: dict[str, int] = {}
        for shelf in ("Children / Young Adult",
                      "Crime, Mystery & Thriller",
                      "Science-Fiction & Fantasy"):
            icon = CATEGORY_ICONS.get(shelf, "📖")
            labels.append(f"── {icon} {shelf} ──")
            for bid, title in grouped.get(shelf, []):
                label = f"  [{bid}] {title}"
                labels.append(label)
                label_to_id[label] = bid

        choice = st.selectbox("Livre", labels,
                               index=labels.index([l for l in labels if "[11]" in l][0]))
        if choice in label_to_id:
            book_id = label_to_id[choice]
            st.caption(f"ID Gutenberg : **{book_id}**")
            if st.button("Charger", type="primary", use_container_width=True):
                with st.spinner(f"Téléchargement de Gutenberg #{book_id}…"):
                    book = cached_gutenberg(int(book_id))
                    st.session_state["book"] = book
        else:
            st.info("Choisis un livre (pas un séparateur de catégorie).")

        
        with st.expander("📋 Aide-mémoire des 21 IDs"):
            for shelf in ("Children / Young Adult",
                          "Crime, Mystery & Thriller",
                          "Science-Fiction & Fantasy"):
                icon = CATEGORY_ICONS.get(shelf, "📖")
                st.markdown(f"**{icon} {shelf}**")
                for bid, title in grouped.get(shelf, []):
                    st.markdown(f"- `{bid}` — {title}")

    
    elif source == "🔢 ID Gutenberg libre":
        book_id = st.number_input("ID Gutenberg", min_value=1, value=11, step=1)
        st.caption("Exemples connus hors catalogue : 1342 (Pride and Prejudice), "
                   "2701 (Moby Dick), 1080 (A Modest Proposal).")
        if st.button("Charger", type="primary", use_container_width=True):
            with st.spinner(f"Téléchargement de Gutenberg #{book_id}…"):
                book = cached_gutenberg(int(book_id))
                st.session_state["book"] = book

    # --- Fichier local ------------------------------------------------------
    else:
        upload = st.file_uploader("Fichier .txt", type=["txt"])
        if upload is not None:
            tmp = Path("data/books") / upload.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(upload.read())
            book = cached_file(str(tmp))
            st.session_state["book"] = book

    st.divider()
    lang = st.selectbox("Langue NLP", ["english", "french", "german", "spanish"], index=0)


book = st.session_state.get("book")
if book is None:
    st.info("👈 Choisis un livre dans la barre latérale pour commencer.")
    st.stop()



c1, c2, c3, c4 = st.columns(4)
c1.metric("Titre", book.title[:30] + ("…" if len(book.title) > 30 else ""))
c2.metric("Auteur", book.author[:25])
c3.metric("Mots", f"{book.word_count:,}")
c4.metric("Caractères", f"{book.char_count:,}")




tab_sum, tab_cls, tab_topics, tab_stats, tab_keywords, tab_text = st.tabs([
    "📝 Résumé", "🎭 Genres", "🔬 Sujets latents",
    "📊 Statistiques", "🔑 Mots-clés", "📄 Texte brut",
])




with tab_sum:
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        algo = st.selectbox("Algorithme", ["textrank", "lexrank", "lsa"])
    with col_b:
        n_sent = st.slider("Phrases", 3, 20, 8)
    with col_c:
        compare = st.checkbox("Comparer les trois algorithmes côte à côte")

    if st.button("Générer le résumé", type="primary"):
        if compare:
            results = summarizer.compare_algorithms(book.text, sentences=n_sent, language=lang)
            cols = st.columns(3)
            for col, (name, res) in zip(cols, results.items()):
                with col:
                    st.subheader(name.upper())
                    st.caption(f"Compression {res.compression_ratio:.1%}")
                    for s in res.sentences:
                        st.write(f"• {s}")
        else:
            with st.spinner("Calcul du résumé hiérarchique…"):
                result = summarizer.hierarchical_summary(
                    book.text, final_sentences=n_sent, algorithm=algo, language=lang
                )
            st.success(
                f"Compression {result.compression_ratio:.2%} — "
                f"{result.source_word_count:,} mots → {result.summary_word_count} mots"
            )
            for i, s in enumerate(result.sentences, 1):
                st.markdown(f"**{i}.** {s}")




with tab_cls:
    st.markdown("Score de chaque genre selon la fréquence relative de ses mots-clés (en ‰).")
    genres = classifier.classify_by_keywords(book.text, top_k=8, lang=lang)
    df = pd.DataFrame([{"Genre": g.genre, "Score": g.score} for g in genres])
    st.bar_chart(df.set_index("Genre"))
    st.dataframe(df, use_container_width=True, hide_index=True)




with tab_topics:
    n_topics = st.slider("Nombre de sujets (LDA)", 2, 10, 5)
    with st.spinner("Apprentissage LDA…"):
        topics = classifier.discover_topics(book.text, n_topics=n_topics, lang=lang)
    if not topics:
        st.warning("Pas assez de matière pour découvrir des sujets.")
    else:
        for t in topics:
            st.markdown(f"**Sujet #{t.topic_id}** — poids {t.weight:.1%}")
            st.write(" · ".join(t.top_words))
            st.divider()




with tab_stats:
    stats = analyzer.text_statistics(book.text, lang=lang)
    sent = analyzer.analyze_sentiment(book.text, lang=lang)

    s1, s2, s3 = st.columns(3)
    s1.metric("Vocabulaire unique", f"{stats.unique_words:,}")
    s2.metric("Richesse lexicale (TTR)", f"{stats.type_token_ratio:.3f}")
    s3.metric("Phrases", f"{stats.sentence_count:,}")

    s4, s5, s6 = st.columns(3)
    s4.metric("Phrase moyenne", f"{stats.avg_sentence_length:.1f} mots")
    s5.metric("Mot moyen", f"{stats.avg_word_length:.1f} car.")
    s6.metric("Sentiment", sent.label, delta=f"{sent.polarity:+.3f}")

    st.subheader("Courbe émotionnelle")
    arc = analyzer.sentiment_arc(book.text, segments=25, lang=lang)
    st.line_chart(pd.DataFrame({"polarité": arc}))

    st.subheader("Personnages / lieux probables")
    entities = analyzer.extract_named_entities(book.text, top_n=15)
    st.dataframe(pd.DataFrame(entities, columns=["Entité", "Occurrences"]),
                 use_container_width=True, hide_index=True)




with tab_keywords:
    n_kw = st.slider("Nombre de mots-clés", 10, 50, 25)
    keywords = analyzer.top_keywords(book.text, n=n_kw, lang=lang)
    df_kw = pd.DataFrame(keywords, columns=["mot", "fréquence"])
    st.bar_chart(df_kw.set_index("mot"))
    st.dataframe(df_kw, use_container_width=True, hide_index=True)




with tab_text:
    st.text_area("Extrait (3000 premiers caractères)", book.text[:3000], height=400)
