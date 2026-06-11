# T-ALICE — T-AIA-600

> Découverte de l'IA à travers le NLP : **résumés de livres**, **classification thématique**,
> **analyse stylistique** et **clustering de corpus**.

Projet pédagogique : on prend des livres entiers (Project Gutenberg ou fichiers locaux),
on les passe dans une pipeline NLP classique (TF-IDF, TextRank, LDA, K-Means, lexique de
sentiment) et on en sort des résumés, des thèmes, des courbes émotionnelles, des nuages
de mots, et une carte de similarité entre livres.

---

## Sommaire

- [Aperçu en 30 secondes](#aperçu-en-30-secondes)
- [Installation](#installation)
- [Utilisation — CLI](#utilisation--cli)
- [Utilisation — interface graphique](#utilisation--interface-graphique-streamlit)
- [Architecture](#architecture)
- [Concepts NLP utilisés](#concepts-nlp-utilisés)
- [Diagrammes des pipelines](#diagrammes-des-pipelines)
- [Tests](#tests)
- [Choix techniques](#choix-techniques)

---

## Aperçu en 30 secondes

```bash
pip install -r requirements.txt

# === bookworm.py — CLI officielle du sujet (Alice) ===
python3 bookworm.py --lexdiv    11   # mesures de diversité lexicale
python3 bookworm.py --topics    11   # top-10 mots par section
python3 bookworm.py --entities  11   # personnages + lieux
python3 bookworm.py --summarize 11   # résumé en quelques phrases
python3 bookworm.py --similar   11   # 5 titres similaires (catalogue imposé)
python3 bookworm.py --card      11   # fiche complète

# === main.py — CLI étendue / démo / interface ===
python3 main.py demo                                   # démo de bout en bout
python3 main.py showcase --gutenberg 11 --output outputs/alice  # TOUT en 1 commande
streamlit run app.py                                   # interface graphique
```

---

## Installation

Pré-requis : Python 3.10+ (testé avec 3.13).

```bash
git clone <ce-repo>
cd T-ALICE
pip install -r requirements.txt
```

Les ressources NLTK (`punkt`, `stopwords`, `wordnet`) se téléchargent automatiquement
au premier appel — pas besoin d'un script d'installation séparé.

---

## Utilisation — CLI

### Outils BOOTSTRAP — `calculator.py` & `tools.py`

Deux utilitaires CLI autonomes, en `argparse`, qui répondent au BOOTSTRAP du sujet.

**`calculator.py`** — arithmétique de base sur deux nombres :

```bash
./calculator.py --add 42 -4.0          # 42.0 + -4.0 = 38.0
./calculator.py --sub 42 -4.0          # 42.0 - -4.0 = 46.0
./calculator.py --mul 42 -4.0          # 42.0 × -4.0 = -168.0
./calculator.py --div 42 -4.0          # division flottante par défaut
./calculator.py --div --int 42 -4.0    # 42.0 // -4.0 = -11
./calculator.py --div --float 42 0     # Error: Division by zero is not allowed.
```

**`tools.py`** — boîte à outils NLP (téléchargement Gutenberg + prétraitement) :

```bash
./tools.py --info 11                                  # métadonnées du catalogue
./tools.py --download 11                              # sauvegarde le livre en UTF-8
./tools.py --clean "Oh dear!   Oh dear!" --lower      # nettoyage de texte
./tools.py --tokenize "Alice fell down." --sent       # tokenisation (mots / phrases)
./tools.py --tokenize "Alice fell down." --punct --stop  # + retrait ponctuation/stopwords
./tools.py --postag "The Caterpillar was first."      # POS tagging (Penn Treebank)
./tools.py --normalize "going better" --stem          # normalisation (stem / lemme)
```

| Option `tools.py` | Effet | Flags |
|---|---|---|
| `--info ID` | Métadonnées du catalogue Gutenberg | — |
| `--download ID` | Télécharge le livre en Plain Text UTF-8 | — |
| `--clean TEXT` | Retire l'en-tête/pied Gutenberg, normalise les espaces | `--lower` |
| `--tokenize TEXT` | Segmente en mots / phrases | `--sent` `--stop` `--punct` |
| `--postag TOKENS` | Étiquette grammaticale par token | — |
| `--normalize TOKENS` | Lemmatisation (défaut) ou stemming | `--stem` |

### `bookworm.py` — CLI **officielle** du sujet

C'est le script qui répond mot pour mot à la nomenclature demandée :

| Option | Effet | Type de sortie |
|---|---|---|
| `--lexdiv ID` | Mesures de diversité lexicale | `dict` (clés `tok`, `typ`, `hap`, `ttr`, `mwl`, `mwf`) |
| `--topics ID` | Top-10 mots du sujet principal de chaque section | `dict` `{1: [...], 2: [...], ...}` |
| `--entities ID` | Personnages et lieux du livre | `dict` `{"characters": [...], "locations": [...]}` |
| `--summarize ID` | Résumé en quelques phrases | `str` |
| `--similar ID` | 5 titres similaires (catalogue de 21 livres imposé) | `list[str]` |
| `--card ID` | Carte de livre complète (info + tous les NLP) | `dict` agrégé |

Options communes :
- `--no-cache` : recalcule sans relire le cache disque (`outputs/cache/<ID>/<task>.json`).
- `--sentences N` : longueur du résumé (option `--summarize` uniquement).

Les ID sont ceux de Project Gutenberg. Exemples sur le catalogue imposé :

```bash
python3 bookworm.py --lexdiv     11      # Alice's Adventures in Wonderland
python3 bookworm.py --topics     84      # Frankenstein
python3 bookworm.py --entities   1661    # The Adventures of Sherlock Holmes
python3 bookworm.py --summarize  345     # Dracula
python3 bookworm.py --similar    12      # Through the Looking-Glass
python3 bookworm.py --card       16      # Peter Pan
```

**Caching disque** : chaque résultat coûteux (téléchargement Gutenberg, LDA, résumé
hiérarchique, similarité de tout le catalogue) est sauvegardé dans
`outputs/cache/<ID>/<task>.json`. Le second appel est quasi-instantané.

### `main.py` — CLI étendue / visualisations

Le second point d'entrée fournit des sous-commandes plus exploratoires et tous les
exports d'images. Sept sous-commandes :

| Commande | Rôle |
|---|---|
| `summarize` | Résume un livre (TextRank / LexRank / LSA, mode hiérarchique pour les longs textes) |
| `classify`  | Identifie genre dominant (lexique) + sujets latents (LDA, globaux et par section avec `--per-section`) |
| `analyze`   | Statistiques de style, diversité lexicale (7 mesures), sentiment, personnages, lieux |
| `corpus`    | Clusterise un dossier de livres (TF-IDF + K-Means + similarité cosinus) |
| `similar`   | Recommande les livres les plus proches d'un livre cible dans un corpus |
| `book`      | « Fiche de livre » : un dictionnaire récapitulatif complet (méta + stats + résumé + …) |
| `demo`      | Pipeline complète sur Alice in Wonderland, exporte les visuels |

### Source du livre

Toutes les commandes (sauf `corpus` et `demo`) acceptent :

- `--file PATH` : un fichier `.txt` local (un en-tête `Title:` / `Author:` est lu si présent)
- `--gutenberg ID` : un identifiant numérique de Project Gutenberg (téléchargé puis mis en cache dans `data/books/`)

### Exemples

```bash
# Comparer les trois algorithmes de résumé sur Alice
python3 main.py summarize --gutenberg 11 --compare

# Forcer le résumé hiérarchique (deux passes — utile pour les très longs livres)
python3 main.py summarize --file data/books/moby_dick.txt --hierarchical --sentences 12

# Classer un livre français
python3 main.py classify --file data/books/candide.txt --lang french

# Exporter le rapport JSON + nuage de mots + courbe de sentiment
python3 main.py analyze --gutenberg 84 --output outputs/

# Clusteriser un corpus en 4 groupes
python3 main.py corpus --dir data/books --clusters 4 --output outputs/
```

---

## Utilisation — interface graphique (Streamlit)

```bash
streamlit run app.py
```

Onglets disponibles : Résumé · Genres · Sujets latents · Statistiques · Mots-clés · Texte brut.
Une dizaine de classiques anglais sont préchargés (Alice, Frankenstein, Pride and Prejudice,
Sherlock Holmes, Dracula, Moby Dick, The Time Machine, Peter Pan…) ; tu peux aussi téléverser
ton propre `.txt`.

---

## Architecture

```
T-ALICE/
├── calculator.py           # BOOTSTRAP — calculatrice CLI (argparse)
├── tools.py                # BOOTSTRAP — boîte à outils NLP (info/download/clean/tokenize/postag/normalize)
├── bookworm.py             # *** CLI officielle du sujet (6 options + cache) ***
├── main.py                 # CLI étendue (7 sous-commandes : démo, showcase, visuels)
├── app.py                  # interface Streamlit
├── requirements.txt
├── src/
│   ├── loader.py           # Project Gutenberg + fichiers locaux + cache disque + BookLoadError
│   ├── preprocessing.py    # nettoyage, tokenisation, lemmatisation, chapitrage, chunking
│   ├── summarizer.py       # TextRank, LexRank, LSA, résumé hiérarchique
│   ├── classifier.py       # genres, K-Means, LDA, topics_per_section, find_similar_books
│   ├── analyzer.py         # stats, sentiment, diversité (7 mesures), characters / places, book_sheet
│   └── visualization.py    # nuage de mots, courbe émotionnelle, heatmap de similarité
├── docs/
│   ├── pipelines.md        # diagrammes Mermaid des pipelines (topics, NER, résumé, reco)
│   └── defense.md          # antisèche orale + justification des choix techniques
├── data/books/             # textes .txt (régénérables, non versionnés)
├── outputs/                # JSON + PNG + cache bookworm (régénérables, non versionnés)
└── tests/test_pipeline.py  # 21 tests (sans pytest : `python3 tests/test_pipeline.py`)
```

---

## Concepts NLP utilisés

Cette section explique les notions de base qui sous-tendent la pipeline. Elles
sont volontairement réduites à l'essentiel pour rester lisibles.

### Nettoyage

Avant **toute** opération, chaque fichier passe par `preprocessing.clean_text()` :
suppression de l'en-tête / pied de page légal Project Gutenberg
(`_strip_gutenberg_boilerplate`), normalisation des fins de ligne CRLF→LF,
écrasement des espaces multiples et des sauts de ligne triplés. Sans ce
nettoyage, le TextRank inclurait par exemple la licence Gutenberg dans le résumé.

### Mots vides (stopwords)

Les **mots vides** sont des mots fréquents mais sémantiquement pauvres
(« le », « of », « and », « il », « est »…). Pris isolément, ils ne portent quasiment
aucune information sur le contenu d'un texte mais ils dominent les statistiques
de fréquence : le top-10 brut d'un livre serait presque toujours `the, of, and, …`.

**Ce qu'on en fait au prétraitement** : on les **retire** systématiquement avant
toute opération basée sur le vocabulaire (TF-IDF, LDA, top mots-clés, scoring de
genre). On les **garde** uniquement quand on a besoin de phrases entières
(résumé extractif, statistiques de longueur de phrase).

Implémentation : `nltk.corpus.stopwords` (multi-langue, déjà entraîné) via
`preprocessing._stopwords(lang)`.

### Normalisation des jetons

Deux grandes techniques permettent de ramener plusieurs formes d'un même mot
à une représentation commune :

| Technique | Comment | Avantages | Inconvénients |
|---|---|---|---|
| **Stemming** (racinisation) | Coupe les suffixes à l'aide de règles (« running » → « run », « studies » → « studi ») | Très rapide, indépendant du dictionnaire, robuste aux mots inconnus | Produit des « racines » qui ne sont pas des mots réels ; agressif → fausses fusions |
| **Lemmatisation** | Ramène chaque mot à sa **forme canonique de dictionnaire** (lemme), avec connaissance du POS (« was » → « be », « better » → « good ») | Résultat lisible, sémantiquement juste | Plus lent, dépend d'un dictionnaire (WordNet), POS-tagging utile |

**Principale différence** : le stemmer est purement morphologique (algorithmes
de Porter / Snowball), le lemmatiseur est *lexical* (il connaît les mots).

Choix dans T-ALICE : **lemmatisation** par défaut (`WordNetLemmatizer`) parce
qu'on veut afficher de vrais mots dans les mots-clés et les sujets latents.
Le stemmer reste utilisable dans `sumy` (résumé) où la sortie n'est jamais
montrée à l'utilisateur.

### Vectorisation

Transformer du texte en vecteurs numériques est indispensable pour appliquer
n'importe quel algorithme de ML / clustering. Trois méthodes courantes :

1. **Bag of Words (CountVectorizer)** — un vecteur de taille = vocabulaire, chaque
   composante = nombre d'occurrences du mot dans le document. Simple, interprétable,
   utilisé ici pour **LDA** (qui veut des entiers).
2. **TF-IDF (TfidfVectorizer)** — pondère le BoW par l'inverse de la fréquence
   documentaire (`tf × log(N/df)`). Les mots fréquents dans **tous** les
   documents sont écrasés, ceux discriminants pour un document précis ressortent.
   Utilisé ici pour **K-Means** (clustering de corpus) et **find_similar_books**
   (similarité cosinus entre TF-IDF).
3. **Word embeddings** (Word2Vec, GloVe, fastText) — chaque mot devient un vecteur
   dense de dimension fixe (~300) appris à partir de cooccurrences. On peut
   ensuite agréger en moyenne pour représenter une phrase ou un document. Non
   utilisé ici par choix pédagogique (pas de modèle à télécharger), mais facile
   à brancher.

### Mesures de diversité lexicale (`analyzer.lexical_diversity`)

Sept indicateurs renvoyés dans un seul dictionnaire :

| Mesure | Formule | Lecture |
|---|---|---|
| TTR | V / N | Rapide mais sensible à la longueur |
| RTTR (Guiraud) | V / √N | Corrige partiellement |
| CTTR (Carroll) | V / √(2N) | Autre correction classique |
| Herdan C | log V / log N | Robuste, stable entre textes |
| Maas a² | (log N − log V) / log² N | Plus stable sur longs textes |
| Yule K | 10⁴(Σ m²f_m − N) / N² | Concentration du vocabulaire (faible = varié) |
| Hapax ratio | mots vus 1 fois / V | Indicateur de richesse de rare-vocabulaire |

---

## Diagrammes des pipelines

> 📊 **Version Mermaid (rendu graphique sur GitHub) : [`docs/pipelines.md`](docs/pipelines.md).**
> Antisèche de défense orale et justification des choix : [`docs/defense.md`](docs/defense.md).
> Les schémas ASCII ci-dessous en sont l'équivalent texte, autonome.

### Résumé (summarize)

```
.txt ──► loader ──► clean_text ──► chunk_by_words (1500 mots)
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │ pour chaque chunk :   │
                              │ sumy.TextRank (4 ph.) │
                              └───────────────────────┘
                                          │
                                          ▼
                             concaténation des résumés
                                          │
                                          ▼
                              sumy.TextRank (N phrases)
                                          │
                                          ▼
                                   résumé final
```

### Modélisation des sujets (topics)

```
.txt ──► clean_text ──► split_into_chapters ──► chunk_by_words(500)
                                                      │
                                                      ▼
                                          tokenize + stopwords + lemme
                                                      │
                                                      ▼
                                      CountVectorizer (BoW, min_df=2)
                                                      │
                                                      ▼
                              LatentDirichletAllocation (n_topics)
                                                      │
                                                      ▼
                          top-k mots par sujet  +  poids relatif
```

`topics_per_section` boucle ce pipeline sur chaque chapitre détecté.

### Reconnaissance d'entités nommées (NER)

```
.txt ──► découpe en phrases (regex .!?)
              │
              ▼
   on saute le 1er mot (toujours capitalisé)
              │
              ▼
   regex \b[A-Z][a-z]{2,}(\s+[A-Z][a-z]{2,})?\b
              │
              ▼
   ┌──────────────┴──────────────┐
   ▼                              ▼
 mot précédent ∈ titres        mot précédent ∈ prépositions de lieu
 (Mr, Lady, Captain…)          (in, at, to, from…)
   │ ou                          │ ou
 mot précédent ∈ verbes de      suffixe ∈ {-ville, -shire, -town…}
 dialogue (said, asked…)         │
   │ ou                          ▼
 prénom + nom                LIEU (extract_places)
   │
   ▼
 PERSONNAGE (extract_characters)
```

### Résumé NLP-par-graphe (TextRank)

```
phrases ──► matrice de similarité (cosinus TF-IDF)
                          │
                          ▼
              graphe pondéré (sentences = nœuds)
                          │
                          ▼
                PageRank itératif (sumy)
                          │
                          ▼
            top-N phrases par score → résumé
```

### Recommandation de livres similaires

```
corpus de .txt ──► tokenize + lemme   ┐
                                       ├──► TF-IDF (max_features=5000)
livre cible    ──► tokenize + lemme   ┘
                                                │
                                                ▼
                          cosine_similarity(target, corpus)
                                                │
                                                ▼
                                  tri décroissant, top-N
                                                │
                                                ▼
                              [{title, author, similarity}, …]
```

### Flux d'une carte `bookworm.py --card ID`

```
                          ┌────────────────────────────────────────┐
   Project Gutenberg ID ──►│ load_from_gutenberg(ID) → cache .txt   │
                          └──────────────┬─────────────────────────┘
                                         │
                          ┌──────────────┼─────────────────┐
                          ▼              ▼                 ▼
                 ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
                 │ --lexdiv    │  │ --topics     │  │ --entities   │
                 │ tok/typ/hap │  │ LDA / section│  │ pers + lieux │
                 │ ttr/mwl/mwf │  │  → top-10    │  │              │
                 └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
                        │                │                 │
                        │   ┌────────────┴───────────────┐ │
                        │   │   --summarize (TextRank)   │ │
                        │   │   --similar (TF-IDF cos.)  │ │
                        │   └────────────────────────────┘ │
                        ▼                                  ▼
                  ╔══════════════════════════════════════════╗
                  ║   --card  →  dict {info, lexdiv, topics, ║
                  ║              entities, summary, similar} ║
                  ╚══════════════════════════════════════════╝
                                  │
                                  ▼
                  Chaque résultat est mis en cache dans
                  outputs/cache/<ID>/<task>.json (réutilisable).
```

### Flux d'une commande `book` (fiche de livre, via main.py)

```
livre ──► metadata
        ├─► text_statistics
        ├─► lexical_diversity   (7 mesures)
        ├─► analyze_sentiment
        ├─► top_keywords
        ├─► extract_characters / extract_places
        ├─► classify_by_keywords (genres)
        ├─► discover_topics      (LDA)
        └─► hierarchical_summary (résumé)
                    │
                    ▼
        dict unique  →  JSON   +  affichage CLI
```

---

## Tests

```bash
python3 tests/test_pipeline.py        # runner intégré, sans pytest
# ou
pytest tests/ -v                      # si pytest est installé
```

**21 tests**, environ 5 secondes, sans appel réseau (les tests `bookworm`
court-circuitent le téléchargement via monkey-patching). Ils couvrent :

- chargement d'un livre + erreur sur fichier manquant
- tokenisation + retrait des stopwords, découpage en phrases
- détection de chapitres (cas dégradé)
- résumé TextRank (compression dans `]0, 1[`)
- classification par lexique (score « mystère » dominant sur le sample mystery)
- rapport d'analyse complet + fiche de livre
- diversité lexicale (≥ 5 mesures, toutes `float`)
- sujets par section, livres similaires (tri décroissant, cible exclue)
- `bookworm` : clés exactes de `--lexdiv`, format de `--topics` / `--entities`,
  structure de `--card`, catalogue de 21 livres, rejet des flags inconnus et
  des collisions de flags

---

## Choix techniques

| Domaine | Bibliothèque | Pourquoi |
|---|---|---|
| Résumé extractif | `sumy` (TextRank, LexRank, LSA) | Trois algorithmes pédagogiquement distincts, sans dépendance GPU |
| Vectorisation | `scikit-learn` (TF-IDF, CountVectorizer) | Standard, rapide, bien documenté |
| Clustering | `KMeans` + similarité cosinus | Conserve l'interprétabilité (top-termes par cluster) |
| Sujets latents | `LatentDirichletAllocation` | Découvre des thèmes sans étiquettes |
| Tokenisation / phrases | `nltk` | Multi-langue, gestion robuste des contractions |
| Sentiment | Lexique positif/négatif maison | Pas de modèle à télécharger, transparent, suffisant pour une vue d'ensemble |
| Entités nommées | Heuristique de casse | Marche bien sur des romans en anglais ; pour spaCy, ajouter une dépendance |
| Visualisation | `matplotlib` + `wordcloud` | Backend Agg → utilisable sans display (CLI, Streamlit) |
| UI | `streamlit` | Une page, six onglets, zéro JS écrit |

### Et un modèle abstractif (BART, T5, GPT-…) ?

Volontairement laissé de côté : c'est un projet de découverte du NLP « classique », et
les modèles transformer demandent de télécharger plusieurs centaines de Mo de poids,
ce qui dilue le côté pédagogique. Brancher `transformers` est trivial à faire en
extension :

```python3
# src/summarizer.py — extension possible
from transformers import pipeline
abstractive = pipeline("summarization", model="facebook/bart-large-cnn")
```

---

## Limites connues

- Le résumé extractif copie des phrases entières du livre — il ne reformule pas.
- L'analyse de sentiment lexicale ne gère ni la négation, ni l'ironie, ni le sarcasme.
  Pour Alice qui dit « curiouser and curiouser » avec joie, un modèle pré-entraîné
  ferait mieux.
- L'extraction d'entités basée sur la casse confond personnages, lieux et institutions.
- LDA sur un seul livre reste exploratoire : les sujets sont plus clairs sur un corpus.
