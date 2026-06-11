# Diagrammes des pipelines NLP — T-ALICE

Ce document illustre, sous forme de diagrammes, le **parcours de traitement** de
chaque grande fonctionnalité du projet. Les diagrammes sont écrits en
[Mermaid](https://mermaid.js.org/) : ils s'affichent automatiquement sur GitHub
et peuvent être exportés en image pour une présentation.

> Convention de lecture : un rectangle = une étape de code (fonction réelle du
> projet), un losange = une décision, un cylindre = une sortie persistée.

---

## 0. Vue d'ensemble — du livre à la « book card »

```mermaid
flowchart LR
    G[Project Gutenberg<br/>ID] --> L[loader.load_from_gutenberg]
    F[Fichier .txt local] --> L2[loader.load_from_file]
    L --> CLEAN[Nettoyage<br/>strip boilerplate + clean_text]
    L2 --> CLEAN
    CLEAN --> P[preprocessing<br/>tokenize / sentences / lemmatize]
    P --> A1[analyzer<br/>lexdiv · sentiment · entités]
    P --> A2[classifier<br/>topics LDA · genres · similarité]
    P --> A3[summarizer<br/>TextRank hiérarchique]
    A1 --> CARD[bookworm.task_card]
    A2 --> CARD
    A3 --> CARD
    CARD --> OUT[(outputs/cache/&lt;ID&gt;/card.json)]
```

**Point clé (trophée « faire le ménage ») :** le nettoyage est la **première**
étape, avant toute tokenisation, vectorisation ou analyse. Sans lui, l'en-tête
légal Gutenberg polluerait les résumés et les fréquences de mots.

---

## 1. Modélisation des sujets — `--topics` (trophée *sujets-doc*)

```mermaid
flowchart TD
    A[Texte du livre nettoyé] --> B[split_into_chapters<br/>regex 'Chapter/Book/Part']
    B --> C{≥ 2 chapitres<br/>de ≥ 100 mots ?}
    C -->|oui| D[Sections = chapitres]
    C -->|non| E[chunk_by_words 2000<br/>blocs de secours]
    D --> F[Pour CHAQUE section]
    E --> F
    F --> G[chunk_by_words 500<br/>→ pseudo-documents]
    G --> H[tokenize<br/>minuscules + ponctuation retirée<br/>+ stopwords retirés + lemmatisation]
    H --> I[CountVectorizer<br/>Bag-of-Words, min_df = 2]
    I --> J[LatentDirichletAllocation<br/>n_components = 1]
    J --> K{LDA renvoie<br/>≥ 10 mots ?}
    K -->|oui| L[Top-10 mots du sujet saillant]
    K -->|non, section courte| M[Complète avec les lemmes<br/>les plus fréquents hors stopwords]
    M --> L
    L --> N[(dict — section_id → liste de 10 mots)]
```

**Choix de modélisation :** une LDA par section (`n_components=1`) donne le sujet
*dominant* de chaque chapitre — plus lisible pour une fiche qu'une LDA globale
qui mélange tous les thèmes. Le repli sur les fréquences garantit toujours
10 mots, même sur un chapitre trop court pour que LDA converge.

---

## 2. Reconnaissance d'entités nommées — `--entities` (trophée *entités_doc*)

```mermaid
flowchart TD
    A[Texte du livre] --> B[Découpe en phrases<br/>regex sur . ! ?]
    B --> C[Ignore le 1er mot de chaque phrase<br/>toujours capitalisé → faux positif]
    C --> D["Regex noms propres :<br/>[A-Z][a-z]{2,} (+ 2e mot optionnel)"]
    D --> E{Quel est le mot<br/>PRÉCÉDENT ?}
    E -->|titre : Mr, Lady, Captain, Dr…| P[PERSONNAGE]
    E -->|verbe de dialogue : said, asked, replied…| P
    E -->|préposition de lieu : in, at, to, from…| Q[LIEU]
    D --> F{Deux mots capitalisés<br/>« John Smith » ?}
    F -->|oui| P
    D --> G{Suffixe toponymique ?<br/>-ville, -shire, -ton, -land…}
    G -->|oui| Q
    P --> H[extract_characters<br/>Counter → top-N<br/>moins les lieux détectés]
    Q --> I[extract_places<br/>Counter → top-N]
    H --> J[(dict — characters / locations)]
    I --> J
```

**Choix de NER :** heuristique par **casse + contexte**, sans dépendance lourde
(pas de spaCy ni de modèle à télécharger → meilleure *portabilité*). La
désambiguïsation personnage/lieu repose sur le mot précédent (titre/verbe de
dialogue ⇒ personne ; préposition de lieu/suffixe ⇒ lieu). Limite assumée :
quelques faux positifs sur les romans (institutions, apostrophes).

---

## 3. Résumé automatique — `--summarize` (trophée *résumé_doc*)

```mermaid
flowchart TD
    A[Livre nettoyé] --> B{Texte long ?<br/>> 1500 mots}
    B -->|non| D[summarize direct<br/>TextRank → N phrases]
    B -->|oui| E[chunk_by_words 1500]
    E --> F[Passe 1 — pour chaque bloc :<br/>TextRank → 4 phrases]
    F --> G[Concatène les résumés partiels]
    G --> H[Passe 2 — TextRank → N phrases]
    D --> Z[(Résumé final — string)]
    H --> Z

    subgraph TR["Cœur TextRank (extractif, par graphe)"]
        direction TB
        T1[Phrases = nœuds du graphe] --> T2[Arêtes = similarité cosinus<br/>entre phrases]
        T2 --> T3[Graphe pondéré]
        T3 --> T4[PageRank itératif<br/>score d'importance par phrase]
        T4 --> T5[Sélection des N phrases<br/>les mieux notées]
    end
```

**Choix de résumé :** **extractif** (TextRank via `sumy`) plutôt qu'abstractif
(BART/T5) — pas de modèle de plusieurs centaines de Mo à télécharger, traitement
transparent et reproductible. Le **résumé hiérarchique en 2 passes** est
indispensable pour un livre entier : résumer 25 000 mots d'un coup fait exploser
la mémoire de l'algorithme de graphe.

---

## 4. Recommandation de livres similaires — `--similar` (trophée *document similaire*)

```mermaid
flowchart TD
    T[Livre cible] --> TK[tokenize + lemmatisation]
    C[Corpus : 21 livres du catalogue imposé] --> CK[tokenize + lemmatisation]
    TK --> V[TfidfVectorizer<br/>max_features = 5000, min_df = 1]
    CK --> V
    V --> M[Matrice TF-IDF<br/>1 ligne = 1 livre]
    M --> S[cosine_similarity<br/>cible vs tous les autres]
    S --> R[Tri décroissant par similarité]
    R --> X[Exclut la cible elle-même]
    X --> O[(Top-5 titres similaires — list&lt;str&gt;)]
```

**Choix de reco :** **TF-IDF + similarité cosinus**. TF-IDF écrase les mots
communs à tous les livres et fait ressortir le vocabulaire discriminant ; le
cosinus mesure l'angle entre deux profils lexicaux, indépendamment de la
longueur des livres. Simple, interprétable, sans entraînement.

---

## 5. Carte de livre — `--card` (trophée *carte de livre*)

```mermaid
flowchart TD
    ID[Gutenberg ID] --> LOAD[load_from_gutenberg<br/>+ cache disque .txt]
    LOAD --> T1[task_lexdiv]
    LOAD --> T2[task_topics]
    LOAD --> T3[task_entities]
    LOAD --> T4[task_summarize]
    LOAD --> T5[task_similar]
    LOAD --> INFO[_info_for<br/>id · authors · bookshelves]
    T1 --> AGG[task_card — agrégation]
    T2 --> AGG
    T3 --> AGG
    T4 --> AGG
    T5 --> AGG
    INFO --> AGG
    AGG --> J[(dict : info · lexdiv · topics<br/>· entities · summary · similar)]
    J --> CACHE[(outputs/cache/&lt;ID&gt;/card.json)]
```

Chaque sous-tâche est **mise en cache** individuellement : un second appel
(`--card`, ou n'importe quelle sous-commande) relit le JSON au lieu de tout
recalculer.

---

## Export en image (pour la présentation)

Les diagrammes ci-dessus se rendent nativement sur GitHub. Pour les projeter en
slides, deux options :

```bash
# 1) Mermaid CLI (Node)
npx -p @mermaid-js/mermaid-cli mmdc -i docs/pipelines.md -o docs/pipeline.png

# 2) Éditeur en ligne : copier un bloc ```mermaid``` dans https://mermaid.live
#    puis « Export PNG/SVG ».
```
