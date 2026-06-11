# Fiche de défense — arguments & réponses aux trophées

Antisèche pour l'oral et source de contenu pour les slides. Chaque section
répond à un trophée « conceptuel » par des phrases prêtes à dire, avec les
arguments qui justifient nos choix.

---

## 1. Faire le ménage (nettoyage)

**Ce qu'on fait, dans l'ordre, AVANT toute autre opération :**
1. Retrait de l'en-tête et du pied de page légaux Project Gutenberg
   (`_strip_gutenberg_boilerplate`, marqueurs `*** START/END OF … ***`).
2. Normalisation des fins de ligne `CRLF → LF`.
3. Écrasement des espaces / tabulations multiples en un seul espace.
4. (optionnel) passage en minuscules.

**Argument :** sans nettoyage, le résumé TextRank ressortirait des phrases de la
licence Gutenberg et le top des mots serait pollué. Le nettoyage conditionne la
qualité de **toutes** les étapes en aval → c'est pour ça qu'il est en premier.

---

## 2. Tokenisation

**Définition à dire :** segmenter un texte continu en unités (« tokens ») —
soit en **mots**, soit en **phrases**.

- Mots : `word_tokenize` (NLTK) — gère les contractions (« don't » → « do », « n't »).
- Phrases : `sent_tokenize` (NLTK, modèle Punkt) — ne coupe pas sur « Mr. » ou « Dr. ».

**Pourquoi NLTK plutôt qu'un `split(" ")` :** un split naïf colle la ponctuation
aux mots (« late. ») et coupe mal les phrases. Punkt est entraîné sur de vrais
corpus.

---

## 3. Mots vides (stopwords)

**Définition :** mots très fréquents mais pauvres en sens (« the », « of »,
« and », « le », « est »…).

**Ce qu'on en fait :**
- On les **retire** avant tout ce qui repose sur le vocabulaire : TF-IDF, LDA,
  top mots-clés, scoring de genre. Sinon le top-10 d'un livre serait toujours
  `the, of, and, to…`.
- On les **garde** quand on a besoin de phrases entières et grammaticales :
  résumé extractif, calcul de la longueur moyenne de phrase.

**Source :** `nltk.corpus.stopwords` (listes multilingues pré-établies).

---

## 4. Normalisation des tokens — LES DEUX TECHNIQUES

| | **Stemming** (racinisation) | **Lemmatisation** |
|---|---|---|
| Principe | Coupe les suffixes par règles | Ramène au lemme = forme de dictionnaire |
| Exemple | running → run, studies → studi | was → be, better → good |
| Connaît les mots ? | Non (purement morphologique) | Oui (lexical, via WordNet) |
| **Avantages** | Très rapide, robuste aux mots inconnus, indépendant d'un dictionnaire | Résultat = vrais mots, sémantiquement juste |
| **Inconvénients** | Produit des « racines » non-mots, fusions abusives | Plus lent, dépend de WordNet + POS-tagging |

**Différence principale à dire :** le stemmer est **morphologique** (Porter/Snowball,
des règles de découpe) ; le lemmatiseur est **lexical** (il connaît le vocabulaire
et la nature grammaticale du mot).

**Notre choix :** lemmatisation par défaut, car on **affiche** les mots (mots-clés,
sujets) → ils doivent être lisibles. **Amélioration de précision :** on passe le
**POS tag** au lemmatiseur (`was`+verbe → `be`), sinon WordNet suppose « nom » par
défaut et rate `better → good`. Le stemming reste dispo (`tools.py --normalize --stem`).

---

## 5. Vectorisation — AU MOINS DEUX MÉTHODES

| Méthode | Comment ça marche | Où on l'utilise |
|---|---|---|
| **Bag-of-Words** (`CountVectorizer`) | Vecteur de taille = vocabulaire ; chaque case = nombre d'occurrences du mot | **LDA** (les sujets latents ont besoin de comptes entiers) |
| **TF-IDF** (`TfidfVectorizer`) | BoW pondéré par `tf × log(N/df)` : écrase les mots présents partout, fait ressortir les mots discriminants | **K-Means** (clustering) et **similarité** entre livres |
| **Word embeddings** (Word2Vec/GloVe) | Chaque mot = vecteur dense ~300 dim. appris sur les cooccurrences ; sémantique capturée | Non utilisé (choix : zéro modèle à télécharger), mais branchable |

**Phrase clé :** BoW compte, TF-IDF **pondère** pour faire ressortir ce qui est
spécifique à un document. Les embeddings vont plus loin (sens), au prix d'un
modèle pré-entraîné.

---

## 6. Diversité lexicale (≥ 5 mesures)

On renvoie **7 mesures** dans un seul dict (`analyzer.lexical_diversity`) :
TTR, RTTR (Guiraud), CTTR (Carroll), Herdan C, Maas a², Yule K, Hapax ratio.

**Pourquoi plusieurs :** le TTR seul chute mécaniquement quand le texte
s'allonge (plus de mots ⇒ plus de répétitions). RTTR/CTTR/Herdan/Maas corrigent
cet effet de longueur ; Yule K mesure la concentration du vocabulaire ; Hapax
compte les mots vus une seule fois. On peut donc comparer des livres de tailles
différentes.

> Note : `bookworm.py --lexdiv` renvoie les clés courtes imposées par le sujet
> (`tok, typ, hap, ttr, mwl, mwf`) ; `analyzer.lexical_diversity` en fournit 7
> académiques pour `main.py`. Les deux satisfont « ≥ 5 mesures ».

---

## 7. Justification des outils (à défendre)

| Besoin | Outil retenu | Pourquoi LUI et pas un autre |
|---|---|---|
| Résumé | `sumy` (TextRank/LexRank/LSA) | Extractif, 3 algos pédagogiques, **aucun GPU/modèle lourd** vs `transformers` (BART = 400+ Mo) |
| Vectorisation | `scikit-learn` | Standard de l'industrie, TF-IDF + BoW intégrés, rapide |
| Sujets latents | `LDA` (sklearn) | Découvre des thèmes **sans étiquettes** (non supervisé) |
| Clustering | `KMeans` + cosinus | Garde l'interprétabilité (top-termes par cluster) |
| Tokenisation | `nltk` | Multilingue, Punkt robuste aux abréviations |
| Sentiment | Lexique maison | Transparent, zéro téléchargement, suffisant pour une vue d'ensemble |
| NER | Heuristique casse+contexte | Pas de spaCy → installation légère, **portable** |
| Visualisation | `matplotlib` + `wordcloud` | Backend Agg → marche sans écran (CLI/serveur) |
| UI | `streamlit` | Une page, zéro JavaScript à écrire |

**Arbitrage central à assumer :** on a privilégié le NLP **classique,
transparent et portable** au NLP **transformer** (plus précis mais lourd et
opaque). C'est cohérent avec un projet de **découverte** : on veut comprendre
chaque brique, pas appeler une boîte noire.

---

## 8. Robustesse (exemples concrets)

- ID Gutenberg inexistant → `BookLoadError` + message clair, code de sortie 2.
- Fichier local manquant → message « Fichier introuvable ».
- Division par zéro (`calculator.py`) → « Error: Division by zero is not allowed. ».
- Mauvais nombre d'arguments → message explicite.
- Flag inconnu / collision de flags → argparse rejette proprement (code 2),
  **sans** changer le comportement des autres options (trophée *collision*).
- Erreur d'usage `bookworm.py` → on **liste les 21 IDs** du catalogue pour guider.

---

## 9. Pitch d'ouverture (30 s, pour les slides)

> « T-ALICE prend un livre entier de Project Gutenberg et en sort, en une
> commande, une *carte de livre* : mesures de diversité lexicale, sujets par
> chapitre, personnages et lieux, résumé en quelques phrases et cinq livres
> similaires. Toute la pipeline repose sur du NLP classique — TF-IDF, LDA,
> TextRank — choisi pour rester transparent, reproductible et installable
> partout, sans modèle de plusieurs centaines de mégaoctets. »
