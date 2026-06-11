"""Génération d'images : nuage de mots, courbe de sentiment, matrice de similarité."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend non-interactif : indispensable pour le CLI / Streamlit.
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


def save_wordcloud(keywords: list[tuple[str, int]], output_path: str | Path) -> Path:
    """Génère un nuage de mots à partir d'une liste (mot, fréquence)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freqs = dict(keywords)
    if not freqs:
        freqs = {"vide": 1}
    wc = WordCloud(width=1200, height=600, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(freqs)
    wc.to_file(str(output_path))
    return output_path


def save_sentiment_arc(arc: list[float], output_path: str | Path, title: str = "Courbe émotionnelle") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(arc))
    ax.plot(x, arc, color="#2563eb", linewidth=2)
    ax.fill_between(x, arc, 0, where=[v >= 0 for v in arc], color="#10b981", alpha=0.35)
    ax.fill_between(x, arc, 0, where=[v < 0 for v in arc], color="#ef4444", alpha=0.35)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Progression du livre →")
    ax.set_ylabel("Polarité (-1 à +1)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
    return output_path


def save_similarity_heatmap(
    matrix: list[list[float]],
    labels: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) * 0.8)))
    im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center",
                    color="white" if arr[i, j] < 0.5 else "black", fontsize=8)
    ax.set_title("Similarité cosinus entre livres")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
    return output_path
