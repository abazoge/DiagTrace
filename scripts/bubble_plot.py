import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from utils import get_data, llm_types

# ---------- Bubble-plot helper ----------
def bubble_agreement_plot(x, y, x_col, y_col, ax=None,
                          xlab="", ylab="",
                          xmin=1, xmax=18, ymin=1, ymax=18,
                          layers=20, size_scale=60):
    if ax is None:
        ax = plt.gca()

    tab = pd.crosstab(x, y).astype(float)
    idx = pd.Index(range(int(xmin), int(xmax)+1), name=x_col)
    cols = pd.Index(range(int(ymin), int(ymax)+1), name=y_col)
    tab = tab.reindex(index=idx, columns=cols, fill_value=0.0)

    if tab.values.max() > 0:
        d = 0.16 + tab / tab.values.max()
        d[d == 0.16] = 0.0
    else:
        d = tab.copy()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticks(range(int(xmin), int(xmax)+1))
    ax.set_yticks(range(int(ymin), int(ymax)+1))
    ax.tick_params(axis='both', which='major')
    ax.grid(False)

    ax.plot([xmin, xmax], [ymin, ymax], linestyle="--", color="grey",
            linewidth=2, dashes=(4, 4))

    base_color = (0.12, 0.56, 1.0)
    for i in d.index:
        for j in d.columns:
            v = float(d.loc[i, j])
            if v <= 0:
                continue
            for t in np.linspace(v, 0, layers):
                s = (size_scale * t)**2
                ax.scatter(i, j, s=s, c=[base_color], alpha=0.2,
                           marker='o', edgecolors='none')
    return ax

# ---------- Configuration ----------
models = [
    # Top row
    {"fr": "o3_FR",         "en": "o3_EN",          "name": "o3"},
    {"fr": "DeepSeek_FR",   "en": "DeepSeek_EN",   "name": "DeepSeek-R1"},
    {"fr": "GPT_FR",        "en": "GPT_EN",        "name": "GPT-4-Turbo"},
    # Bottom row
    {"fr": "Llama_FR",      "en": "Llama_EN",      "name": "Llama-3.1-405B-Instruct"},
    {"fr": "Biomistral_FR", "en": "Biomistral_EN",  "name": "BioMistral-7B"},
]

p_adjusted = {
    "o3_FR":         r"p = 1.000",
    "DeepSeek_FR":   r"p = 0.0207",
    "GPT_FR":        r"p < 0.001",
    "Llama_FR":      r"p < 0.001",
    "Biomistral_FR": r"p = 0.006",
}

panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)']

# ---------- Create figure with gridspec ----------
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.45)

axes = [
    fig.add_subplot(gs[0, 0:2]),  # top left
    fig.add_subplot(gs[0, 2:4]),  # top center
    fig.add_subplot(gs[0, 4:6]),  # top right
    fig.add_subplot(gs[1, 1:3]),  # bottom left (centered)
    fig.add_subplot(gs[1, 3:5]),  # bottom right (centered)
]

# ---------- Plot all models ----------
for ax, model, letter in zip(axes, models, panel_letters):
    data_fr = get_data(model["fr"])
    data_en = get_data(model["en"])

    notes_fr = np.array(data_fr.Note)
    notes_en = np.array(data_en.Note)

    bubble_agreement_plot(
        notes_fr, notes_en,
        x_col=model["fr"],
        y_col=model["en"],
        xlab=f"{model['name']} - French",
        ylab=f"{model['name']} - English",
        ax=ax,
        xmin=1, xmax=18, ymin=1, ymax=18,
        layers=20,
        size_scale=60
    )

    # P-value annotation
    ax.text(12, 3, f"${p_adjusted[model['fr']]}$", fontsize=9)

    # Panel letter
    ax.text(-0.09, 1.06, letter, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom')

plt.savefig("../figures/Figure1_bubble.png", dpi=300, bbox_inches="tight")
plt.close(fig)