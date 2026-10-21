#!/usr/bin/env python3
"""Progress-update figures, read straight from the cafaeval leaderboard files.

fig1_methods.png  average IA-weighted F-max per method (NK/LK/PK mean)
fig2_layers.png   ESM C score by layer, 300M vs 600M, ProtT5 as reference
Run from rotation2/: python runs/human/figures/make_figures.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

OUT = "runs/human/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def scores(path):
    d = pd.read_csv(path, sep="\t")
    return dict(zip(d["filename"].str.replace(".tsv", "", regex=False), d["final_score"]))


main = scores("runs/human/eval/leaderboard_score.tsv")
sweep = scores("runs/human/sweep/eval/leaderboard_score.tsv")
layers = scores("runs/human/layers/eval/leaderboard_score.tsv")
prott5_best = max(v for k, v in sweep.items() if k.startswith("prot_t5"))


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# --- Figure 1: methods ---------------------------------------------------
rows = [  # label, score, is_esmc
    ("BLAST", main["blast"], False),
    ("Naive", main["naive"], False),
    ("ESM C 300M, last layer", main["esmc_300m_euclidean_k3"], True),
    ("ProtT5 (best setting)", prott5_best, False),
    ("ESM C 300M, layer 25", layers["esmc_300m_L25_euclidean_k3"], True),
    ("ESM C 600M, layer 24", layers["esmc_600m_L24_euclidean_k3"], True),
]
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
style(ax)
y = range(len(rows))
ax.barh(y, [r[1] for r in rows], height=0.6, color=[BLUE if r[2] else GRAY for r in rows],
        edgecolor=SURFACE, linewidth=2)
for i, (label, v, _) in enumerate(rows):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=11, color=INK)
ax.set_yticks(list(y), [r[0] for r in rows], fontsize=11, color=INK)
ax.set_xlim(0, 0.25)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_xlabel("Average IA-weighted F-max (NK, LK, PK)")
fig.suptitle("ESM C middle layers match ProtT5; both beat BLAST and Naive",
             x=0.02, ha="left", fontsize=13, fontweight="bold")
fig.text(0.02, 0.005, "Human-only CAFA-style test set (t0 May 2025 → t1 July 2026). "
         "ESM C layers were chosen on the test set.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/fig1_methods.png", dpi=200, facecolor=SURFACE)

# --- Figure 2: layers ----------------------------------------------------
series = [  # label, color, n_blocks, middle layers
    ("ESM C 300M", BLUE, 30, [10, 15, 20, 25], "esmc_300m"),
    ("ESM C 600M", ORANGE, 36, [12, 18, 24, 30], "esmc_600m"),
]
fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
style(ax)
ax.axhline(prott5_best, color=INK2, linestyle=(0, (4, 3)), linewidth=1.5)
ax.text(10, prott5_best + 0.002, f"ProtT5 best ({prott5_best:.3f})", fontsize=10, color=INK2)
for label, color, n, mids, key in series:
    xs = mids + [n]
    ys = [layers[f"{key}_L{m}_euclidean_k3"] for m in mids] + [layers[f"{key}_euclidean_k3"]]
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.annotate(f"last layer\n{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(10, -8),
                textcoords="offset points", fontsize=9, color=INK2)
    best = max(range(len(ys) - 1), key=lambda i: ys[i])
    ax.annotate(f"{ys[best]:.3f}", (xs[best], ys[best]), xytext=(-12, 9),
                textcoords="offset points", fontsize=10, color=INK)
ax.set_xticks([10, 12, 15, 18, 20, 24, 25, 30, 36])
ax.set_xlim(8, 40)
ax.margins(y=0.15)
ax.yaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.set_xlabel("Transformer layer used for the embedding")
ax.set_ylabel("Average IA-weighted F-max")
ax.legend(frameon=False, loc="lower left", fontsize=10)
fig.suptitle("The layer matters more than model size", x=0.02, ha="left",
             fontsize=13, fontweight="bold")
fig.text(0.02, 0.005, "Euclidean distance, k = 3. 'Last layer' = final normalized output, "
         "used in the default pipeline.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/fig2_layers.png", dpi=200, facecolor=SURFACE)
print("saved", f"{OUT}/fig1_methods.png", f"{OUT}/fig2_layers.png")
