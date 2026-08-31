import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11})

NAVY = "#1a3c5e"
TEAL = "#1f6f78"
GOLD = "#b8862b"
GREY = "#e6e6e6"
DARK_GREY = "#333333"

# --- Chart A: RMSE comparison across the modelling stages (readable, non-zero-clipped) ---
labels = [
    "GLM Gamma\n(référence actuarielle)",
    "XGBoost\npar défaut",
    "XGBoost\n+ Optuna",
    "XGBoost\n+ GridSearchCV",
    "LightGBM\n+ Optuna",
    "+ Autoencodeur",
    "+ Autoencodeur\n+ NLP",
]
rmse = [27578.61, 28306.73, 25813.09, 26904.67, 25994.66, 25764.28, 25565.13]
colors = ["#6b4c9a", "#9e9e9e", "#1f77b4", "#7fb3d5", "#f4a340", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(labels, rmse, color=colors)
ax.set_ylim(24000, 29000)
ax.set_ylabel("RMSE (test)")
ax.set_title("RMSE du coût ultime selon la variante du pipeline")
for bar, val in zip(bars, rmse):
    ax.annotate(f"{val:,.0f}".replace(",", " "), (bar.get_x() + bar.get_width() / 2, val + 60),
                ha="center", fontsize=9.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "custom_rmse_comparison.png"), dpi=150)
plt.close()

# --- Chart B: marginal gain of each enrichment vs the tuned baseline (%), can be negative ---
variant_labels = ["Baseline\n(features tabulaires)", "+ Autoencodeur", "+ Autoencodeur + NLP"]
gains = [0.0, 0.19, 0.96]
bar_colors = ["#9e9e9e", "#f4a340", "#2ca02c"]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(variant_labels, gains, color=bar_colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Gain de RMSE vs baseline (%)")
ax.set_title("Contribution marginale de l'autoencodeur et du flux NLP")
ax.set_ylim(-0.1, 1.15)
for bar, val in zip(bars, gains):
    offset = 0.03
    ax.annotate(f"{val:+.2f}%", (bar.get_x() + bar.get_width() / 2, val + offset), ha="center", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "custom_ablation_gain.png"), dpi=150)
plt.close()

# --- Chart C: Autoencodeur vs Isolation Forest anomaly detection agreement + cost profile ---
groups = ["Aucune\nméthode\n(n=42 430)", "Autoencodeur\nseul (n=338)", "Isolation Forest\nseul (n=338)", "Les deux\nméthodes (n=94)"]
mean_costs = [10713, 25315, 23800, 45964]
bar_colors2 = ["#9e9e9e", "#2ca02c", "#f4a340", "#d62728"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(groups, mean_costs, color=bar_colors2)
ax.set_ylabel("Coût moyen du sinistre (train)")
ax.set_title("Coût moyen selon l'accord entre Autoencodeur et Isolation Forest")
for bar, val in zip(bars, mean_costs):
    ax.annotate(f"{val:,.0f}".replace(",", " "), (bar.get_x() + bar.get_width() / 2, val + 500), ha="center", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "custom_anomaly_cost.png"), dpi=150)
plt.close()

# --- Chart D: end-to-end pipeline architecture diagram ---


def box(ax, x, y, w, h, text, facecolor=NAVY, textcolor="white", fontsize=9.5, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=facecolor, edgecolor=DARK_GREY, linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, weight=weight, zorder=3, linespacing=1.4)


def arrow(ax, xy_from, xy_to, color=DARK_GREY, style="-|>", lw=1.3):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=14,
                                  color=color, linewidth=lw, zorder=1))


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

# Row 1: data & preprocessing
box(ax, 0.4, 6.6, 3.0, 1.0, "Données brutes\nKaggle - 54 000 sinistres\n(workers compensation)", facecolor=DARK_GREY)
box(ax, 4.1, 6.6, 3.4, 1.0, "Prétraitement\nCatégorie «Missing» + règles\nmétier + winsorization", facecolor=TEAL)
box(ax, 8.2, 6.6, 3.4, 1.0, "Feature engineering\nOne-hot, dates, split\ntrain/test (80/20)", facecolor=TEAL)
arrow(ax, (3.4, 7.1), (4.1, 7.1))
arrow(ax, (7.5, 7.1), (8.2, 7.1))

# connector down to 3 branches
arrow(ax, (9.9, 6.6), (9.9, 5.9))
arrow(ax, (9.9, 5.9), (2.0, 5.9))
arrow(ax, (9.9, 5.9), (5.95, 5.9))
arrow(ax, (2.0, 5.9), (2.0, 5.35))
arrow(ax, (5.95, 5.9), (5.95, 5.35))
arrow(ax, (9.9, 5.9), (9.9, 5.35))

# Row 2: three parallel flows
box(ax, 0.4, 3.9, 3.2, 1.45,
    "Flux 1 - ML supervisé\nXGBoost + Optuna (TPE)\nvs GridSearchCV, LightGBM\nCV répétée 5x5", facecolor=NAVY)
box(ax, 4.35, 3.9, 3.2, 1.45,
    "Flux 2 - DL non supervisé\nAutoencodeur (64-32-16-32-64)\nvs Isolation Forest\n-> score + flag anomalie", facecolor=NAVY)
box(ax, 8.3, 3.9, 3.3, 1.45,
    "Flux 3 - NLP / Agentique\nEmbeddings + zero-shot +\nAgent LLM (tool calling)\n-> gravité + escalade", facecolor=NAVY)

# converge to feature fusion
arrow(ax, (2.0, 3.9), (5.9, 3.15))
arrow(ax, (5.95, 3.9), (5.95, 3.15))
arrow(ax, (9.95, 3.9), (6.0, 3.15))

box(ax, 3.9, 2.5, 4.1, 0.65, "Fusion progressive des features (ablation mesurée)", facecolor=GOLD, textcolor="white", fontsize=9)
arrow(ax, (5.95, 2.5), (5.95, 1.95))

box(ax, 3.9, 1.3, 4.1, 0.65, "XGBoost final tuné (Optuna)", facecolor=NAVY, fontsize=10)
arrow(ax, (5.95, 1.3), (5.95, 0.75))

box(ax, 0.4, 0.1, 3.5, 0.65, "Coût ultime prédit\n+ intervalle (CV 5x5)", facecolor=TEAL, fontsize=8.8)
box(ax, 4.25, 0.1, 3.4, 0.65, "Audit SHAP / permutation\n/ ablation (importance des var.)", facecolor=TEAL, fontsize=8.8)
box(ax, 8.0, 0.1, 3.6, 0.65, "Flags anomalie + escalade\n(revue humaine ciblée)", facecolor=TEAL, fontsize=8.8)
arrow(ax, (4.7, 1.3), (2.15, 0.75))
arrow(ax, (5.95, 1.3), (5.95, 0.75))
arrow(ax, (7.2, 1.3), (9.8, 0.75))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "custom_architecture_diagram.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Custom charts written to", OUT_DIR)
