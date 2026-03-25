"""
OmniRAG — Plot Results
======================
Reads outputs/eval_results.json and saves 4 charts to outputs/plots/.

Usage:
    python scripts/plot_results.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Installing matplotlib…")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib",
                    "--break-system-packages"], check=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

RESULTS_PATH = Path("outputs/eval_results.json")
PLOTS_DIR    = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with open(RESULTS_PATH) as f:
    data = json.load(f)

modalities = [m for m, v in data.items() if "error" not in v]
if not modalities:
    print("No valid results found in eval_results.json")
    sys.exit(1)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "text":  "#38bdf8",   # sky
    "pdf":   "#fb923c",   # amber
    "image": "#a78bfa",   # violet
    "audio": "#34d399",   # emerald
    "video": "#fb7185",   # rose
}
palette = [COLORS.get(m, "#94a3b8") for m in modalities]

# Style
plt.rcParams.update({
    "figure.facecolor":  "#09090b",
    "axes.facecolor":    "#18181b",
    "axes.edgecolor":    "#3f3f46",
    "axes.labelcolor":   "#a1a1aa",
    "xtick.color":       "#71717a",
    "ytick.color":       "#71717a",
    "text.color":        "#e4e4e7",
    "grid.color":        "#27272a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})


def save(fig, name):
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Ingestion Time per Modality (bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
times = [data[m]["ingestion_time_s"] for m in modalities]
bars = ax.bar(modalities, times, color=palette, width=0.5, zorder=2)
ax.set_title("Ingestion Time per Modality (seconds)")
ax.set_ylabel("Seconds")
ax.grid(axis="y", zorder=1)
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=8, color="#e4e4e7")
fig.tight_layout()
save(fig, "1_ingestion_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Avg Query Latency per Modality (bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
latencies = [data[m].get("avg_latency", 0) for m in modalities]
bars = ax.bar(modalities, latencies, color=palette, width=0.5, zorder=2)
ax.set_title("Average Query Latency per Modality (seconds)")
ax.set_ylabel("Seconds")
ax.grid(axis="y", zorder=1)
for bar, val in zip(bars, latencies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}s", ha="center", va="bottom", fontsize=8, color="#e4e4e7")
fig.tight_layout()
save(fig, "2_query_latency.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Recall@10 and Faithfulness side-by-side (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
x      = np.arange(len(modalities))
width  = 0.32

recalls = [data[m].get("avg_recall", 0) for m in modalities]
faiths  = [data[m].get("avg_faithfulness", 0) for m in modalities]

b1 = ax.bar(x - width/2, recalls, width, label="Recall@10",    color=palette, alpha=0.9, zorder=2)
b2 = ax.bar(x + width/2, faiths,  width, label="Faithfulness", color=palette, alpha=0.45, zorder=2,
            edgecolor=palette, linewidth=1)

ax.set_title("Retrieval Quality: Recall@10 vs Faithfulness (ROUGE-L)")
ax.set_ylabel("Score (0–1)")
ax.set_xticks(x)
ax.set_xticklabels(modalities)
ax.set_ylim(0, 1.15)
ax.grid(axis="y", zorder=1)
ax.legend(facecolor="#27272a", edgecolor="#3f3f46")

for bar, val in zip(b1, recalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color="#e4e4e7")
for bar, val in zip(b2, faiths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="#e4e4e7")
fig.tight_layout()
save(fig, "3_recall_faithfulness.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — Chunk counts per modality (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
chunks = [data[m].get("chunk_count", 0) for m in modalities]
bars   = ax.barh(modalities, chunks, color=palette, height=0.45, zorder=2)
ax.set_title("Indexed Chunk Count per Modality")
ax.set_xlabel("Number of chunks")
ax.grid(axis="x", zorder=1)
for bar, val in zip(bars, chunks):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8, color="#e4e4e7")
ax.invert_yaxis()
fig.tight_layout()
save(fig, "4_chunk_counts.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print synopsis-ready table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  SYNOPSIS TABLE  (copy-paste ready)")
print("="*72)
print(f"{'Modality':<10} | {'Ingest(s)':<10} | {'Chunks':<8} | "
      f"{'Recall@10':<10} | {'Faithfulness':<13} | {'Latency(s)'}")
print("-"*72)
for m in modalities:
    r = data[m]
    print(f"{m:<10} | "
          f"{r.get('ingestion_time_s','?'):<10} | "
          f"{r.get('chunk_count','?'):<8} | "
          f"{r.get('avg_recall','?'):<10} | "
          f"{r.get('avg_faithfulness','?'):<13} | "
          f"{r.get('avg_latency','?')}")
print("="*72)
print(f"\nAll charts saved to: outputs/plots/\n")