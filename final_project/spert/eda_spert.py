"""
EDA script for ADKG and MDKG biomedical datasets.
Outputs:
  - printed dataset overview table
  - figures/fig1_entity_distributions.png
  - figures/fig2_relation_distributions.png
  - figures/fig3_top_relation_pairs.png
  - figures/fig4_density_comparison.png

How to use:
    python eda_spert.py | tee logs/eda_log.txt
Make sure ADKG.json and MDKG.json are in the same folder as this script,
or update ADKG_PATH / MDKG_PATH below.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────────────
ADKG_PATH = "ADKG.json"
MDKG_PATH = "MDKG.json"
OUT_DIR   = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e5e5e5",
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

ADKG_COLOR = "#378ADD"
MDKG_COLOR = "#1D9E75"

PALETTE = [
    "#378ADD", "#1D9E75", "#D85A30", "#D4537E",
    "#BA7517", "#7F77DD", "#639922", "#E24B4A", "#888780",
]


# ── helpers ─────────────────────────────────────────────────────────────────

def load_all(path):
    with open(path) as f:
        raw = json.load(f)
    docs = []
    for split in raw.values():
        docs.extend(split)
    return raw, docs


def compute_stats(raw, docs):
    splits = {k: v for k, v in raw.items()}
    ent_counts  = Counter()
    rel_counts  = Counter()
    pair_counts = Counter()

    ents_per_sent = []
    rels_per_sent = []
    span_chars    = []

    for doc in docs:
        ents_per_sent.append(len(doc["entities"]))
        rels_per_sent.append(len(doc["relations"]))
        for e in doc["entities"]:
            ent_counts[e["type"]] += 1
            span_chars.append(e["end"] - e["start"])
        for r in doc["relations"]:
            rel_counts[r["type"]] += 1
            pair_counts[f"{r['head']['type']} → {r['tail']['type']}"] += 1

    return {
        "splits":          {k: len(v) for k, v in splits.items()},
        "total_sents":     len(docs),
        "total_entities":  sum(ent_counts.values()),
        "total_relations": sum(rel_counts.values()),
        "sents_with_rels": sum(1 for d in docs if d["relations"]),
        "sents_no_ents":   sum(1 for d in docs if not d["entities"]),
        "avg_ents":        np.mean(ents_per_sent),
        "avg_rels":        np.mean(rels_per_sent),
        "avg_span_chars":  np.mean(span_chars),
        "max_span_chars":  max(span_chars),
        "ent_counts":      ent_counts,
        "rel_counts":      rel_counts,
        "pair_counts":     pair_counts,
    }


# ── load ─────────────────────────────────────────────────────────────────────
print("Loading data ...")
adkg_raw, adkg_docs = load_all(ADKG_PATH)
mdkg_raw, mdkg_docs = load_all(MDKG_PATH)

adkg = compute_stats(adkg_raw, adkg_docs)
mdkg = compute_stats(mdkg_raw, mdkg_docs)


# ── 1. Dataset overview (printed) ────────────────────────────────────────────
SEP = "-" * 62
print(f"\n{'DATASET OVERVIEW':^62}")
print(SEP)
print(f"{'Metric':<35} {'ADKG':>12} {'MDKG':>12}")
print(SEP)

rows = [
    ("Total sentences",          adkg["total_sents"],     mdkg["total_sents"]),
    ("  train",                  adkg["splits"]["train"], mdkg["splits"]["train"]),
    ("  dev",                    adkg["splits"]["dev"],   mdkg["splits"]["dev"]),
    ("  test",                   adkg["splits"]["test"],  mdkg["splits"]["test"]),
    ("Total entities",           adkg["total_entities"],  mdkg["total_entities"]),
    ("Total relations",          adkg["total_relations"], mdkg["total_relations"]),
    ("Sentences with relations", adkg["sents_with_rels"], mdkg["sents_with_rels"]),
    ("Sentences with no entities", adkg["sents_no_ents"], mdkg["sents_no_ents"]),
    ("Avg entities / sentence",  f"{adkg['avg_ents']:.2f}", f"{mdkg['avg_ents']:.2f}"),
    ("Avg relations / sentence", f"{adkg['avg_rels']:.2f}", f"{mdkg['avg_rels']:.2f}"),
    ("Avg entity span (chars)",  f"{adkg['avg_span_chars']:.1f}", f"{mdkg['avg_span_chars']:.1f}"),
    ("Max entity span (chars)",  adkg["max_span_chars"],  mdkg["max_span_chars"]),
    ("Entity type count",        len(adkg["ent_counts"]), len(mdkg["ent_counts"])),
    ("Relation type count",      len(adkg["rel_counts"]), len(mdkg["rel_counts"])),
]

for label, a, m in rows:
    print(f"{label:<35} {str(a):>12} {str(m):>12}")
print(SEP)


# ── 2. Figure 1 — entity distributions ───────────────────────────────────────
def bar_chart(ax, counts, color, title, top_n=None):
    items = counts.most_common(top_n) if top_n else sorted(counts.items(), key=lambda x: -x[1])
    labels, values = zip(*items)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Count")
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="y", labelsize=10)


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_chart(axes[0], adkg["ent_counts"], ADKG_COLOR, "ADKG — entity type distribution")
bar_chart(axes[1], mdkg["ent_counts"], MDKG_COLOR, "MDKG — entity type distribution")
fig.suptitle("Entity Type Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig1_entity_distributions.png")
plt.savefig(path)
plt.close()
print(f"\nSaved → {path}")


# ── 3. Figure 2 — relation distributions ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bar_chart(axes[0], adkg["rel_counts"], ADKG_COLOR, "ADKG — relation type distribution")
bar_chart(axes[1], mdkg["rel_counts"], MDKG_COLOR, "MDKG — relation type distribution")
fig.suptitle("Relation Type Distributions", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig2_relation_distributions.png")
plt.savefig(path)
plt.close()
print(f"Saved → {path}")


# ── 4. Figure 3 — top relation entity pairs ───────────────────────────────────
def pair_chart(ax, pair_counts, title, top_n=10):
    items = pair_counts.most_common(top_n)
    labels, values = zip(*items)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Count")
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="y", labelsize=10)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
pair_chart(axes[0], adkg["pair_counts"], "ADKG — top 10 relation entity pairs")
pair_chart(axes[1], mdkg["pair_counts"], "MDKG — top 10 relation entity pairs")
fig.suptitle("Top Relation Entity-Type Pairs (Head → Tail)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig3_top_relation_pairs.png")
plt.savefig(path)
plt.close()
print(f"Saved → {path}")


# ── 5. Figure 4 — density comparison ─────────────────────────────────────────
metrics = ["Avg entities\n/ sentence", "Avg relations\n/ sentence"]
adkg_vals = [adkg["avg_ents"], adkg["avg_rels"]]
mdkg_vals = [mdkg["avg_ents"], mdkg["avg_rels"]]

x = np.arange(len(metrics))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(x - width / 2, adkg_vals, width, label="ADKG", color=ADKG_COLOR, edgecolor="white")
bars2 = ax.bar(x + width / 2, mdkg_vals, width, label="MDKG", color=MDKG_COLOR, edgecolor="white")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.2f}", ha="center", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.2f}", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Average count per sentence")
ax.set_title("Entity and Relation Density: ADKG vs MDKG",
             fontsize=13, fontweight="bold", pad=10)
ax.legend(fontsize=11)
ax.set_ylim(0, max(mdkg_vals) * 1.25)
plt.tight_layout()
path = os.path.join(OUT_DIR, "fig4_density_comparison.png")
plt.savefig(path)
plt.close()
print(f"Saved → {path}")

print(f"\n All figures saved to ./{OUT_DIR}/")
