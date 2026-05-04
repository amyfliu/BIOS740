"""
prepare_combined.py
Combines ADKG and MDKG converted SpERT datasets into a single training set
and generates a unified types file covering all entity and relation types.

Run from your spert/ directory:
    python prepare_combined.py

Outputs:
    data/datasets/combined/combined_train.json
    data/datasets/combined/combined_types.json
    data/datasets/combined/adkg_test.json   (symlinked/copied for eval convenience)
    data/datasets/combined/mdkg_test.json
"""

import json
import os
import shutil

OUT_DIR = "data/datasets/combined"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Combine train sets ────────────────────────────────────────────────
print("Step 1: Combining train sets...")

with open("data/datasets/adkg/adkg_train.json") as f:
    adkg_train = json.load(f)

with open("data/datasets/mdkg/mdkg_train.json") as f:
    mdkg_train = json.load(f)

combined_train = adkg_train + mdkg_train

out_path = os.path.join(OUT_DIR, "combined_train.json")
with open(out_path, "w") as f:
    json.dump(combined_train, f)

print(f"  ADKG train:    {len(adkg_train):,} sentences")
print(f"  MDKG train:    {len(mdkg_train):,} sentences")
print(f"  Combined:      {len(combined_train):,} sentences → {out_path}")


# ── Step 2: Combine dev sets ──────────────────────────────────────────────────
print("\nStep 2: Combining dev sets...")

with open("data/datasets/adkg/adkg_dev.json") as f:
    adkg_dev = json.load(f)

with open("data/datasets/mdkg/mdkg_dev.json") as f:
    mdkg_dev = json.load(f)

combined_dev = adkg_dev + mdkg_dev

out_path = os.path.join(OUT_DIR, "combined_dev.json")
with open(out_path, "w") as f:
    json.dump(combined_dev, f)

print(f"  ADKG dev:      {len(adkg_dev):,} sentences")
print(f"  MDKG dev:      {len(mdkg_dev):,} sentences")
print(f"  Combined:      {len(combined_dev):,} sentences → {out_path}")


# ── Step 3: Copy test sets for eval convenience ───────────────────────────────
print("\nStep 3: Copying test sets...")

for src, dst in [
    ("data/datasets/adkg/adkg_test.json",
     os.path.join(OUT_DIR, "adkg_test.json")),
    ("data/datasets/mdkg/mdkg_test.json",
     os.path.join(OUT_DIR, "mdkg_test.json")),
]:
    shutil.copy(src, dst)
    print(f"  Copied {src} → {dst}")


# ── Step 4: Generate combined types file ──────────────────────────────────────
print("\nStep 4: Generating combined types file...")

# Collect all types from both datasets
all_entity_types   = set()
all_relation_types = set()

for path in [
    "data/datasets/adkg/adkg_types.json",
    "data/datasets/mdkg/mdkg_types.json",
]:
    with open(path) as f:
        types = json.load(f)
    all_entity_types.update(types["entities"].keys())
    all_relation_types.update(types["relations"].keys())

# Build combined types dict
combined_types = {
    "entities": {
        t: {"short": t[:4].upper(), "verbose": t}
        for t in sorted(all_entity_types)
    },
    "relations": {
        t: {"short": t[:4].upper(), "verbose": t, "symmetric": False}
        for t in sorted(all_relation_types)
    }
}

out_path = os.path.join(OUT_DIR, "combined_types.json")
with open(out_path, "w") as f:
    json.dump(combined_types, f, indent=2)

print(f"  Entity types ({len(all_entity_types)}):   {sorted(all_entity_types)}")
print(f"  Relation types ({len(all_relation_types)}): {sorted(all_relation_types)}")
print(f"  Saved → {out_path}")

print("\n✅ Done! Ready to train with configs/combined_train.conf")
