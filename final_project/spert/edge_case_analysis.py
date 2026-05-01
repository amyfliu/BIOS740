"""
Edge case analysis for SpERT predictions on ADKG and MDKG.
Compares gold labels vs SpERT predictions to find:
  1. Near-miss entities (off by 1 token)
  2. Missed entities (false negatives)
  3. Spurious entities (false positives)
  4. Confused relation types (wrong type but correct spans)
  5. Failed long-range relations (>10 tokens apart)

Usage:
    python edge_case_analysis.py
Outputs:
    - printed summary to terminal
    - logs/adkg_edge_cases.txt
    - logs/mdkg_edge_cases.txt
"""

import json
import os
from collections import Counter, defaultdict

os.makedirs("logs", exist_ok=True)

# ── file paths ────────────────────────────────────────────────────────────────
CONFIGS = {
    "ADKG": {
        "gold":  "data/datasets/adkg/adkg_test.json",
        "pred":  "data/log/adkg/adkg_test/2026-04-19_18:30:23.869365/predictions_test_epoch_0.json",
        "log":   "logs/adkg_edge_cases.txt",
    },
    "MDKG": {
        "gold":  "data/datasets/mdkg/mdkg_test.json",
        "pred":  "data/log/mdkg/mdkg_test/2026-04-19_19:42:43.624292/predictions_test_epoch_0.json",
        "log":   "logs/mdkg_edge_cases.txt",
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)


def ent_key(e):
    """Hashable key for an entity span+type."""
    return (e["start"], e["end"], e["type"])


def ent_span(e):
    """Hashable key for span only (ignores type)."""
    return (e["start"], e["end"])


def rel_key(r):
    """Hashable key for a relation (uses entity indices)."""
    return (r["head"], r["tail"], r["type"])


def rel_span_key(r):
    """Relation key ignoring type — used to detect type confusion."""
    return (r["head"], r["tail"])


def token_distance(doc_tokens, r):
    """Token distance between head and tail entity midpoints."""
    h_mid = (r["head_start"] + r["head_end"]) / 2
    t_mid = (r["tail_start"] + r["tail_end"]) / 2
    return abs(h_mid - t_mid)


def is_near_miss(g, p):
    """True if predicted span overlaps gold span and differs by <=1 token."""
    overlap = g["start"] < p["end"] and p["start"] < g["end"]
    if not overlap:
        return False
    start_diff = abs(g["start"] - p["start"])
    end_diff   = abs(g["end"]   - p["end"])
    return (start_diff + end_diff) <= 2 and g["type"] == p["type"]


# ── main analysis ─────────────────────────────────────────────────────────────

for dataset_name, cfg in CONFIGS.items():
    gold_docs = load(cfg["gold"])
    pred_docs = load(cfg["pred"])

    assert len(gold_docs) == len(pred_docs), \
        f"Mismatch: {len(gold_docs)} gold vs {len(pred_docs)} pred docs"

    # counters
    total_gold_ents   = 0
    total_pred_ents   = 0
    exact_ent_matches = 0
    near_misses       = []
    missed_ents       = []
    spurious_ents     = []

    total_gold_rels   = 0
    total_pred_rels   = 0
    exact_rel_matches = 0
    confused_rels     = []
    failed_long_range = []
    long_range_total  = 0
    long_range_correct= 0

    for gold, pred in zip(gold_docs, pred_docs):
        tokens = gold["tokens"]

        # ── entity analysis ───────────────────────────────────────────────
        gold_ents = gold["entities"]
        pred_ents = pred["entities"]

        total_gold_ents += len(gold_ents)
        total_pred_ents += len(pred_ents)

        gold_keys = {ent_key(e) for e in gold_ents}
        pred_keys = {ent_key(e) for e in pred_ents}

        exact_ent_matches += len(gold_keys & pred_keys)

        # find near misses and missed entities
        matched_gold = set()
        for g in gold_ents:
            if ent_key(g) in pred_keys:
                matched_gold.add(ent_key(g))
                continue
            # check near miss
            nm_found = False
            for p in pred_ents:
                if is_near_miss(g, p):
                    near_misses.append({
                        "text":       " ".join(tokens),
                        "gold_span":  tokens[g["start"]:g["end"]],
                        "pred_span":  tokens[p["start"]:p["end"]],
                        "type":       g["type"],
                        "gold_range": (g["start"], g["end"]),
                        "pred_range": (p["start"], p["end"]),
                    })
                    nm_found = True
                    matched_gold.add(ent_key(g))
                    break
            if not nm_found:
                missed_ents.append({
                    "text":      " ".join(tokens),
                    "span":      tokens[g["start"]:g["end"]],
                    "type":      g["type"],
                    "range":     (g["start"], g["end"]),
                })

        # spurious predictions
        for p in pred_ents:
            if ent_key(p) not in gold_keys:
                # check if it's a near miss (already counted)
                is_nm = any(is_near_miss(g, p) for g in gold_ents)
                if not is_nm:
                    spurious_ents.append({
                        "text":  " ".join(tokens),
                        "span":  tokens[p["start"]:p["end"]],
                        "type":  p["type"],
                        "range": (p["start"], p["end"]),
                    })

        # ── relation analysis ─────────────────────────────────────────────
        gold_rels = gold["relations"]
        pred_rels = pred["relations"]

        total_gold_rels += len(gold_rels)
        total_pred_rels += len(pred_rels)

        gold_rel_keys  = {rel_key(r) for r in gold_rels}
        pred_rel_spans = {rel_span_key(r): r["type"] for r in pred_rels}

        for r in gold_rels:
            span_k = rel_span_key(r)

            # compute token distance using entity spans
            h_ent = next((e for e in gold_ents
                          if gold_ents.index(e) == r["head"]), None)
            t_ent = next((e for e in gold_ents
                          if gold_ents.index(e) == r["tail"]), None)

            dist = None
            if h_ent and t_ent:
                dist = abs(((h_ent["start"] + h_ent["end"]) / 2) -
                           ((t_ent["start"] + t_ent["end"]) / 2))

            is_long_range = dist is not None and dist > 10

            if rel_key(r) in gold_rel_keys and rel_key(r) in \
                    {rel_key(p) for p in pred_rels}:
                exact_rel_matches += 1
                if is_long_range:
                    long_range_correct += 1

            elif span_k in pred_rel_spans and pred_rel_spans[span_k] != r["type"]:
                confused_rels.append({
                    "text":      " ".join(tokens),
                    "gold_type": r["type"],
                    "pred_type": pred_rel_spans[span_k],
                    "head":      tokens[gold_ents[r["head"]]["start"]:
                                        gold_ents[r["head"]]["end"]]
                                 if r["head"] < len(gold_ents) else "?",
                    "tail":      tokens[gold_ents[r["tail"]]["start"]:
                                        gold_ents[r["tail"]]["end"]]
                                 if r["tail"] < len(gold_ents) else "?",
                    "distance":  dist,
                })

            elif is_long_range and span_k not in pred_rel_spans:
                failed_long_range.append({
                    "text":      " ".join(tokens),
                    "gold_type": r["type"],
                    "head":      tokens[gold_ents[r["head"]]["start"]:
                                        gold_ents[r["head"]]["end"]]
                                 if r["head"] < len(gold_ents) else "?",
                    "tail":      tokens[gold_ents[r["tail"]]["start"]:
                                        gold_ents[r["tail"]]["end"]]
                                 if r["tail"] < len(gold_ents) else "?",
                    "distance":  round(dist, 1),
                })

            if is_long_range:
                long_range_total += 1

    # ── write log ─────────────────────────────────────────────────────────
    SEP  = "=" * 70
    SEP2 = "-" * 70

    with open(cfg["log"], "w") as f:
        def w(s=""): f.write(s + "\n")

        w(SEP)
        w(f"  EDGE CASE ANALYSIS — {dataset_name}")
        w(SEP)

        # entity summary
        w("\n[1] ENTITY SUMMARY")
        w(SEP2)
        w(f"  Gold entities:          {total_gold_ents}")
        w(f"  Predicted entities:     {total_pred_ents}")
        w(f"  Exact matches:          {exact_ent_matches}")
        w(f"  Near misses (±1 token): {len(near_misses)}")
        w(f"  Missed (false neg):     {len(missed_ents)}")
        w(f"  Spurious (false pos):   {len(spurious_ents)}")

        # near miss examples
        w(f"\n[2] NEAR-MISS ENTITIES (showing first 10)")
        w(SEP2)
        for ex in near_misses[:10]:
            w(f"  Type:  {ex['type']}")
            w(f"  Gold:  {' '.join(ex['gold_span'])}  {ex['gold_range']}")
            w(f"  Pred:  {' '.join(ex['pred_span'])}  {ex['pred_range']}")
            w(f"  Text:  {ex['text'][:100]}...")
            w()

        # missed entity type breakdown
        w(f"\n[3] MISSED ENTITIES — by type")
        w(SEP2)
        missed_by_type = Counter(e["type"] for e in missed_ents)
        for etype, count in missed_by_type.most_common():
            w(f"  {etype:<20} {count}")

        # relation summary
        w(f"\n[4] RELATION SUMMARY")
        w(SEP2)
        w(f"  Gold relations:         {total_gold_rels}")
        w(f"  Predicted relations:    {total_pred_rels}")
        w(f"  Exact matches:          {exact_rel_matches}")
        w(f"  Confused types:         {len(confused_rels)}")
        w(f"  Long-range total:       {long_range_total}")
        w(f"  Long-range correct:     {long_range_correct}")
        lr_pct = 100 * long_range_correct / long_range_total if long_range_total else 0
        w(f"  Long-range accuracy:    {lr_pct:.1f}%")
        w(f"  Failed long-range:      {len(failed_long_range)}")

        # confused relation examples
        w(f"\n[5] CONFUSED RELATION TYPES (showing first 10)")
        w(SEP2)
        for ex in confused_rels[:10]:
            dist_str = f"{ex['distance']:.1f} tokens" if ex["distance"] else "?"
            w(f"  Gold:  {ex['gold_type']}")
            w(f"  Pred:  {ex['pred_type']}")
            w(f"  Head:  {' '.join(ex['head'])}")
            w(f"  Tail:  {' '.join(ex['tail'])}")
            w(f"  Dist:  {dist_str}")
            w(f"  Text:  {ex['text'][:100]}...")
            w()

        # confusion matrix for relations
        w(f"\n[6] RELATION CONFUSION (gold → predicted)")
        w(SEP2)
        confusion = Counter((c["gold_type"], c["pred_type"]) for c in confused_rels)
        for (gold_t, pred_t), count in confusion.most_common(15):
            w(f"  {gold_t:<25} → {pred_t:<25} ({count})")

        # failed long-range examples
        w(f"\n[7] FAILED LONG-RANGE RELATIONS (showing first 10)")
        w(SEP2)
        for ex in failed_long_range[:10]:
            w(f"  Type:  {ex['gold_type']}")
            w(f"  Head:  {' '.join(ex['head'])}")
            w(f"  Tail:  {' '.join(ex['tail'])}")
            w(f"  Dist:  {ex['distance']} tokens")
            w(f"  Text:  {ex['text'][:100]}...")
            w()

    print(f"✅ {dataset_name} edge case log saved → {cfg['log']}")
    print(f"   Near misses: {len(near_misses)} | Confused rels: {len(confused_rels)} | Failed long-range: {len(failed_long_range)}")

print("\nDone! Check logs/adkg_edge_cases.txt and logs/mdkg_edge_cases.txt")
