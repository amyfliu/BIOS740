"""
Script to check for overlapping entities (entities whose token spans overlap)
and long-range relations (relation between distant entities) in ADKG and MDKG datasets.

"non-small cell lung cancer"
 └─ "lung cancer" (disease)
 └─ "non-small cell lung cancer" (disease)  ← overlapping!

 "The drug, which was approved last year after extensive trials, significantly reduced symptoms of Parkinson's disease."
 └─ "drug" (token 1)                                    "Parkinson's disease" (token 18) ← far apart!

Outputs:
  - printed summary of overlapping entity pairs and long-range relations for each dataset

How to use:
    python entity_overlap_long_range.py | tee logs/entity_overlap_long_range_log.txt
"""

import json, re

# ── Overlapping entity ─────────────────────────────────────────────────────────────────

def tokenize(text):
    return [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]

for fname, name in [('ADKG.json', 'ADKG'), ('MDKG.json', 'MDKG')]:
    with open(fname) as f:
        data = json.load(f)
    overlaps = 0
    total = 0
    for split in data.values():
        for doc in split:
            spans = [(e['start'], e['end']) for e in doc['entities']]
            for i in range(len(spans)):
                for j in range(i+1, len(spans)):
                    total += 1
                    s1, e1 = spans[i]
                    s2, e2 = spans[j]
                    if s1 < e2 and s2 < e1:
                        overlaps += 1
    print(f'{name}: {overlaps} overlapping entity pairs out of {total} pairs')


# ── Long-range Dependencies ─────────────────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r'\S+', text)

for fname, name in [('ADKG.json', 'ADKG'), ('MDKG.json', 'MDKG')]:
    with open(fname) as f:
        data = json.load(f)
    distances = []
    for split in data.values():
        for doc in split:
            tokens = tokenize(doc['text'])
            token_spans = [(m.start(), m.end()) for m in re.finditer(r'\S+', doc['text'])]
            ent_map = {}
            for e in doc['entities']:
                for i, (ts, te) in enumerate(token_spans):
                    if te > e['start'] and ts < e['end']:
                        ent_map[e['id']] = i
                        break
            for r in doc['relations']:
                h = ent_map.get(r['head']['id'])
                t = ent_map.get(r['tail']['id'])
                if h is not None and t is not None:
                    distances.append(abs(h - t))
    import numpy as np
    print(f'{name}: avg token distance={np.mean(distances):.1f}, '
          f'max={max(distances)}, >10 tokens: {sum(1 for d in distances if d > 10)} '
          f'({100*sum(1 for d in distances if d > 10)/len(distances):.1f}%)')
