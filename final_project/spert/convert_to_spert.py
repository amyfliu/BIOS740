import json
import re
import os


# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text):
    """Whitespace-tokenize text and return (tokens, token_spans)."""
    tokens = []
    spans = []
    for m in re.finditer(r'\S+', text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def char_to_tok(spans, char_start, char_end):
    """Map character offsets to token indices (end-exclusive)."""
    tok_start, tok_end = None, None
    for i, (ts, te) in enumerate(spans):
        if te > char_start and ts < char_end:
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
    return tok_start, tok_end


def convert_split(docs):
    """Convert a list of raw documents to SpERT format."""
    spert_docs = []
    skipped_entities = 0
    skipped_relations = 0

    for doc in docs:
        text = doc['text']
        tokens, spans = tokenize(text)

        # ── convert entities ──────────────────────────────────────────────
        spert_entities = []
        entity_id_to_index = {}   # "T1" -> 0, "T2" -> 1, ...

        for ent in doc['entities']:
            tok_start, tok_end = char_to_tok(spans, ent['start'], ent['end'])

            if tok_start is None:
                print(f"  WARNING: could not map entity '{ent['text']}' "
                      f"in sent '{doc['sent_id']}' — skipping")
                skipped_entities += 1
                continue

            spert_entities.append({
                'type':  ent['type'],
                'start': tok_start,
                'end':   tok_end
            })
            entity_id_to_index[ent['id']] = len(spert_entities) - 1

        # ── convert relations ─────────────────────────────────────────────
        spert_relations = []
        for rel in doc.get('relations', []):
            head_id = rel['head']['id']
            tail_id = rel['tail']['id']

            if head_id not in entity_id_to_index or tail_id not in entity_id_to_index:
                print(f"  WARNING: relation references missing entity "
                      f"in sent '{doc['sent_id']}' — skipping")
                skipped_relations += 1
                continue

            spert_relations.append({
                'type': rel['type'],
                'head': entity_id_to_index[head_id],
                'tail': entity_id_to_index[tail_id]
            })

        spert_docs.append({
            'tokens':    tokens,
            'entities':  spert_entities,
            'relations': spert_relations
        })

    return spert_docs, skipped_entities, skipped_relations


def generate_types(data_splits, dataset_name):
    """Collect all entity and relation types and build types.json."""
    entity_types = set()
    relation_types = set()

    for split in data_splits.values():
        for doc in split:
            for ent in doc['entities']:
                entity_types.add(ent['type'])
            for rel in doc.get('relations', []):
                relation_types.add(rel['type'])

    types = {
        'entities': {
            t: {'short': t[:4].upper(), 'verbose': t}
            for t in sorted(entity_types)
        },
        'relations': {
            t: {'short': t[:4].upper(), 'verbose': t, 'symmetric': False}
            for t in sorted(relation_types)
        }
    }

    print(f"\n{dataset_name} entity types:   {sorted(entity_types)}")
    print(f"{dataset_name} relation types: {sorted(relation_types)}")
    return types


def convert_dataset(input_path, output_dir, dataset_name):
    """Full pipeline: load → convert all splits → save → generate types."""
    print(f"\n{'='*60}")
    print(f"Converting {dataset_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path) as f:
        raw = json.load(f)

    total_skipped_ents = 0
    total_skipped_rels = 0

    for split_name, docs in raw.items():
        print(f"\n[{split_name}] {len(docs)} sentences")
        converted, skipped_e, skipped_r = convert_split(docs)
        total_skipped_ents += skipped_e
        total_skipped_rels += skipped_r

        out_path = os.path.join(output_dir, f'{dataset_name}_{split_name}.json')
        with open(out_path, 'w') as f:
            json.dump(converted, f, indent=2)
        print(f"  Saved {len(converted)} docs → {out_path}")

    # generate types.json
    types = generate_types(raw, dataset_name)
    types_path = os.path.join(output_dir, f'{dataset_name}_types.json')
    with open(types_path, 'w') as f:
        json.dump(types, f, indent=2)
    print(f"\n  Types saved → {types_path}")

    print(f"\n  Total skipped entities:  {total_skipped_ents}")
    print(f"  Total skipped relations: {total_skipped_rels}")


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── ADKG ──
    convert_dataset(
        input_path='ADKG.json',
        output_dir='data/datasets/adkg',
        dataset_name='adkg'
    )

    # ── MDKG ──
    convert_dataset(
        input_path='MDKG.json',
        output_dir='data/datasets/mdkg',
        dataset_name='mdkg'
    )

    print("\n✅ All done! Files saved to data/datasets/")
