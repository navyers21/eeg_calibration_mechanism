"""Semantic categories that do NOT come from CLIP: the THINGS human-rated ("bottom-up") categories.

Source: data/things_concepts.tsv from the THINGS database (Hebart et al., 2019, CC-BY 4.0), as redistributed by
THINGSvision. About half of the 1,854 concepts have a rated category; uncategorized concepts are EXCLUDED from
category analyses (lumping them into an 'other' class would make a heterogeneous class). Multi-label entries take the
first label; fine labels are merged into coarse ones (fruit -> food, bird -> animal, ...). Categories with fewer than
`min_concepts` concepts in the image set are excluded too, so every class has enough concepts for grouped CV.
"""
import os

import numpy as np

TSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data',
                   'things_concepts.tsv')
MERGE = {'vegetable': 'food', 'fruit': 'food', 'dessert': 'food', 'drink': 'food', 'bird': 'animal',
         'insect': 'animal', 'kitchen tool': 'kitchen', 'kitchen appliance': 'kitchen', 'clothing accessory': 'clothing',
         'electronic device': 'electronic device', 'part of car': 'vehicle'}
_MAP = {}


def concept_of(path):
    """'train_images/00001_aardvark/aardvark_01b.jpg' -> 'aardvark'."""
    folder = str(path).replace('\\', '/').split('/')[-2]
    head, _, rest = folder.partition('_')
    return rest if head.isdigit() else folder


def concept_map():
    if not _MAP:
        import pandas as pd
        d = pd.read_csv(TSV, sep='\t')
        for cid, c in zip(d.uniqueID, d['Bottom-up Category (Human Raters)']):
            if isinstance(c, str) and c.strip():
                c = c.split(',')[0].strip()
                _MAP[cid] = MERGE.get(c, c)
    return _MAP


def labels(paths, min_concepts=5):
    """(labels, names): integer category per path, -1 = no usable category (exclude from category analyses)."""
    m = concept_map()
    concepts = [concept_of(p) for p in paths]
    cat = {c: m.get(c) for c in set(concepts)}
    counts = {}
    for k in cat.values():
        if k:
            counts[k] = counts.get(k, 0) + 1
    names = sorted(k for k, n in counts.items() if n >= min_concepts)
    return np.array([names.index(cat[c]) if cat[c] in names else -1 for c in concepts]), names
