"""Scores: official test set (primary) and held-out training concepts (secondary). Chance = 0.5%."""
import math

import numpy as np
import torch

from . import ubp as U


@torch.no_grad()
def evaluate(m, T):
    top1, top5 = U.retrieval(m, T.test_x, T.test_g, T.y)
    p = top1 / 100
    held = np.array([U.retrieval(m, x, g, T.y)[0] for x, g in zip(T.held_x, T.held_g)])
    return dict(test_top1=top1, test_top5=top5, test_se=100 * math.sqrt(p * (1 - p) / len(T.y)),
                test_clean_top1=U.retrieval(m, T.test_x, T.test_gc, T.y)[0],
                held_top1=float(held.mean()), held_se=float(held.std(ddof=1) / math.sqrt(len(held))))
