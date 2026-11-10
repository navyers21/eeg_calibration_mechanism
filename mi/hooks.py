"""Capture and edit intermediate activations with forward hooks.

Every hooked output is viewed as (B, N, D): batch, positions (tokens/time/channels), features.
`feat_axis` says which axis of the module output holds the feature dimension D.
"""
import contextlib
import torch


def _as_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def to_bnd(t, feat_axis=-1):
    t = torch.movedim(t, feat_axis, -1)
    B, D = t.shape[0], t.shape[-1]
    return t.reshape(B, -1, D)


def from_bnd(x, like, feat_axis=-1):
    moved_shape = torch.movedim(like, feat_axis, -1).shape
    return torch.movedim(x.reshape(moved_shape), -1, feat_axis)


@contextlib.contextmanager
def capture(model, layers, feat_axes=None, keep_tokens=False):
    """layers: {name: module}. Yields dict name -> list of captured tensors (CPU).

    Pooled (B, D) by default; (B, N, D) if keep_tokens.
    """
    feat_axes = feat_axes or {}
    store = {n: [] for n in layers}
    handles = []
    for name, mod in layers.items():
        ax = feat_axes.get(name, -1)

        def hook(m, i, o, name=name, ax=ax):
            x = to_bnd(_as_tensor(o).detach().float(), ax)
            store[name].append((x if keep_tokens else x.mean(1)).cpu())
        handles.append(mod.register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def edit(module, fn, feat_axis=-1):
    """Replace module output with fn applied to its (B, N, D) view."""
    def hook(m, i, o):
        t = _as_tensor(o)
        new = from_bnd(fn(to_bnd(t, feat_axis)).to(t.dtype), t, feat_axis)
        if isinstance(o, tuple):
            return (new,) + tuple(o[1:])
        if isinstance(o, list):
            return [new] + list(o[1:])
        return new
    h = module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
