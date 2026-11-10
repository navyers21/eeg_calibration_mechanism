"""LEACE concept erasure (Belrose et al., 2023) plus a matched random-subspace control.

The eraser removes, in whitened coordinates, the subspace spanned by the cross-covariance
between activations X and concept labels Z. After erasure no linear predictor can recover
Z from X better than a constant (on the fitting distribution).

The control removes a random subspace of the SAME rank in the SAME whitened space, so any
damage it causes is the baseline cost of "deleting r dimensions" from this layer.
"""
import torch


def _labels_to_Z(z):
    z = torch.as_tensor(z)
    if z.dtype in (torch.int32, torch.int64, torch.bool):
        z = torch.nn.functional.one_hot(z.long()).float()
    elif z.ndim == 1:
        z = z.float()[:, None]
    z = z.float()
    return (z - z.mean(0)) / (z.std(0) + 1e-6)


class Eraser:
    def __init__(self, mean, W, W_pinv, P, support=None):
        self.mean, self.W, self.W_pinv, self.P = mean, W, W_pinv, P
        self.support = support  # (D, k) eigenvectors with non-negligible variance
        self.scale = 1.0        # >1 or <1 rescales the edit (used to norm-match random controls)

    @property
    def rank(self):
        return int(round(self.P.trace().item()))

    def __call__(self, x):
        dev = x.device
        mu, W, Wp, P = (t.to(dev) for t in (self.mean, self.W, self.W_pinv, self.P))
        return x - (self.scale * ((x.float() - mu) @ W.T @ P.T @ Wp.T)).to(x.dtype)

    def random_control(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        r = self.rank
        B = self.support if self.support is not None else torch.eye(self.P.shape[0])
        # random r-dim subspace INSIDE the data's support, so the control is comparable when n < D
        Q, _ = torch.linalg.qr(B @ torch.randn(B.shape[1], r, generator=g))
        return Eraser(self.mean, self.W, self.W_pinv, Q @ Q.T, self.support)


def fit_leace(X, z, shrinkage=1e-2, rank_tol=1e-6, support_tol=1e-3):
    """X: (n, D) float tensor. z: (n,) int labels, (n,) float, or (n, k) float."""
    X = torch.as_tensor(X).double()
    Z = _labels_to_Z(z).double()
    n, D = X.shape
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / n
    S = S + shrinkage * S.trace() / D * torch.eye(D, dtype=S.dtype)
    evals, evecs = torch.linalg.eigh(S)
    support = evecs[:, evals > support_tol * evals.max()]
    evals = evals.clamp_min(1e-12)
    W = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    W_pinv = evecs @ torch.diag(evals.sqrt()) @ evecs.T
    Sxz = Xc.T @ Z / n
    U, s, _ = torch.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, s > rank_tol * s.max()]
    P = U @ U.T
    return Eraser(mu.float(), W.float(), W_pinv.float(), P.float(), support.float())
