"""Image-side labels for probing.

Low-level properties are computed directly from pixels. Semantic labels come from
clustering the frozen image-encoder features of the *training* images, so the probe
target does not depend on the EEG model.
"""
import numpy as np
from PIL import Image

LOW_LEVEL = ["luminance", "contrast", "saturation", "colorfulness", "high_sf_ratio", "edge_density"]


def _load(path, size=224):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def low_level_props(path):
    rgb = _load(path)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0)
    # Hasler & Suesstrunk colorfulness
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    # fraction of spectral energy above 1/4 of Nyquist
    F = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    h, w = F.shape
    yy, xx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2]
    rad = np.sqrt(yy ** 2 + xx ** 2) / (min(h, w) / 2)
    high = F[rad > 0.25].sum() / (F.sum() + 1e-8)
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)
    edges = float((grad > 0.05).mean())  # fixed absolute threshold so it varies across images
    return dict(luminance=float(lum.mean()), contrast=float(lum.std()), saturation=float(sat.mean()),
                colorfulness=float(colorful), high_sf_ratio=float(high), edge_density=edges)


def compute_low_level(paths):
    rows = [low_level_props(p) for p in paths]
    return np.array([[r[k] for k in LOW_LEVEL] for r in rows], dtype=np.float32)


def semantic_clusters(train_feats, eval_feats, k=20, seed=0):
    """KMeans on L2-normalized image-encoder features of TRAINING images; assign eval images.

    Caveat: encoder clusters are partly driven by low-level statistics too. Check the
    low-level/semantic correlation table the pipeline prints before interpreting.
    """
    from sklearn.cluster import KMeans
    nrm = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(nrm(train_feats))
    return km.predict(nrm(eval_feats)), km
