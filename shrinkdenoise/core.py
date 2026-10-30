"""Public API: salt-and-pepper noise, classical median/CFM, and the shrinkage estimators E1-E6 of the paper.

Pipeline of the paper: p = 15 predictors, window R = 9, kappa = 3 (E6), lambda = 30 (E4); the initial estimate E0 is the
noise-free median (base 'M') or the noise-free CFM (base 'C') of the clean pixels around the pixel.
"""
from __future__ import annotations
import numpy as np
from ._kernels import clean_median_expanding, clean_cfm_expanding
from ._solvers import build_stats, solve, solve_ebridge

BASES = {"M": 0, "C": 1}
ESTIMATORS = {
    "E1": ("LS-LP",  "Least squares, no shrinkage",                      lambda S: solve(S, mode="ols")[0]),
    "E2": ("JS-LP",  "James-Stein shrinkage",                            lambda S: solve(S, mode="js")[0]),
    "E3": ("GJS-LP", "Grouped James-Stein (3 groups)",                   lambda S: solve(S, mode="gjs", ng=3)[0]),
    "E4": ("RR-LP",  "Ridge, fixed penalty (lambda = 30)",               lambda S: solve(S, mode="ridge", lam=30)[0]),
    "E5": ("EB-LP",  "Empirical-Bayes ridge (kappa = 1)",                lambda S: solve_ebridge(S, rule="hkb", kappa=1.0)[0]),
    "E6": ("EBR-LP", "Empirical-Bayes ridge (kappa = 3)  [best in paper]", lambda S: solve_ebridge(S, rule="hkb", kappa=3.0)[0]),
}
R_WINDOW = 9          # local window radius used in the paper
MARGIN = 24           # tile overlap: > the maximal dependency radius (9 + 10 + 1) of the pipeline, so tiling is exact
_W = (0.298936021293776, 0.587043074451121, 0.114020904255103)   # MATLAB rgb2gray weights (as in the paper)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> uint8 gray with the weights of MATLAB rgb2gray."""
    if rgb.ndim == 2:
        return rgb
    g = rgb[..., 0] * _W[0] + rgb[..., 1] * _W[1] + rgb[..., 2] * _W[2]
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, density: float, seed: int | None = 0) -> np.ndarray:
    """Each pixel (each channel for colour) is hit with probability `density`; salt/pepper equiprobable."""
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    out[r < density / 2] = 0
    out[(r >= density / 2) & (r < density)] = 255
    return out


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def noise_fraction(img: np.ndarray) -> float:
    """Fraction of pixels with value 0 or 255 (blind detection used by all selective methods)."""
    return float(np.mean((img == 0) | (img == 255)))


def classical(noisy: np.ndarray, base: str = "M") -> np.ndarray:
    """The classical algorithm that supplies E0: noise-free median (M) or noise-free CFM (C), on a gray image."""
    f = clean_median_expanding if base == "M" else clean_cfm_expanding
    return f(np.ascontiguousarray(noisy))


def _denoise_gray(noisy, base, estimator, progress=None, tile=256):
    if noisy.ndim != 2 or noisy.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 image")
    solver = ESTIMATORS[estimator][2]
    b = BASES[base]
    h, w = noisy.shape
    out = noisy.copy()
    ys = list(range(0, h, tile)); xs = list(range(0, w, tile)); total = len(ys) * len(xs); done = 0
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ya, yb = max(y0 - MARGIN, 0), min(y1 + MARGIN, h)
            xa, xb = max(x0 - MARGIN, 0), min(x1 + MARGIN, w)
            sub = np.ascontiguousarray(noisy[ya:yb, xa:xb])
            if ((sub == 0) | (sub == 255)).any():
                S = build_stats(sub, R=R_WINDOW, rf=1, use_cands=True, base=b)
                res = solver(S)
                out[y0:y1, x0:x1] = res[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
            done += 1
            if progress:
                progress(done / total)
    return out


def denoise(noisy: np.ndarray, base: str = "M", estimator: str = "E6", progress=None, tile: int = 256) -> np.ndarray:
    """Remove salt-and-pepper noise with estimator E1..E6 on base 'M' (median) or 'C' (CFM).

    `noisy`: uint8 array, gray (H, W) or colour (H, W, 3); colour images are processed channel by channel
    (an extension, not evaluated in the paper).  Noise is detected blindly as the values 0 / 255.
    """
    if base not in BASES:
        raise ValueError("base must be 'M' or 'C'")
    if estimator not in ESTIMATORS:
        raise ValueError("estimator must be one of " + ", ".join(ESTIMATORS))
    if noisy.ndim == 2:
        return _denoise_gray(noisy, base, estimator, progress, tile)
    chans = []
    for c in range(noisy.shape[2]):
        cb = None if progress is None else (lambda p, c=c: progress((c + p) / noisy.shape[2]))
        chans.append(_denoise_gray(np.ascontiguousarray(noisy[..., c]), base, estimator, cb, tile))
    return np.stack(chans, -1)
