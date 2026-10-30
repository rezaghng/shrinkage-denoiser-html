"""Numba kernels of the paper: predictor bank and the classical initial estimates.

Copied verbatim from the research code of the paper (stack_filter.py, baselines.py) so that results are identical.
base = 0 : E0 = noise-free median;  base = 1 : E0 = noise-free CFM (K=150, C=1.5, Chebyshev distance).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _cfm_clean(img, clean, y, x, r):
    """Certainty-based fuzzy median (K=150, C=1.5, Chebyshev distance) of the CLEAN pixels of the
    (2r+1)^2 window centred at (y, x), the centre itself excluded (leave-one-out).
    Returns the value of the first candidate (row-major scan) with the maximal certainty score."""
    h, w = img.shape
    ys = []; xs = []; vs = []
    for yy in range(max(y - r, 0), min(h, y + r + 1)):
        for xx in range(max(x - r, 0), min(w, x + r + 1)):
            if (yy != y or xx != x) and clean[yy, xx]:
                ys.append(yy); xs.append(xx); vs.append(img[yy, xx])
    best = -1.0; bv = 0.0
    n = len(vs)
    for i in range(n):
        sc = 0.0
        for j in range(n):
            d = abs(vs[j] - vs[i])
            cb = max(abs(ys[j] - ys[i]), abs(xs[j] - xs[i]))
            sc += (150.0 / (150.0 + d)) * (1.5 / (1.5 + cb))
        if sc > best:
            best = sc; bv = vs[i]
    return bv


@njit(cache=True)
def _features(img, clean, rmax, need, base=0):
    """Predictor bank.  `base` selects the initial estimate E0 (column 0) and the 'other' candidate (column 7):
         base = 0 : E0 = noise-free MEDIAN (expanding window), column 7 = CFM of the clean pixels (r=2)   [original]
         base = 1 : E0 = noise-free CFM    (expanding window), column 7 = median of the clean pixels (r=2)
    Both E0 variants are functions of the clean pixels around x only, x itself excluded (leave-one-out)."""
    h, w = img.shape
    P = 8
    F = np.zeros((h, w, P))
    for y in range(h):
        for x in range(w):
            # E0: expanding noise-free median, centre excluded
            e0 = -1.0
            for r in range(1, rmax + 1):
                vals = []
                for yy in range(max(y - r, 0), min(h, y + r + 1)):
                    for xx in range(max(x - r, 0), min(w, x + r + 1)):
                        if (yy != y or xx != x) and clean[yy, xx]:
                            vals.append(img[yy, xx])
                if len(vals) >= need:
                    if base == 0:
                        a = np.array(vals)
                        e0 = np.median(a)
                    else:
                        e0 = _cfm_clean(img, clean, y, x, r)
                    break
            if e0 < 0:                       # no support at all: fall back to wide mean of clean
                s = 0.0; n = 0
                for yy in range(max(y - 10, 0), min(h, y + 11)):
                    for xx in range(max(x - 10, 0), min(w, x + 11)):
                        if clean[yy, xx]:
                            s += img[yy, xx]; n += 1
                e0 = s / n if n > 0 else 128.0
            F[y, x, 0] = e0
            # E1 gaussian clean mean (radius 3, sigma 1.2)
            sw = 0.0; sv = 0.0
            for yy in range(max(y - 3, 0), min(h, y + 4)):
                for xx in range(max(x - 3, 0), min(w, x + 4)):
                    if (yy != y or xx != x) and clean[yy, xx]:
                        g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.2 ** 2))
                        sw += g; sv += g * img[yy, xx]
            F[y, x, 1] = sv / sw if sw > 0 else e0
            # E2..E5 directional medians (radius 3)
            for d in range(4):
                dy = 0; dx = 1
                if d == 1: dy = 1; dx = 0
                if d == 2: dy = 1; dx = 1
                if d == 3: dy = 1; dx = -1
                vals = []
                for s in range(-3, 4):
                    if s == 0: continue
                    yy = y + s * dy; xx = x + s * dx
                    if yy >= 0 and yy < h and xx >= 0 and xx < w and clean[yy, xx]:
                        vals.append(img[yy, xx])
                F[y, x, 2 + d] = np.median(np.array(vals)) if len(vals) >= 2 else e0
            # E6 median-guided bilateral mean (radius 3, range sigma 12)
            sw = 0.0; sv = 0.0
            for yy in range(max(y - 3, 0), min(h, y + 4)):
                for xx in range(max(x - 3, 0), min(w, x + 4)):
                    if (yy != y or xx != x) and clean[yy, xx]:
                        g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2) - (img[yy, xx] - e0) ** 2 / (2 * 12.0 ** 2))
                        sw += g; sv += g * img[yy, xx]
            F[y, x, 6] = sv / sw if sw > 1e-9 else e0
            # E7: base=0 -> CFM on clean pixels only (radius 2); base=1 -> median of clean pixels (radius 2)
            if base == 0:
                best = -1.0; bv = e0
                for yy in range(max(y - 2, 0), min(h, y + 3)):
                    for xx in range(max(x - 2, 0), min(w, x + 3)):
                        if (yy != y or xx != x) and clean[yy, xx]:
                            sc = 0.0
                            for y2 in range(max(y - 2, 0), min(h, y + 3)):
                                for x2 in range(max(x - 2, 0), min(w, x + 3)):
                                    if (y2 != y or x2 != x) and clean[y2, x2]:
                                        d = abs(img[y2, x2] - img[yy, xx])
                                        cb = max(abs(y2 - yy), abs(x2 - xx))
                                        sc += (150.0 / (150.0 + d)) * (1.5 / (1.5 + cb))
                            if sc > best:
                                best = sc; bv = img[yy, xx]
                F[y, x, 7] = bv
            else:
                vals = []
                for yy in range(max(y - 2, 0), min(h, y + 3)):
                    for xx in range(max(x - 2, 0), min(w, x + 3)):
                        if (yy != y or xx != x) and clean[yy, xx]:
                            vals.append(img[yy, xx])
                F[y, x, 7] = np.median(np.array(vals)) if len(vals) >= 1 else e0
    return F


@njit(cache=True)
def _clean_median(img, rmax, need):
    h, w = img.shape
    out = img.astype(np.float64)
    for y in range(h):
        for x in range(w):
            if img[y, x] == 0 or img[y, x] == 255:
                for r in range(1, rmax + 1):
                    win = img[max(y - r, 0):min(h, y + r + 1), max(x - r, 0):min(w, x + r + 1)].ravel()
                    good = win[(win != 0) & (win != 255)]
                    if good.size >= need:
                        out[y, x] = np.median(good)
                        break
    return out


def clean_median_expanding(noisy, rmax=5, need=3):
    """Selective median of the noise-free pixels in the smallest (2r+1)^2 window (r<=rmax)
    holding at least `need` of them -- the strong, standard 'switching median' baseline."""
    return np.floor(_clean_median(np.ascontiguousarray(noisy), rmax, need) + 0.5).astype(np.uint8)


@njit(cache=True)
def _clean_cfm(img, rmax, need):
    h, w = img.shape
    out = img.astype(np.float64)
    for y in range(h):
        for x in range(w):
            if img[y, x] == 0 or img[y, x] == 255:
                for r in range(1, rmax + 1):
                    ys = []; xs = []; vs = []
                    for yy in range(max(y - r, 0), min(h, y + r + 1)):
                        for xx in range(max(x - r, 0), min(w, x + r + 1)):
                            v = img[yy, xx]
                            if v != 0 and v != 255:
                                ys.append(yy); xs.append(xx); vs.append(float(v))
                    n = len(vs)
                    if n >= need:
                        best = -1.0; bv = 0.0
                        for i in range(n):
                            sc = 0.0
                            for j in range(n):
                                d = abs(vs[j] - vs[i])
                                cb = max(abs(ys[j] - ys[i]), abs(xs[j] - xs[i]))
                                sc += (150.0 / (150.0 + d)) * (1.5 / (1.5 + cb))
                            if sc > best:
                                best = sc; bv = vs[i]
                        out[y, x] = bv
                        break
    return out


def clean_cfm_expanding(noisy, rmax=5, need=3):
    """Selective CFM of the noise-free pixels in the smallest (2r+1)^2 window (r<=rmax) holding at least `need`
    of them: the CFM counterpart of clean_median_expanding (same window rule, same K=150, C=1.5)."""
    return np.floor(_clean_cfm(np.ascontiguousarray(noisy), rmax, need) + 0.5).astype(np.uint8)
