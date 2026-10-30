"""Local regression + shrinkage solvers E1-E6 of the paper (from bank_filter.py; one numerical safeguard added in solve_ebridge for windows without data)."""
import numpy as np
from scipy.ndimage import uniform_filter
from ._kernels import _features


def _box(a, size):
    return uniform_filter(a, size=size, mode='reflect') * size * size

def offsets(rf):
    return [(dy, dx) for dy in range(-rf, rf + 1) for dx in range(-rf, rf + 1) if (dy, dx) != (0, 0)]

def build_stats(noisy, R=5, rf=2, use_cands=False, rmax=5, need=3, base=0):
    """base=0: initial estimate E0 = noise-free median (original); base=1: E0 = noise-free CFM."""
    noisy = np.ascontiguousarray(noisy)
    clean = ~((noisy == 0) | (noisy == 255)); img = noisy.astype(np.float64)
    F = _features(img, clean, rmax, need, base); E0 = F[..., 0]
    h, w = img.shape; cols = []
    pad = rf
    P = np.pad(img, pad, mode='edge'); C = np.pad(clean, pad, mode='edge')
    for dy, dx in offsets(rf):
        v = P[pad + dy:pad + dy + h, pad + dx:pad + dx + w]; c = C[pad + dy:pad + dy + h, pad + dx:pad + dx + w]
        cols.append(np.where(c, v, E0) - E0)
    D = np.stack(cols, -1)
    if use_cands: D = np.concatenate([D, F[..., 1:] - E0[..., None]], -1)
    r = img - E0; cf = clean.astype(np.float64); size = 2 * R + 1; q = D.shape[-1]
    n = _box(cf, size)
    A = np.empty(E0.shape + (q, q), dtype=np.float32); b = np.empty(E0.shape + (q,)); 
    for i in range(q):
        b[..., i] = _box(cf * D[..., i] * r, size)
        for j in range(i, q):
            A[..., i, j] = A[..., j, i] = _box(cf * D[..., i] * D[..., j], size)
    rr = _box(cf * r * r, size)
    return dict(noisy=noisy, clean=clean, E0=E0, D=D, A=A, b=b, rr=rr, n=n, q=q, base=base)

def _groups(q, ng):
    idx = np.array_split(np.arange(q), ng)           # ascending eigenvalue order; each group size >= 3 required
    return [list(g) for g in idx]

def solve(S, mode='gjs', ng=3, gamma=1.0, eps=1e-3, lam=30.0, min_nu=4, chunk=8192):
    A, b, n, q, D, E0, clean, noisy, rr = S['A'], S['b'], S['n'], S['q'], S['D'], S['E0'], S['clean'], S['noisy'], S['rr']
    h, w = E0.shape; N = h * w
    noisy_idx = np.flatnonzero(~clean.ravel())              # only noisy pixels need a solution
    A2 = A.reshape(N, q, q); b2 = b.reshape(N, q); n2 = n.ravel(); rr2 = rr.ravel(); D2 = D.reshape(N, q); E02 = E0.ravel()
    est = E02.copy(); facs = np.ones((N, q))
    groups = _groups(q, ng) if mode == 'gjs' else [list(range(q))]
    for s in range(0, len(noisy_idx), chunk):
        ii = noisy_idx[s:s + chunk]; Ac = A2[ii].astype(np.float64); bc = b2[ii]; nc = n2[ii]; rc = rr2[ii]
        if mode == 'ridge':
            beta = np.linalg.solve(Ac + (lam * np.maximum(nc, 1.0))[:, None, None] * np.eye(q), bc[..., None])[..., 0]
            fac = np.ones((len(ii), q))
        else:
            L, U = np.linalg.eigh(Ac); L = np.clip(L, 0, None)
            c = np.einsum('nji,nj->ni', U, bc); mu0 = (eps * L.mean(-1) + 1e-9)[:, None]
            bo = c / (L + mu0); W = np.clip(rc - 2 * (bo * c).sum(-1) + (L * bo ** 2).sum(-1), 0, None)
            nu = nc - q; z2 = L * bo ** 2; fac = np.ones((len(ii), q))
            if mode in ('js', 'gjs'):
                for g in groups:
                    Tg = z2[:, g].sum(-1)
                    f = np.clip(1 - gamma * (len(g) - 2) * W / ((np.maximum(nu, 1) + 2) * np.maximum(Tg, 1e-12)), 0, None)
                    fac[:, g] = f[:, None]
            fac = np.where((nu >= min_nu)[:, None], fac, 0.0) if mode != 'ols' else np.where((nu >= min_nu)[:, None], 1.0, 0.0) * np.ones_like(fac)
            beta = np.einsum('nij,nj->ni', U, bo * fac)
        est[ii] = E02[ii] + np.einsum('ni,ni->n', D2[ii], beta); facs[ii] = fac
    out = noisy.astype(np.float64); m = ~clean; out[m] = est.reshape(h, w)[m]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8), facs.reshape(h, w, q)


def solve_ebridge(S, rule='hkb', kappa=1.0, eps=1e-3, min_nu=4, chunk=8192):
    """Empirical-Bayes ridge (Lindley--Smith): beta ~ N(0, tau^2 I) => posterior mean = ridge with mu = sigma^2/tau^2.
    rule='hkb': mu = kappa * p s^2 / ||b_ols||^2 (Hoerl--Kennard--Baldwin);  rule='mom': tau^2 by moments,
    tau2 = max(0, (||b_ols||^2 - s^2 tr(A^-1)) / p), mu = s^2/tau2."""
    A, b, n, q, D, E0, clean, noisy, rr = S['A'], S['b'], S['n'], S['q'], S['D'], S['E0'], S['clean'], S['noisy'], S['rr']
    h, w = E0.shape; N = h * w; idx = np.flatnonzero(~clean.ravel())
    A2 = A.reshape(N, q, q); b2 = b.reshape(N, q); n2 = n.ravel(); rr2 = rr.ravel(); D2 = D.reshape(N, q); E02 = E0.ravel()
    est = E02.copy(); I = np.eye(q); mus = np.zeros(N)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; As = A2[ii].astype(np.float64); bs = b2[ii]; ns = n2[ii]; rs = rr2[ii]
        tr = np.trace(As, axis1=1, axis2=2) / q
        Ao = As + (eps * tr + 1e-9)[:, None, None] * I
        bo = np.linalg.solve(Ao, bs[..., None])[..., 0]
        W = np.clip(rs - 2 * (bo * bs).sum(-1) + np.einsum('ni,nij,nj->n', bo, As, bo), 0, None)
        nu = ns - q; s2 = W / np.maximum(nu, 1); nb = (bo ** 2).sum(-1)
        if rule == 'hkb': mu = kappa * q * s2 / np.maximum(nb, 1e-12)
        else:
            trinv = np.trace(np.linalg.inv(Ao), axis1=1, axis2=2)
            tau2 = np.maximum((nb - s2 * trinv) / q, 1e-12); mu = kappa * s2 / tau2
        # windows without usable data (A = 0, e.g. no clean pixel or a constant area) would give a singular system:
        # add a unit ridge there only (beta is forced to 0 for them anyway); other pixels are unchanged
        jit = np.where(tr > 0, 1e-9 * tr, 1.0)
        beta = np.linalg.solve(As + mu[:, None, None] * I + jit[:, None, None] * I, bs[..., None])[..., 0]
        beta = np.where((nu >= min_nu)[:, None], beta, 0.0)
        est[ii] = E02[ii] + np.einsum('ni,ni->n', D2[ii], beta); mus[ii] = mu
    out = noisy.astype(np.float64); m = ~clean; out[m] = est.reshape(h, w)[m]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8), mus.reshape(h, w)
