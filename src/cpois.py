"""Fixed-effects Poisson via the conditional (multinomial) likelihood.

Model: log E[y_it] = alpha_i + x_it' beta + sum_g gamma_g[code_g(it)],
with one high-dimensional effect alpha_i (the article) and a few
low-dimensional ones (calendar month, article age).  Conditioning on
n_i = sum_t y_it removes alpha_i exactly; beta and the gamma's are then
estimated by Newton's method on the multinomial log-likelihood, which gives
the same estimates as Poisson with article dummies.  The covariance is
cluster-robust (by article unless another cluster is given).

The age and calendar dummies are collinear with the article effects along
one direction (age-period-cohort); this does not affect beta, and the
Newton steps and the bread use a pseudo-inverse.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln


def _codes(f):
    u, inv = np.unique(f, return_inverse=True)
    return inv.astype(np.int64), u


def _keep_mask(y, unit, small):
    keep = np.ones(len(y), bool)
    while True:
        bad = np.zeros(len(y), bool)
        for f in [unit] + small:
            s = np.bincount(f[keep], weights=y[keep], minlength=f.max() + 1)
            bad |= keep & (s[f] == 0)
        if not bad.any():
            return keep
        keep &= ~bad


def _design(X, small):
    """Sparse design [X | dummies(small FE, first level dropped)]."""
    n = X.shape[0]
    blocks = [sp.csr_matrix(X)]
    for f in small:
        L = f.max() + 1
        D = sp.csr_matrix((np.ones(n), (np.arange(n), f)), shape=(n, L))[:, 1:]
        blocks.append(D)
    return sp.hstack(blocks, format="csr")


def cpois(y, X, unit, small=(), cluster=None, theta0=None, tol=1e-10, maxit=100, offset=None):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    unit = np.asarray(unit)
    small = [np.asarray(f) for f in small]
    keep = _keep_mask(y, _codes(unit)[0], [_codes(f)[0] for f in small])
    y, X = y[keep], X[keep]
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)[keep]
    unit = _codes(unit[keep])[0]
    small = [_codes(f[keep])[0] for f in small]
    clus = unit if cluster is None else _codes(np.asarray(cluster)[keep])[0]
    n, p = X.shape
    Z = _design(X, small)
    q = Z.shape[1]
    nu = unit.max() + 1
    U = sp.csr_matrix((np.ones(n), (unit, np.arange(n))), shape=(nu, n))
    ni = np.bincount(unit, weights=y)

    def evaluate(theta, need_hess=True):
        eta = Z @ theta + off
        m = np.full(nu, -np.inf)
        np.maximum.at(m, unit, eta)
        e = np.exp(eta - m[unit])
        s = np.bincount(unit, weights=e, minlength=nu)
        pi = e / s[unit]
        ll = float(np.sum(y * np.log(np.maximum(pi, 1e-300))))
        if not need_hess:
            return ll, pi
        r = y - ni[unit] * pi
        g = Z.T @ r
        w = ni[unit] * pi
        A = (Z.T @ Z.multiply(w[:, None]).tocsr()).toarray()
        M = (U @ Z.multiply(pi[:, None]).tocsr()).toarray()  # nu x q
        H = A - M.T @ (M * ni[:, None])
        return ll, pi, np.asarray(g).ravel(), H

    if theta0 is None:
        theta = np.zeros(q)
    elif len(theta0) == q:
        theta = np.array(theta0, float)
    else:
        theta = np.r_[theta0, np.zeros(q - len(theta0))]
    converged = False
    for it in range(maxit):
        ll, pi, g, H = evaluate(theta)
        step = np.linalg.lstsq(H, g, rcond=1e-12)[0]
        t = 1.0
        for _ in range(40):
            ll_new = evaluate(theta + t * step, need_hess=False)[0]
            if ll_new >= ll - 1e-9 * (1 + abs(ll)):
                break
            t *= 0.5
        theta = theta + t * step
        if (p == 0 or np.max(np.abs(t * step[:p])) < tol) and abs(ll_new - ll) < 1e-10 * (1 + abs(ll)):
            converged = True
            break

    ll, pi, g, H = evaluate(theta)
    Hinv = np.linalg.pinv(H, rcond=1e-12, hermitian=True)
    r = y - ni[unit] * pi
    G = clus.max() + 1
    C = sp.csr_matrix((np.ones(n), (clus, np.arange(n))), shape=(G, n))
    S = (C @ Z.multiply(r[:, None]).tocsr()).toarray()
    meat = S.T @ S * G / (G - 1)
    V = Hinv @ meat @ Hinv
    mu = ni[unit] * pi
    # Poisson log-likelihood with the article effects profiled out
    ll_pois = float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu - gammaln(y + 1)))
    nfe = nu + q - p - 1
    disp = float(np.sum(np.where(mu > 0, (y - mu) ** 2 / np.maximum(mu, 1e-300), 0)) / max(n - p - nfe, 1))
    return dict(beta=theta[:p], se=np.sqrt(np.diag(V)[:p]), vcov=V[:p, :p],
                loglik=ll_pois, ll_cond=ll, n=n, nunits=nu, nclusters=G,
                dispersion=disp, converged=converged, iters=it + 1, theta=theta,
                keep=keep, mu=mu)
