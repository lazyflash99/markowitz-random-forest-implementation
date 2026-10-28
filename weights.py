# Tree weighting methods: f(rf, X_tr, y_tr, task) -> normalized weights vector w.
import logging
import hashlib
import numpy as np
import cvxpy as cp
from sklearn.metrics import (average_precision_score, f1_score, accuracy_score,
                             mean_absolute_percentage_error, mean_absolute_error)
from config import CONFIG
from forest import tree_outputs, tree_oob_masks
import risk

log = logging.getLogger("mrf")


# Cache tree outputs and out-of-bag masks to avoid redundant computation across methods.
def _cache(rf, X, task):
    arr = np.ascontiguousarray(np.asarray(X))
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
    key = (arr.shape, arr.dtype.str, task, digest)
    c = getattr(rf, "_mrf_cache", None)
    if c is None or c.get("key") != key:
        c = {"key": key, "out": tree_outputs(rf, X, task),
             "oob": tree_oob_masks(rf, X.shape[0])}
        rf._mrf_cache = c
    return c


def prepare(rf, X, y, task):
    _cache(rf, X, task)


def _normalise(s):
    s = np.clip(np.nan_to_num(np.asarray(s, float)), 0, None)
    return np.full(len(s), 1 / len(s)) if s.sum() <= 0 else s / s.sum()


def _per_tree(out, y, mask, task, kind):
    k = out.shape[1]
    vals = np.full(k, np.nan)
    for j in range(k):
        idx = mask[j]
        if idx.sum() == 0:
            continue
        yt = y[idx]
        if task == "binary":
            p = out[idx, j]
            if kind == "mrf":
                if yt.sum() == 0:
                    continue
                vals[j] = average_precision_score(yt, p)
            else:
                vals[j] = accuracy_score(yt, (p >= 0.5).astype(int))
        elif task == "multiclass":
            pred = out[idx, j, :].argmax(1)
            vals[j] = (f1_score(yt, pred, average="macro", zero_division=0) if kind == "mrf"
                       else accuracy_score(yt, pred))
        else:
            pred = out[idx, j]
            vals[j] = (np.exp(-mean_absolute_percentage_error(yt, pred)) if kind == "mrf"
                       else 1.0 / (mean_absolute_error(yt, pred) + 1e-8))
    if np.isnan(vals).all():
        return np.zeros(k)
    return np.where(np.isnan(vals), np.nanmean(vals), vals)


# Extract score matrix per example and tree: true-class prob for clf or signed residual for reg.
def _example_scores(out, y, task):
    if task == "binary":
        return np.where(y[:, None] == 1, out, 1 - out)
    if task == "multiclass":
        m, k, _ = out.shape
        return out[np.arange(m)[:, None], np.arange(k)[None, :], y[:, None]]
    return y[:, None] - out


# Delegate risk matrix computation according to the task type.
def _risk(out, y, task):
    if task == "binary":
        return risk.risk_binary(out, y)
    if task == "multiclass":
        return risk.risk_multiclass(out, y)
    return risk.risk_regression(out, y)


# Solve Mean-Variance Analysis quadratic program: max w'r - lam*w'Kw - gamma*|w|^2.
def _solve_mva(r, K, lam, gamma):
    k = len(r)
    # Symmetrize and project risk matrix onto positive semi-definite cone.
    K = (K + K.T) / 2
    vals, vecs = np.linalg.eigh(K)
    A = vecs * np.sqrt(np.clip(vals, 0, None))
    w = cp.Variable(k)
    obj = cp.Maximize(r @ w - lam * cp.sum_squares(A.T @ w) - gamma * cp.sum_squares(w))
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0, w <= 1])
    try:
        prob.solve()
    except Exception as e:
        log.warning("MVA solver failed (%s); using uniform weights", e)
        return np.full(k, 1 / k)
    if w.value is None:
        log.warning("MVA returned no solution (%s); using uniform weights", prob.status)
        return np.full(k, 1 / k)
    w = np.clip(w.value, 0, None)
    w[w < 1e-6] = 0.0
    return _normalise(w)


# Markowitz Random Forest weights: OOB performance vector r and full-training risk matrix K.
def mrf_weights(rf, X, y, task):
    c = _cache(rf, X, task)
    r = _per_tree(c["out"], y, c["oob"], task, "mrf")
    K = _risk(c["out"], y, task)
    return _solve_mva(r, K, CONFIG["LAMBDA"], CONFIG["L2_GAMMA"])


# Inverse Variance Random Forest: tree scores scaled by individual prediction variance.
def ivrf_weights(rf, X, y, task):
    c = _cache(rf, X, task)
    r = _per_tree(c["out"], y, c["oob"], task, "mrf")
    var = np.maximum(_example_scores(c["out"], y, task).var(0), 1e-12)
    return _normalise(r / var)


# Score-Weighted Random Forest: tree accuracy over the entire training set (Eq. 26).
def swrf_weights(rf, X, y, task):
    c = _cache(rf, X, task)
    return _normalise(_per_tree(c["out"], y, np.ones_like(c["oob"]), task, "acc"))


# Out-of-Bag Weighted Random Forest: tree accuracy over out-of-bag rows only (Eq. 27).
def owrf_weights(rf, X, y, task):
    c = _cache(rf, X, task)
    return _normalise(_per_tree(c["out"], y, c["oob"], task, "acc"))


# Standard Random Forest: equal uniform weights 1 / k for all trees.
def uniform_weights(rf, X, y, task):
    k = len(rf.estimators_)
    return np.full(k, 1 / k)
