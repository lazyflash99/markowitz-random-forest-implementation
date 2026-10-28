# Risk matrices K
import warnings
import numpy as np

MIN_ROWS = 3


# Compute pooled Pearson correlation over the union-of-mistakes set
def _pair_corr(Xs, B):
    Bf = B.astype(float)
    nb = Bf.sum(0)
    n_rows = nb[:, None] + nb[None, :] - Bf.T @ Bf
    Sx = Sxx = Sxy = 0.0
    for X in Xs:
        XB = X * Bf
        Sx = Sx + XB.sum(0)[:, None] + X.T @ Bf - XB.T @ Bf
        X2 = X * X; X2B = X2 * Bf
        Sxx = Sxx + X2B.sum(0)[:, None] + X2.T @ Bf - X2B.T @ Bf
        Sxy = Sxy + XB.T @ X + X.T @ XB - XB.T @ XB
    Sy, Syy = Sx.T, Sxx.T
    N = n_rows * len(Xs)
    with np.errstate(all="ignore"):
        mx, my = Sx / N, Sy / N
        cov = Sxy / N - mx * my
        vx, vy = Sxx / N - mx ** 2, Syy / N - my ** 2
        c = cov / np.sqrt(vx * vy)
    ok = (n_rows >= MIN_ROWS) & (vx > 1e-12) & (vy > 1e-12) & np.isfinite(c)
    return np.where(ok, np.clip(c, -1, 1), np.nan)


def _finalize(risks, k):
    if not risks:
        K = np.zeros((k, k))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K = np.nanmean(np.stack(risks), axis=0)
        K = np.nan_to_num(K, nan=0.0)
    K = (K + K.T) / 2
    np.fill_diagonal(K, 1.0)
    return K


# Binary risk: average Pearson correlation on false positive and false negative mistake sets.
def risk_binary(P, y):
    Q = np.where(y[:, None] == 1, P, 1 - P)  # Probability assigned to the true label
    risks = []
    for lab in (0, 1):
        rows = y == lab
        if rows.sum() < MIN_ROWS:
            continue
        Qs = Q[rows]
        risks.append(_pair_corr([Qs], Qs < 0.5))
    return _finalize(risks, P.shape[1])


# Multiclass risk: correlation of probability vectors on mistake examples averaged across classes.
def risk_multiclass(P3, y):
    m, k, C = P3.shape
    D = P3.argmax(2)
    chans = [P3[:, :, c] for c in range(C)]
    risks = []
    for c in range(C):
        rows = np.where(y == c)[0]
        if len(rows) < MIN_ROWS:
            continue
        risks.append(_pair_corr([ch[rows] for ch in chans], D[rows] != c))
    return _finalize(risks, k)


# Regression risk: sample covariance matrix of signed residuals.
def risk_regression(Yhat, y):
    res = y[:, None] - Yhat
    K = np.atleast_2d(np.cov(res, rowvar=False))
    return np.nan_to_num((K + K.T) / 2)
