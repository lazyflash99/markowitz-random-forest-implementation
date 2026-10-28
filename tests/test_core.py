import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import risk, weights
from predict import weighted_predict

METHODS = [weights.mrf_weights, weights.ivrf_weights, weights.swrf_weights, weights.owrf_weights,
           weights.uniform_weights]


def _setup(task):
    if task == "regression":
        X, y = make_regression(400, 8, noise=10, random_state=0)
        rf = RandomForestRegressor(15, max_depth=6, random_state=0).fit(X, y)
    else:
        nc = 2 if task == "binary" else 3
        X, y = make_classification(500, 10, n_informative=6, n_classes=nc, flip_y=0.1, random_state=0)
        rf = RandomForestClassifier(15, max_depth=6, class_weight="balanced", random_state=0).fit(X, y)
    return rf, X, y


def _naive_binary(P, y):
    m, k = P.shape
    Q = np.where(y[:, None] == 1, P, 1 - P)
    K = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            rs = []
            for lab in (0, 1):
                s = (y == lab) & (np.minimum(Q[:, a], Q[:, b]) < 0.5)
                if s.sum() >= 3 and Q[s, a].std() > 1e-6 and Q[s, b].std() > 1e-6:
                    rs.append(np.corrcoef(Q[s, a], Q[s, b])[0, 1])
            K[a, b] = K[b, a] = np.mean(rs) if rs else 0.0
    return K


def test_vectorised_matches_naive_binary():
    rng = np.random.default_rng(1)
    P = rng.uniform(size=(300, 5)); y = rng.integers(0, 2, 300)
    assert np.allclose(risk.risk_binary(P, y), _naive_binary(P, y), atol=1e-8)



def _naive_multiclass(P3, y):
    m, k, C = P3.shape
    D = P3.argmax(2)
    K = np.eye(k)
    for a in range(k):
        for b in range(a + 1, k):
            rs = []
            for c in range(C):
                rows = np.where(y == c)[0]
                mask = (D[rows, a] != c) | (D[rows, b] != c)
                if mask.sum() < 3:
                    continue
                xa = P3[rows[mask], a, :].reshape(-1)
                xb = P3[rows[mask], b, :].reshape(-1)
                if xa.std() > 1e-6 and xb.std() > 1e-6:
                    rs.append(np.corrcoef(xa, xb)[0, 1])
            K[a, b] = K[b, a] = np.mean(rs) if rs else 0.0
    return K


def test_vectorised_matches_naive_multiclass():
    rng = np.random.default_rng(17)
    P3 = rng.dirichlet(np.ones(4), size=(600, 6))
    y = rng.integers(0, 4, 600)
    assert np.allclose(risk.risk_multiclass(P3, y), _naive_multiclass(P3, y), atol=1e-8)

def test_identical_trees_risk_one():
    rng = np.random.default_rng(0)
    P = rng.uniform(size=(300, 3)); P[:, 1] = P[:, 0]; y = rng.integers(0, 2, 300)
    K = risk.risk_binary(P, y)
    assert np.isclose(K[0, 1], 1) and np.allclose(K, K.T)
    P3 = rng.dirichlet(np.ones(3), size=(200, 3)); P3[:, 1] = P3[:, 0]; y3 = rng.integers(0, 3, 200)
    K3 = risk.risk_multiclass(P3, y3)
    assert np.isclose(K3[0, 1], 1) and np.allclose(K3, K3.T)


def test_small_sets_do_not_crash():
    P = np.random.default_rng(0).uniform(size=(5, 3)); y = np.array([0, 1, 0, 1, 1])
    K = risk.risk_binary(P, y)
    assert np.isfinite(K).all()


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_weights_contract(task):
    rf, X, y = _setup(task)
    for f in METHODS:
        w = f(rf, X, y, task)
        assert len(w) == len(rf.estimators_) and (w >= 0).all() and np.isclose(w.sum(), 1)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_uniform_reproduces_rf(task):
    rf, X, y = _setup(task)
    w = weights.uniform_weights(rf, X, y, task)
    out = weighted_predict(rf, w, X, task)
    ref = rf.predict(X) if task == "regression" else rf.predict_proba(X)
    if task == "binary":
        ref = ref[:, 1]
    assert np.allclose(out, ref)


def test_mrf_differs_from_uniform():
    rf, X, y = _setup("binary")
    w = weights.mrf_weights(rf, X, y, "binary")
    assert not np.allclose(w, 1 / len(w))


def test_model_aggregation():
    from forest import tune_rf, tree_oob_masks
    from config import CONFIG
    X, y = make_classification(120, 6, n_classes=2, random_state=42)
    old_evals = CONFIG["TPE_EVALS"]
    CONFIG["TPE_EVALS"] = 2
    try:
        rf = tune_rf(X, y, "binary", seed=42)
        assert hasattr(rf, "_mrf_oob_masks")
        oob = tree_oob_masks(rf, len(X))
        assert oob.shape[0] == len(rf.estimators_)
        assert oob.shape[1] == len(X)
        assert oob.any(axis=1).all()  # every tree has some OOB samples
        w = weights.mrf_weights(rf, X, y, "binary")
        assert len(w) == len(rf.estimators_)
        assert np.isclose(w.sum(), 1.0)
    finally:
        CONFIG["TPE_EVALS"] = old_evals

