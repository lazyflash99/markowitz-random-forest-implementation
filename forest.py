from functools import partial
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.utils.class_weight import compute_sample_weight
from config import CONFIG


def rf_space(task):
    sp = {
        "n_estimators": hp.choice("n_estimators", list(range(10, CONFIG["N_EST_MAX"] + 1, 10))),
        "max_depth": hp.choice("max_depth", [None] + list(range(2, 11))),
        "min_samples_split": hp.choice("mss", [hp.quniform("mss_i", 2, 20, 1), hp.uniform("mss_f", 0.01, 0.5)]),
        "min_samples_leaf": hp.choice("msl", [hp.quniform("msl_i", 1, 20, 1), hp.uniform("msl_f", 0.01, 0.2)]),
        "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
    }
    if task != "regression":
        sp["criterion"] = hp.choice("criterion", ["gini", "entropy"])
    return sp


# Disambiguate numeric vs float fraction parameter choices from Hyperopt.
def clean_rf_params(p):
    p = dict(p)
    p["n_estimators"] = int(p["n_estimators"])
    v = p["min_samples_split"]; p["min_samples_split"] = int(v) if v >= 2 else float(v)
    v = p["min_samples_leaf"];  p["min_samples_leaf"] = int(v) if v >= 1 else float(v)
    return p


# Instantiate scikit-learn Random Forest model for regression or classification.
def make_rf(task, params, seed):
    p = clean_rf_params(params)
    if task == "regression":
        return RandomForestRegressor(**p, criterion="squared_error", bootstrap=True, n_jobs=-1, random_state=seed)
    return RandomForestClassifier(**p, class_weight="balanced", bootstrap=True, n_jobs=-1, random_state=seed)


# Compute validation loss to minimize during hyperparameter optimization.
def _loss(model, Xv, yv, task):
    if task == "binary":
        return -average_precision_score(yv, model.predict_proba(Xv)[:, 1])       # Objective: -PRAUC
    if task == "multiclass":
        return -f1_score(yv, model.predict(Xv), average="macro", zero_division=0)  # Objective: -F1_macro
    return mean_squared_error(yv, model.predict(Xv))                              # Objective: MSE


# Run Tree-structured Parzen Estimator (TPE) with 3-fold cross validation.
def tune_params(space, build, X, y, task, seed, sample_weight=False):
    cv = (KFold if task == "regression" else StratifiedKFold)(3, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y))
    best = {"loss": np.inf, "params": None}

    def objective(params):
        ls = []
        for tr, va in splits:
            m = build(params)
            if sample_weight:
                m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
            else:
                m.fit(X[tr], y[tr])
            ls.append(_loss(m, X[va], y[va], task))
        l = float(np.mean(ls))
        if l < best["loss"]:
            best.update(loss=l, params=dict(params))
        return {"loss": l, "status": STATUS_OK}

    n = CONFIG["TPE_EVALS"]
    fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=min(5, n)), max_evals=n,
         trials=Trials(), rstate=np.random.default_rng(seed), show_progressbar=False)
    return best["params"]


# Tune hyperparameters and aggregate the 3 CV fold models into one unified ensemble.
def tune_rf(X_tr, y_tr, task, seed):
    params = tune_params(rf_space(task), lambda p: make_rf(task, p, seed), X_tr, y_tr, task, seed)
    if not CONFIG.get("MODEL_AGGREGATION", False):
        return make_rf(task, params, seed).fit(X_tr, y_tr)

    from copy import deepcopy
    cv = (KFold if task == "regression" else StratifiedKFold)(3, shuffle=True, random_state=seed)
    models = []
    oob_masks = []
    n_rows = len(X_tr)
    # Fit one model per fold and construct exact out-of-bag masks.
    for fold_i, (tr, va) in enumerate(cv.split(X_tr, y_tr)):
        m = make_rf(task, params, seed + fold_i).fit(X_tr[tr], y_tr[tr])
        models.append(m)
        for boot_idx in getattr(m, "estimators_samples_", []):
            in_bag = tr[boot_idx]
            mask = np.ones(n_rows, dtype=bool)
            mask[in_bag] = False
            oob_masks.append(mask)

    aggregated = deepcopy(models[0])
    all_estimators = []
    for m in models:
        all_estimators.extend(m.estimators_)
    aggregated.estimators_ = all_estimators
    aggregated.n_estimators = len(all_estimators)
    aggregated._mrf_oob_masks = np.array(oob_masks, dtype=bool)
    return aggregated


# Return boolean matrix (k, m) where True marks out-of-bag examples for each tree.
def tree_oob_masks(rf, n_rows):
    if hasattr(rf, "_mrf_oob_masks"):
        return rf._mrf_oob_masks
    masks = np.ones((len(rf.estimators_), n_rows), dtype=bool)
    for j, idx in enumerate(getattr(rf, "estimators_samples_", [])):
        masks[j, idx] = False
    return masks


# Collect raw predictions from every decision tree: (m, k) or (m, k, C).
def tree_outputs(rf, X, task):
    Xn = np.asarray(X, dtype=np.float32)
    if task == "regression":
        return np.column_stack([t.predict(Xn) for t in rf.estimators_])
    if task == "binary":
        return np.column_stack([t.predict_proba(Xn)[:, 1] for t in rf.estimators_])
    return np.stack([t.predict_proba(Xn) for t in rf.estimators_], axis=1)
