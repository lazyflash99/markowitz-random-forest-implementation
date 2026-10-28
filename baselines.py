# XGBoost and Extremely Randomized Trees (ERT) baselines tuned with TPE/3-CV.
import time
import numpy as np
from hyperopt import hp
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
from config import CONFIG
from forest import rf_space, clean_rf_params, tune_params


# Generate model predictions: positive probability for binary, probability matrix for multiclass, values for regression.
def _predict(model, X, task):
    if task == "regression":
        return model.predict(X)
    p = model.predict_proba(X)
    return p[:, 1] if task == "binary" else p


# Instantiate ExtraTrees model.
def make_ert(task, params, seed):
    p = clean_rf_params(params)
    if task == "regression":
        return ExtraTreesRegressor(**p, criterion="squared_error", n_jobs=-1, random_state=seed)
    return ExtraTreesClassifier(**p, class_weight="balanced", n_jobs=-1, random_state=seed)


# Tune and fit ExtraTrees, combining 3-CV fold models if model aggregation is active.
def fit_ert(X_tr, y_tr, X_te, task, seed):
    t0 = time.perf_counter()
    params = tune_params(rf_space(task), lambda p: make_ert(task, p, seed), X_tr, y_tr, task, seed)
    if not CONFIG.get("MODEL_AGGREGATION", False):
        m = make_ert(task, params, seed).fit(X_tr, y_tr)
        pred = _predict(m, X_te, task)
    else:
        from sklearn.model_selection import KFold, StratifiedKFold
        cv = (KFold if task == "regression" else StratifiedKFold)(3, shuffle=True, random_state=seed)
        preds = []
        for fold_i, (tr, va) in enumerate(cv.split(X_tr, y_tr)):
            m = make_ert(task, params, seed + fold_i).fit(X_tr[tr], y_tr[tr])
            preds.append(_predict(m, X_te, task))
        pred = np.mean(preds, axis=0)
    return pred, time.perf_counter() - t0


# XGBoost hyperparameter search space from Table B.4.
def xgb_space():
    return {
        "learning_rate": hp.loguniform("learning_rate", -5, 0),
        "gamma": hp.uniform("gamma", 0, 5),
        "max_depth": hp.choice("max_depth", list(range(1, 8))),
        "min_child_weight": hp.uniform("min_child_weight", 1, 10),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
        "reg_lambda": hp.uniform("reg_lambda", 1, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        "max_bin": hp.choice("max_bin", list(range(64, 769, 64))),
    }


# Instantiate XGBoost model.
def make_xgb(task, params, seed):
    kw = dict(params, n_estimators=CONFIG["XGB_N_EST"], tree_method="hist", n_jobs=-1, random_state=seed)
    if task == "regression":
        return XGBRegressor(objective="reg:squarederror", **kw)
    if task == "binary":
        return XGBClassifier(objective="binary:logistic", **kw)
    return XGBClassifier(objective="multi:softprob", **kw)


# Tune XGBoost via 3-CV, then refit on the full training set.
def fit_xgb(X_tr, y_tr, X_te, task, seed):
    t0 = time.perf_counter()
    clf = task != "regression"
    use_weights = clf and CONFIG.get("XGB_BALANCED_WEIGHTS", False)
    params = tune_params(xgb_space(), lambda p: make_xgb(task, p, seed), X_tr, y_tr, task, seed,
                         sample_weight=use_weights)
    m = make_xgb(task, params, seed)
    if use_weights:
        m.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        m.fit(X_tr, y_tr)
    pred = _predict(m, X_te, task)
    return pred, time.perf_counter() - t0
