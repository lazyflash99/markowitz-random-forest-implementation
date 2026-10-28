import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CONFIG
from data import load_dataset, preprocess_train_test
from metrics import evaluate
from forest import tune_rf
from weights import (prepare, mrf_weights, ivrf_weights, swrf_weights, owrf_weights, uniform_weights)
from predict import weighted_predict
from forest import tree_outputs
from baselines import fit_xgb, fit_ert

WEIGHTERS = {"RF": uniform_weights, "MRF": mrf_weights, "IVRF": ivrf_weights,
             "SWRF": swrf_weights, "OWRF": owrf_weights}


# Path to the primary output raw results CSV file.
def raw_path():
    return os.path.join(CONFIG["RESULTS_DIR"], "raw.csv")


# Convert metric dictionary and execution runtime into long-format records.
def _rows(ds, task, trial, method, metrics, t):
    base = dict(dataset=ds, task=task, trial=trial, method=method)
    return [dict(base, metric=k, value=float(v)) for k, v in {**metrics, "time_s": t}.items()]

_GLOBAL_CACHE = {}


# Execute one single random trial: split, preprocess, tune, weight, and predict.
def run_trial(ds, X, y, task, trial):
    seed = CONFIG["SEED"] + trial
    # Global preprocessing: transform full matrix prior to train/test split.
    if CONFIG.get("GLOBAL_PREPROCESSING", False):
        if ds not in _GLOBAL_CACHE:
            X_all, _, _ = preprocess_train_test(X, X)
            _GLOBAL_CACHE[ds] = X_all
        X_mat = _GLOBAL_CACHE[ds]
        Xtr, Xte, ytr, yte = train_test_split(
            X_mat, y, test_size=CONFIG["TEST_SIZE"], random_state=seed,
            stratify=None if task == "regression" else y)
        preprocessing_s = 0.0
    # Leak-free preprocessing: fit transformers only on the training split.
    else:
        Xtr_raw, Xte_raw, ytr, yte = train_test_split(
            X, y, test_size=CONFIG["TEST_SIZE"], random_state=seed,
            stratify=None if task == "regression" else y)
        t0 = time.perf_counter()
        Xtr, Xte, _preprocessor = preprocess_train_test(Xtr_raw, Xte_raw)
        preprocessing_s = time.perf_counter() - t0

    # Tune hyperparameters and train base forest.
    t0 = time.perf_counter()
    rf = tune_rf(Xtr, ytr, task, seed)
    rf_train_s = time.perf_counter() - t0
    t0 = time.perf_counter(); prepare(rf, Xtr, ytr, task); prep_s = time.perf_counter() - t0
    t0 = time.perf_counter(); out_te = tree_outputs(rf, Xte, task); test_output_s = time.perf_counter() - t0
    rows, wsave = [], {}
    # Derive weights and evaluate each Random Forest weighting variant.
    for name, fn in WEIGHTERS.items():
        t0 = time.perf_counter(); w = fn(rf, Xtr, ytr, task); weight_s = time.perf_counter() - t0
        wsave[name] = w
        t0 = time.perf_counter(); pred = weighted_predict(rf, w, Xte, task, out=out_te)
        predict_s = time.perf_counter() - t0
        dt = preprocessing_s + rf_train_s + prep_s + test_output_s + weight_s + predict_s
        rows += _rows(ds, task, trial, name, evaluate(yte, pred, task), dt)
    # Fit and evaluate competitor baselines (XGBoost and ExtraTrees).
    for name, fn in (("XGB", fit_xgb), ("ERT", fit_ert)):
        pred, dt = fn(Xtr, ytr, Xte, task, seed)
        dt += preprocessing_s
        rows += _rows(ds, task, trial, name, evaluate(yte, pred, task), dt)
    # Persist tree weights array to disk.
    os.makedirs(os.path.join(CONFIG["RESULTS_DIR"], "weights"), exist_ok=True)
    np.savez(os.path.join(CONFIG["RESULTS_DIR"], "weights", f"{ds}_t{trial}.npz"), **wsave)
    return rows


# Run the full experiment benchmark loop across all requested datasets and trials.
def run(datasets):
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    for ds in datasets:
        Xdf, y, task = load_dataset(ds)
        X = Xdf
        print(f"[{ds}] task={task} rows={X.shape[0]} cols={X.shape[1]}", flush=True)
        for t in range(CONFIG["N_TRIALS"]):
            done = set()
            # Skip already finished trials if resuming interrupted runs.
            if os.path.exists(raw_path()):
                r = pd.read_csv(raw_path(), usecols=["dataset", "trial", "method", "metric"])
                complete = (r[r.metric == "time_s"]
                            .groupby(["dataset", "trial"]).method.nunique())
                expected_methods = len(WEIGHTERS) + 2
                done = {key for key, count in complete.items() if count == expected_methods}
            if (ds, t) in done:
                print(f"[{ds}] trial {t} already done, skipping"); continue
            t0 = time.perf_counter()
            rows = run_trial(ds, X, y, task, t)
            pd.DataFrame(rows).to_csv(raw_path(), mode="a", header=not os.path.exists(raw_path()), index=False)
            print(f"[{ds}] trial {t + 1}/{CONFIG['N_TRIALS']} done in {time.perf_counter() - t0:.1f}s", flush=True)
