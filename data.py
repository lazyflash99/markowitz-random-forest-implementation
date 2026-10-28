import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from config import CONFIG


# Fetch dataset from local data/ directory if present, or download from UCI repository.
def _fetch(name, spec):
    local = os.path.join("data", f"{name}.csv")
    if os.path.exists(local):
        df = pd.read_csv(local)
        if "target" not in df.columns:
            raise ValueError(f"{local} must contain a 'target' column")
        return df.drop(columns="target"), df["target"]

    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=spec["id"])
    X = ds.data.features.copy()
    if ds.data.targets is None or ds.data.targets.shape[1] == 0:
        raise ValueError(f"UCI dataset {spec['id']} ({name}) has no target column")
    # Use primary target column if multiple are provided.
    y = ds.data.targets.iloc[:, 0].copy()
    try:
        orig = ds.data.original
        if name == "wine" and "color" in orig.columns and "color" not in X.columns:
            X["color"] = orig["color"].to_numpy()
    except Exception:
        pass
    return X, y


# Load raw feature matrix, clean duplicates, and encode target array.
def load_dataset(name):
    spec = CONFIG["DATASETS"][name]
    task = spec["task"]
    X, y = _fetch(name, spec)
    X = X.reset_index(drop=True).copy()
    y = pd.Series(np.asarray(y), name="__target__")
    if len(X) != len(y):
        raise ValueError(f"{name}: feature/target row counts differ ({len(X)} vs {len(y)})")

    # Remove identical duplicate rows across features and target.
    combined = X.copy()
    combined["__target__"] = y.to_numpy()
    combined = combined.drop_duplicates().reset_index(drop=True)
    y = combined.pop("__target__")
    X = combined

    # Optional subsampling cap for slow machines.
    cap = CONFIG["MAX_ROWS"].get(name)
    if cap and len(X) > cap:
        chosen = X.sample(n=int(cap), random_state=0).index
        X = X.loc[chosen].reset_index(drop=True)
        y = y.loc[chosen].reset_index(drop=True)

    if task == "regression":
        y = pd.to_numeric(y, errors="raise").to_numpy(dtype=float)
    else:
        y = LabelEncoder().fit_transform(y.astype(str).to_numpy())
        # Ensure positive class is minority for binary classification.
        if task == "binary" and y.mean() > 0.5:
            y = 1 - y
    return X.reset_index(drop=True), np.asarray(y), task


# Impute missing values (mean for numeric, constant for categorical) and one-hot encode.
def preprocess_train_test(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    numeric = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    transformers = []
    # Numeric pipeline: impute missing values with column mean.
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    # Categorical pipeline: handle missing strings and one-hot encode.
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        for frame in (X_train, X_test):
            for col in categorical:
                frame[col] = frame[col].where(frame[col].notna(), "__MISSING__").astype(str)
        transformers.append(("categorical", categorical_pipe, categorical))

    if not transformers:
        raise ValueError("Dataset has no usable feature columns")
    transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    Xtr = transformer.fit_transform(X_train)
    Xte = transformer.transform(X_test)
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
        raise ValueError("Preprocessing produced NaN or infinite feature values")
    return Xtr, Xte, transformer
