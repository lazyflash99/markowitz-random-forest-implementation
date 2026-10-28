import numpy as np
from forest import tree_outputs


# Predict targets by taking a weighted combination of individual tree outputs.
def weighted_predict(rf, w, X, task, out=None):
    # Retrieve raw predictions from each tree if not already provided.
    if out is None:
        out = tree_outputs(rf, X, task)
    # Multiclass: contract tree dimension k with weight vector w to get (m, C) probabilities.
    if task == "multiclass":
        return np.einsum("mkc,k->mc", out, w)
    # Binary and regression: matrix-vector product across trees (m, k) @ (k,) -> (m,).
    return out @ w
