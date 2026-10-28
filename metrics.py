import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error, r2_score)


# Compute standard evaluation metrics depending on the task.
def evaluate(y_true, out, task):
    # Binary classification: output is positive class probability vector.
    if task == "binary":
        pred = (out >= 0.5).astype(int)
        try:
            roc = roc_auc_score(y_true, out)
        except ValueError:
            roc = np.nan
        return {"PRAUC": average_precision_score(y_true, out), "ROC": roc,
                "F1": f1_score(y_true, pred, zero_division=0),
                "Precision": precision_score(y_true, pred, zero_division=0),
                "Recall": recall_score(y_true, pred, zero_division=0),
                "Accuracy": accuracy_score(y_true, pred)}
    # Multiclass classification: output is class probability matrix (m, C).
    if task == "multiclass":
        pred = out.argmax(1)
        return {"F1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
                "F1_micro": f1_score(y_true, pred, average="micro", zero_division=0),
                "Accuracy": accuracy_score(y_true, pred)}
    # Regression: output is continuous prediction vector.
    return {"MAE": mean_absolute_error(y_true, out),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, out))),
            "MAPE": mean_absolute_percentage_error(y_true, out),
            "R2": r2_score(y_true, out)}
