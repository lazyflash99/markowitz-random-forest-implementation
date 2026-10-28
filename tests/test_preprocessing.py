import numpy as np
import pandas as pd
from data import preprocess_train_test


def test_preprocessing_fits_on_train_and_handles_unseen_categories():
    train = pd.DataFrame({"number": [1.0, np.nan, 3.0], "kind": ["a", "b", None]})
    test = pd.DataFrame({"number": [np.nan, 10.0], "kind": ["unseen", "a"]})
    xtr, xte, transformer = preprocess_train_test(train, test)

    assert xtr.shape == (3, xte.shape[1])
    assert np.isfinite(xtr).all() and np.isfinite(xte).all()
    # The missing test numeric value is imputed with the training mean, not test data.
    assert xte[0, 0] == 2.0
    # An unseen test category does not add a feature or fail transformation.
    assert xte.shape[1] == xtr.shape[1]
    assert transformer is not None


def test_preprocessing_keeps_all_missing_numeric_column():
    train = pd.DataFrame({"empty": [np.nan, np.nan], "x": [2.0, 4.0]})
    test = pd.DataFrame({"empty": [np.nan], "x": [6.0]})
    xtr, xte, _ = preprocess_train_test(train, test)
    assert xtr.shape[1] == 2
    assert np.isfinite(xtr).all() and np.isfinite(xte).all()
