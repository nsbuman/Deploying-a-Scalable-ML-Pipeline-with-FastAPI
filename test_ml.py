import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference, train_model


def test_compute_model_metrics():
    """
    Test that compute_model_metrics correctly computes precision, recall, and fbeta.
    """
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision == 1.0
    assert recall == 2.0 / 3.0
    assert fbeta == pytest.approx(0.8)


def test_train_model():
    """
    Test that train_model returns a trained RandomForestClassifier instance.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_inference():
    """
    Test that inference returns predictions of the expected type and shape.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    model = train_model(X_train, y_train)

    X_test = np.array([[2, 3], [6, 7]])
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)
