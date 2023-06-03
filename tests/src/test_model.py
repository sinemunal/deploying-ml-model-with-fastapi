from unittest.mock import patch

import pytest
import numpy as np
import sklearn

from src.model import train_model, inference


@pytest.fixture
def X_train():
    return np.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 1.0]]
    )


@pytest.fixture
def y_train():
    return np.array([1, 1, 1, 0])


def test_train_model_trains_random_forest_classifier(X_train, y_train):
    trained_model = train_model(X_train, y_train)

    assert type(trained_model) == sklearn.ensemble._forest.RandomForestClassifier


def test_train_model_does_gridsearchcv_when_optimized_is_enables(X_train, y_train):
    trained_model = train_model(X_train, y_train, is_optimized=True)

    assert trained_model.get_params()["max_depth"] == 2


def test_inference_return_predictions_for_binary_classification(X_train, y_train):
    trained_model = train_model(X_train, y_train)
    preds = inference(trained_model, X_train)

    assert type(preds) == np.ndarray
    np.testing.assert_array_equal(np.unique(preds), [0, 1])
