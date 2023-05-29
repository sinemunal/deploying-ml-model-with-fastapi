import pandas as pd
import pytest
import numpy as np

from src.model_helpers import compute_model_metrics, compute_model_metrics_on_slices


@pytest.fixture
def y_test():
    return np.array([1, 1, 1, 0])


@pytest.fixture
def y_preds():
    return np.array([0, 0, 1, 0])


@pytest.fixture
def feature_test():
    return pd.Series(["male", "female", "male", "female"])


def test_compute_model_metrics_calculates_precision_recall_fbeta(
        y_test, y_preds
):
    result = compute_model_metrics(y_test, y_preds)
    expected_result = (1.0, 0.3333333333333333, 0.5)

    assert result == expected_result


def test_compute_model_metrics_on_slices_gives_performance_metrics_for_each_category(
        y_test, y_preds, feature_test
):
    result_df = compute_model_metrics_on_slices(y_test, y_preds, feature_test)
    result_df = result_df.astype(float)
    expected_df = pd.DataFrame(
        index=["male", "female"],
        data={"precision": [1.0, 1.0], "recall": [0.5, 0.0], "fbeta": [0.666667, 0.0]}

    )

    pd.testing.assert_frame_equal(result_df, expected_df)
