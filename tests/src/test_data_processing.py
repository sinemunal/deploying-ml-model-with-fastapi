import pandas as pd
import numpy as np
import pytest
import sklearn

from src.data_processing import format_data, process_data


@pytest.fixture
def df():
    return pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"], "target": [">50", "<=50"]})


def test_format_data_removes_white_spaces_from_columns():
    df = pd.DataFrame({" col1": [1, 2], " col2": ["a", "b"]})
    expected_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

    format_data(df)

    pd.testing.assert_frame_equal(df, expected_df)


def test_process_data_encodes_cat_and_creates_binary_labels(df):
    X, y, encoder, lb = process_data(
        df,
        categorical_features=["col2"],
        label="target",
        training=True,
        encoder=None,
        lb=None,
    )
    X_expected = np.array([[1.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
    y_expected = np.array([1, 0])

    np.testing.assert_array_equal(X, X_expected)
    np.testing.assert_array_equal(y, y_expected)
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer


def test_process_data_needs_encoder_and_labelizer_when_training_is_false(df):
    with pytest.raises(AttributeError):
        process_data(
            df,
            categorical_features=["col2"],
            label="target",
            training=False,
            encoder=None,
            lb=None,
        )
