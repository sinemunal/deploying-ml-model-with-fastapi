import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib


def save_model(model, filename):
    """
    Saves model with a given filename
    Inputs
    ------
    model : scikit-learn model
        Trained machine learning model.
    filename: str
        Name of the model
    Returns
    -------
        None
    """
    joblib.dump(model, f"../model/{filename}")


def load_model(filename):
    """
    Load trained model
    Inputs
    ------
    filename: str
        Name of the model
    Returns
    -------
    model :
        Trained machine learning model.
    """
    return joblib.load(f"../model/{filename}")


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def compute_model_metrics_on_slices(y, preds, feature):
    """
    Calculate performance metrics for each category given in
    feature column.
    Parameters
    ----------
    y: np.array
        Known labels, binarized.
    preds: np.array
        Predicted labels, binarized.
    feature: pd.Series
        Corresponding feature column with categorical values.
    Returns
    -------
    Dataframe where rows are the categories and columns are the
    calculated performance metrics for each category.
    """
    categories = list(feature.unique())
    metrics_df = pd.DataFrame(
        columns=["precision", "recall", "fbeta"], index=categories
    )
    for cat in categories:
        mask = feature == cat
        y_sliced = y[mask]
        preds_sliced = preds[mask]
        metrics_df.loc[cat, :] = compute_model_metrics(y_sliced, preds_sliced)

    return metrics_df
