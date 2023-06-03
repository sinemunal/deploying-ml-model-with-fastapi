# Script to record model performance on a given slice.
import pandas as pd

from src.data_processing import process_data
from src.model import inference
from src.model_helpers import load_model, compute_model_metrics_on_slices


def _calculate_inference(test_df, target_name, cat_features):
    # Load trained model, encoder and binarizer
    trained_model = load_model("rf_model.pickle")
    encoder = load_model("encoder.pickle")
    lb = load_model("label_binarizer.pickle")
    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=cat_features,
        label=target_name,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Get predictions
    y_preds = inference(trained_model, X_test)

    return y_test, y_preds


# Settings related to data
feature_name = "sex"
target_name = "salary"  # target column that we classify
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Load census data for test
test = pd.read_csv("data/test.csv")


y_test, y_preds = _calculate_inference(test, target_name, cat_features)

# Calculate performance metrics with respect to gender feature
result_df = compute_model_metrics_on_slices(y_test, y_preds, test[feature_name])
print("Output of Model Performance: \n", result_df)
