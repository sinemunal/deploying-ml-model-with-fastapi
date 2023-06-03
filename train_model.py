# Script to train machine learning model.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Load census data and format columns.
from src.data_processing import format_data, process_data
from src.model import train_model
from src.model_helpers import save_model, load_model

data = pd.read_csv("data/census.csv")
format_data(data)
target = "salary"  # target column that we classify

# Split data into train and test with stratification due to class imbalance.
train, test = train_test_split(
    data, test_size=0.20, random_state=42, shuffle=True, stratify=data[target]
)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=target, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label=target,
    training=False,
    encoder=encoder,
    lb=lb,
)
# Save the encoders
save_model(encoder, "encoder.pickle")
save_model(lb, "label_binarizer.pickle")

# Save train and test data
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

# Train and save a model.
trained_model = train_model(X_train, y_train, is_optimized=True)

save_model(trained_model, "rf_model.pickle")
