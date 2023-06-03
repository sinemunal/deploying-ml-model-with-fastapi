from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "random_state": [42],
    "class_weight": ["balanced"],
}


def train_model(X_train, y_train, is_optimized=False):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    is_optimized: boolean
        When true GridSearchCV is applied for hyperparameter tuning.
    Returns
    -------
    model
        Trained machine learning model.
    """

    if is_optimized is False:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
    else:
        model_cv = GridSearchCV(
            RandomForestClassifier(),
            param_grid=RF_PARAM_GRID,
            cv=StratifiedKFold(n_splits=3),
        )
        model_cv.fit(X_train, y_train)
        model = model_cv.best_estimator_
        print(model_cv.best_params_)
    return model


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
