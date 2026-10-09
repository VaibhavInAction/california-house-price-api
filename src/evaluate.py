"""Measure how good a trained model is."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test) -> dict:
    """Return MAE, RMSE and R2 of the model on the test set."""
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
    }
