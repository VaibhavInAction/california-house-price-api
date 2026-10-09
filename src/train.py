"""Train models and save the best one to models/best_model.pkl.

Run from the project root:  python -m src.train
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data import load_data
from src.evaluate import evaluate
from src.preprocessing import split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"


def train() -> dict:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        m = results[name]
        print(f"{name:<18} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}")

    # The best model is the one with the highest R2
    best_name = max(results, key=lambda name: results[name]["r2"])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH, compress=3)
    print(f"\nBest model: {best_name} -> saved to {MODEL_PATH}")
    return results


if __name__ == "__main__":
    train()
