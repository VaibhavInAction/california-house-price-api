"""Load the saved model and make predictions."""

from functools import lru_cache

import joblib
import pandas as pd

from src.data import FEATURES
from src.train import MODEL_PATH


@lru_cache  # load the file only once, then reuse it
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> float:
    """Predict the house value (in $100,000s) for one house.

    `features` is a dict with the 8 keys listed in src.data.FEATURES.
    """
    row = pd.DataFrame([features], columns=FEATURES)
    return float(load_model().predict(row)[0])
