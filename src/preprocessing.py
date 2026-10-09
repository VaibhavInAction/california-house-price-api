"""Split and prepare the data for training."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into X_train, X_test, y_train, y_test (80% train / 20% test)."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
