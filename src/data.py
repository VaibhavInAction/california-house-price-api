"""Load the California housing dataset."""

import pandas as pd
from sklearn.datasets import fetch_california_housing

# The 8 input columns, in the order the model expects them
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "target"


def load_data() -> pd.DataFrame:
    """Return the dataset as one DataFrame: 8 feature columns + "target"."""
    housing = fetch_california_housing(as_frame=True)
    return housing.frame.rename(columns={"MedHouseVal": TARGET})
