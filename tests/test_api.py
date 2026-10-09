"""Tests for the prediction API.

Run from the project root:  pytest
(Train the model first:  python -m src.train)
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# First row of the dataset (real value: 4.526)
HOUSE = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984127, "AveBedrms": 1.02381,
    "Population": 322.0, "AveOccup": 2.555556, "Latitude": 37.88, "Longitude": -122.23,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_price():
    response = client.post("/predict", json=HOUSE)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["predicted_value"] < 6   # dataset values range from 0.15 to 5.0


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in HOUSE.items() if k != "MedInc"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422      # FastAPI's "invalid input" error
