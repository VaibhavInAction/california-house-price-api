"""FastAPI app that serves predictions.

Run from the project root:  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict

app = FastAPI(title="California House Price API")


class HouseFeatures(BaseModel):
    MedInc: float       # median income in the block (in $10,000s)
    HouseAge: float     # median house age in years
    AveRooms: float     # average rooms per household
    AveBedrms: float    # average bedrooms per household
    Population: float   # block population
    AveOccup: float     # average people per household
    Latitude: float
    Longitude: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(house: HouseFeatures):
    value = predict(house.model_dump())
    return {
        "predicted_value": round(value, 4),        # in $100,000s, same as the dataset
        "predicted_price_usd": round(value * 100_000),
    }
