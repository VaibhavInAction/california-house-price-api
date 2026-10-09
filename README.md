# ML Prediction API

Predicts California house prices with a scikit-learn model, served through a FastAPI REST API.

Two models are trained and compared, and the best one is saved and used by the API.

## Results

Evaluated on a held-out 20% test set (4,128 houses). Prices are in units of $100,000.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.533 | 0.746 | 0.576 |
| **Random Forest** (selected) | **0.328** | **0.505** | **0.805** |

## Dataset

[California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn: 20,640 block groups from the 1990 US census, with 8 numeric features and the median house value as the target. It is downloaded automatically, so nothing needs to be in `data/`.

## Project structure

```
ml-prediction-api/
├── api/
│   └── main.py            # FastAPI app: /health and /predict
├── data/                  # Raw data (not committed)
├── models/                # Trained model, created by src/train.py (not committed)
├── notebook/              # Exploration and experiments
├── src/
│   ├── data.py            # Load the dataset
│   ├── preprocessing.py   # Train/test split
│   ├── train.py           # Train models, save the best one
│   ├── evaluate.py        # MAE, RMSE, R²
│   └── predict.py         # Load the model and predict
├── tests/
│   └── test_api.py        # API tests
├── pytest.ini
└── requirements.txt
```

## Setup

Requires Python 3.12.

```bash
git clone <repo-url>
cd ml-prediction-api
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Usage

**1. Train the model** (creates `models/best_model.pkl`):

```bash
python -m src.train
```

**2. Start the API:**

```bash
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs for interactive documentation.

**3. Run the tests:**

```bash
pytest
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/predict` | Predicts the value of one house |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.984, "AveBedrms": 1.024,
       "Population": 322, "AveOccup": 2.556, "Latitude": 37.88, "Longitude": -122.23}'
```

Response:

```json
{"predicted_value": 4.2658, "predicted_price_usd": 426579}
```

| Field | Description |
|---|---|
| `MedInc` | Median income in the block group (in $10,000s) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household members |
| `Latitude` | Latitude |
| `Longitude` | Longitude |

## Tech stack

Python · pandas · scikit-learn · FastAPI · uvicorn · pytest
