from fastapi import FastAPI
from pydantic import BaseModel
from predictor_engine import PropertyPricePredictor
import pandas as pd

app = FastAPI(title="TruEstates Property Price API", version="1.0")

# Initialize your engine once at startup
engine = PropertyPricePredictor()


# --------- Request Schema ---------
class PredictionRequest(BaseModel):
    selected_area: str
    rooms_en: str
    floor_bin: str
    swimming_pool: int
    balcony: int
    elevator: int
    metro: int
    has_parking: int
    procedure_area: float


# --------- API Endpoint ---------
@app.post("/predict")
def predict(req: PredictionRequest):

    results_df = engine.predict_and_analyze(
        req.selected_area,
        req.rooms_en,
        req.floor_bin,
        req.swimming_pool,
        req.balcony,
        req.elevator,
        req.metro,
        req.has_parking,
        req.procedure_area
    )

    # Convert DataFrame → JSON
    return {
        "status": "success",
        "rows": len(results_df),
        "results": results_df.to_dict(orient="records")
    }
