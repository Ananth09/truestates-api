# main.py
import os
import pickle
from functools import lru_cache
from typing import Optional, Dict, Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from statsmodels.nonparametric.smoothers_lowess import lowess
import io

# ------------------------------
# NEW BASE DIRECTORY (repo-relative)
# ------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__))
print(f"Using BASE_DIR: {BASE_DIR}")

AREA_DIR = os.path.join(BASE_DIR, "area_models_rf")
COLUMNS_DIR = os.path.join(BASE_DIR, "training_columns")

FORECAST_CSV = os.path.join(BASE_DIR, "Sarima_forecast_6M.csv")
HISTORIC_CSV = os.path.join(BASE_DIR, "historical_df.csv")
print(f"Using AREA_DIR: {AREA_DIR}")
print(f"Using COLUMNS_DIR: {COLUMNS_DIR}")
print(f"Using FORECAST_CSV: {FORECAST_CSV}")
print(f"Using HISTORIC_CSV: {HISTORIC_CSV}")

LOWESS_FRAC = 0.03

app = FastAPI(title="truEstates - Area Price Predictor", version="1.0")

# ------------------------------
# Input Model
# ------------------------------
class PredictionInput(BaseModel):
    area_name_en: str
    procedure_area: Optional[float] = None
    has_parking: Optional[int] = None
    floor_bin: Optional[str] = None
    rooms_en: Optional[str] = None
    swimming_pool: Optional[int] = None
    balcony: Optional[int] = None
    elevator: Optional[int] = None
    metro: Optional[int] = None

# ------------------------------
# Helpers for area names
# ------------------------------
def _normalize_area(area_name: str) -> str:
    return area_name.replace("_", " ").strip()

# ------------------------------
# Cached loaders for model & columns
# ------------------------------
@lru_cache(maxsize=128)
def load_columns(area_name_en: str) -> List[str]:
    area_name = _normalize_area(area_name_en)

    if not os.path.isdir(COLUMNS_DIR):
        raise FileNotFoundError(f"training_columns directory not found at {COLUMNS_DIR}")

    for f in os.listdir(COLUMNS_DIR):
        fname = f.lower()
        target1 = f"model_columns_{area_name}.pkl".lower()
        target2 = f"model_columns_{area_name.replace(' ', '_')}.pkl".lower()

        if fname == target1 or fname == target2:
            with open(os.path.join(COLUMNS_DIR, f), "rb") as file:
                return pickle.load(file)

    raise FileNotFoundError(f"model_columns file not found for area: {area_name}")

@lru_cache(maxsize=128)
def load_model(area_name_en: str):
    area_name = _normalize_area(area_name_en)

    if not os.path.isdir(AREA_DIR):
        raise FileNotFoundError(f"area_models_rf directory not found at {AREA_DIR}")

    for f in os.listdir(AREA_DIR):
        fname = f.lower()
        target1 = f"rf_model_{area_name}.pkl".lower()
        target2 = f"rf_model_{area_name.replace(' ', '_')}.pkl".lower()

        if fname == target1 or fname == target2:
            with open(os.path.join(AREA_DIR, f), "rb") as file:
                return pickle.load(file)

    raise FileNotFoundError(f"Model file not found for area: {area_name}")

# ------------------------------
# Core Predict Function
# ------------------------------
def predict_with_area(input_data: Dict[str, Any]) -> pd.DataFrame:
    forecast_df = pd.read_csv(FORECAST_CSV)
    historic_df = pd.read_csv(HISTORIC_CSV)
    print(f"Loaded forecast_df with shape: {forecast_df.shape}")
    print(f"Loaded historic_df with shape: {historic_df.shape}")

    area = input_data["area_name_en"]
    print(f"Predicting for area: {area}")

    train_columns = load_columns(area)
    model = load_model(area)

    temp = pd.DataFrame([input_data])
    temp["area_name_en"] = area
    temp = pd.get_dummies(temp)


    for col in train_columns:
        if col not in temp.columns:
            temp[col] = 0

    temp = temp[train_columns]

    predicted_price = float(model.predict(temp)[0])

    forecast_area = forecast_df[forecast_df["area_name_en"] == area].copy()
    print(f"Forecast area before processing shape: {forecast_area.shape}")
    if not forecast_area.empty:
        forecast_area["median_price"] = predicted_price * forecast_area["growth_factor"]
        forecast_area = forecast_area[["month", "median_price"]]
        print(f"Forecast area shape: {forecast_area.shape}")

    historic_area = historic_df[historic_df["area_name_en"] == area].copy()
    print(f"Historic area before processing shape: {historic_area.shape}")

    if not historic_area.empty:
        historic_area = historic_area.sort_values("month")
        x = historic_area.index.values
        smoothed = lowess(historic_area["median_price"], x, frac=LOWESS_FRAC, return_sorted=False)
        historic_area["median_price"] = smoothed

        if not forecast_area.empty:
            historic_area.loc[historic_area.index[-1], "median_price"] = forecast_area.iloc[0]["median_price"]

    final_df = pd.concat([historic_area[["month", "median_price"]], forecast_area], ignore_index=True)

    return final_df

# ------------------------------
# API Endpoints
# ------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        data = payload.dict()
        extras = data.pop("extras", None)
        if extras:
            for k, v in extras.items():
                if k not in data:
                    data[k] = v

        df = predict_with_area(data)
        return {"area": _normalize_area(payload.area_name_en), "result": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv")
def predict_csv(payload: PredictionInput):
    data = payload.dict()
    extras = data.pop("extras", None)
    if extras:
        for k, v in extras.items():
            if k not in data:
                data[k] = v

    df = predict_with_area(data)

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction.csv"}
    )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
