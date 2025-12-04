import os
import pickle
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# ------------------------------
# 1️⃣ BASE DIRECTORY – MODELS PER AREA
# ------------------------------
# C:\Users\anant\OneDrive\Desktop\truEstates\Modular_code with 18 areas_RF\area_models_rf
BASE_DIR = os.path.join(
    os.path.expanduser("~"),
    "OneDrive",
    "Desktop",
    "truEstates",
    "Modular_code with 18 areas_RF"
)
    # "Modular_Code",
    # "Modular_code with 18 areas_RF"

AREA_DIR = os.path.join(BASE_DIR, "area_models_rf") 
COLUMNS_DIR = os.path.join(BASE_DIR, "training_columns") 
FORECAST_CSV = os.path.join(BASE_DIR, "Sarima_forecast_6M.csv")
HISTORIC_CSV = os.path.join(BASE_DIR, "historical_df.csv") 
# ------------------------------
# 2️⃣ LOAD TRAINING COLUMNS
# ------------------------------
def load_columns(area_name_en):
    area_name_en = area_name_en.replace("_", " ").strip()
    for f in os.listdir(COLUMNS_DIR):
        if f.lower() == f"model_columns_{area_name_en}.pkl".lower() or \
           f.lower() == f"model_columns_{area_name_en.replace(' ', '_')}.pkl".lower():
            with open(os.path.join(COLUMNS_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"❌ model_columns file not found for area '{area_name_en}'")

# ------------------------------
# 3️⃣ LOAD MODEL
# ------------------------------
def load_model(area_name_en):
    area_name_en = area_name_en.replace("_", " ").strip()
    for f in os.listdir(AREA_DIR):
        if f.lower() == f"rf_model_{area_name_en}.pkl".lower() or \
           f.lower() == f"rf_model_{area_name_en.replace(' ', '_')}.pkl".lower():
            with open(os.path.join(AREA_DIR, f), "rb") as file:
                return pickle.load(file)
    raise FileNotFoundError(f"❌ Model file not found for area '{area_name_en}'")

# ------------------------------
# 4️⃣ PREDICTION FUNCTION
# ------------------------------
def predict_with_area(input_data):
    lowess_frac=0.03
    forecast_df = pd.read_csv(FORECAST_CSV)
    historic_df = pd.read_csv(HISTORIC_CSV)
    area = input_data["area_name_en"].replace("_", " ").strip()

    # Step 1: Load model + expected columns
    train_columns = load_columns(area)
    model = load_model(area)

    # Step 2: One-hot encode input
    temp = pd.DataFrame([input_data])
    temp["area_name_en"] = area
    temp = pd.get_dummies(temp)
    for col in train_columns:
        if col not in temp.columns:
            temp[col] = 0
    temp = temp[train_columns]

    # Step 3: Predict median price
    predicted_price = model.predict(temp)[0]
    print("Raw Model Prediction:", predicted_price)

    # Step 4: Prepare forecast dataframe for area
    forecast_area = forecast_df[forecast_df["area_name_en"].replace("_", " ") == area].copy()
    forecast_area["median_price"] = predicted_price * forecast_area["growth_factor"]
    forecast_area = forecast_area[["month", "median_price"]]

    # Step 5: Prepare historic dataframe for area
    historic_area = historic_df[historic_df["area_name_en"].replace("_", " ") == area].copy()

    # Step 6: Apply LOWESS smoothing on historic using index as x
    if not historic_area.empty:
        historic_area = historic_area.sort_values("month").reset_index(drop=True)
        x = historic_area.index.values  # use index for LOWESS
        smoothed = lowess(
            endog=historic_area["median_price"].values,
            exog=x,
            frac=lowess_frac
        )
        historic_area["median_price"] = smoothed[:, 1]

        # Replace last historic value with first forecast median price
        if not forecast_area.empty:
            historic_area.loc[historic_area.index[-1], "median_price"] = forecast_area.iloc[0]["median_price"]

    # Step 7: Combine historic + forecast
    final_df = pd.concat([historic_area[["month", "median_price"]], forecast_area], ignore_index=True)
    final_df = final_df.reset_index(drop=True)

    return final_df


input_data = {
    "area_name_en":  'Al Hebiah Fourth',
    "procedure_area": 70,
    "has_parking": 1,
    "floor_bin": "11-20",
    "rooms_en": "1BR",
    "swimming_pool": 1,
    "balcony": 1,
    "elevator": 1,
    "metro" : 0
}

final_df = predict_with_area(input_data)

print(final_df.tail(10))