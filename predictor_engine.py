import pandas as pd
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
from statsmodels.nonparametric.smoothers_lowess import lowess

# --- CONFIGURATION CONSTANTS ---
# OHE_FILE = 'C:/Users/anant/OneDrive/Desktop/truEstates/Modular_forecast_code/onehot_encoder.pkl'
# COLUMNS_FILE = 'C:/Users/anant/OneDrive/Desktop/truEstates/Modular_forecast_code/train_columns.pkl'
# TRAINING_DATA_FILE = 'C:/Users/anant/OneDrive/Desktop/truEstates/Modular_forecast_code/df_trained_dataset_6000.csv'
# FORECAST_DATA_FILE = 'C:/Users/anant/OneDrive/Desktop/truEstates/Modular_forecast_code/Sarima_forecast_6M.csv'
LOESS_FRAC = 0.1
LOESS_IT = 3
DATE_FORMAT = '%d-%m-%Y' # Standard DD-MM-YYYY format, like 01-01-2025
# BASE_MODEL_DIR = "C:/Users/anant/OneDrive/Desktop/truEstates/Modular_forecast_code/area_models/"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OHE_FILE = os.path.join(BASE_DIR, "onehot_encoder.pkl")
COLUMNS_FILE = os.path.join(BASE_DIR, "train_columns.pkl")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "df_trained_dataset_6000.csv")
FORECAST_DATA_FILE = os.path.join(BASE_DIR, "Sarima_forecast_6M.csv")

BASE_MODEL_DIR = os.path.join(BASE_DIR, "area_models")
AREA_FILES = [os.path.join(BASE_MODEL_DIR, filename) for filename in AREA_MODEL_FILENAMES]





class PropertyPricePredictor:
    """
    A modular class to load models and data, prepare inputs, predict property
    prices, and perform historical trend and future forecast analysis.
    """

    # Base directory for all area model files

    # Just the filenames (only change these in future)
    AREA_MODEL_FILENAMES = [
        "dt_model_Al_Barsha_South_Fifth.pkl",
        "dt_model_Al_Barsha_South_Fourth.pkl",
        "dt_model_Al_Barshaa_South_Third.pkl",
        "dt_model_Al_Hebiah_Fourth.pkl",
        "dt_model_Al_Khairan_First.pkl",
        "dt_model_Al_Merkadh.pkl",
        "dt_model_Al_Thanyah_Fifth.pkl",
        "dt_model_Al_Warsan_First.pkl",
        "dt_model_Al_Yelayiss_2.pkl",
        "dt_model_Bukadra.pkl",
        "dt_model_Burj_Khalifa.pkl",
        "dt_model_Business_Bay.pkl",
        "dt_model_Hadaeq_Sheikh_Mohammed_Bin_Rashid.pkl",
        "dt_model_Jabal_Ali_First.pkl",
        "dt_model_Madinat_Al_Mataar.pkl",
        "dt_model_Madinat_Dubai_Almelaheyah.pkl",
        "dt_model_Marsa_Dubai.pkl",
        "dt_model_Me'Aisem_First.pkl",
        "dt_model_Nadd_Hessa.pkl",
        "dt_model_Wadi_Al_Safa_5.pkl"
    ]

    # Build full paths cleanly
    AREA_FILES = [BASE_MODEL_DIR + filename for filename in AREA_MODEL_FILENAMES]


    def __init__(self):
        self.area_models = self._load_area_models()
        self.ohe, self.train_columns = self._load_encoder_and_columns()
        self.train_data = self._load_training_data()
        self.growth_pivot = self._load_forecasting_data()

        if not self.area_models:
            print("❌ WARNING: No area models loaded. Prediction will fail.")
        if self.ohe is None or self.train_columns is None:
            print("❌ WARNING: Encoder or training columns failed to load.")

    def _load_area_models(self) -> dict:
        loaded_models = {}
        missing_models = []
        for model_file in self.AREA_FILES:
            filename = os.path.basename(model_file)   # extract only the file name
            area_name = (
                filename
                .replace('dt_model_', '')
                .replace('.pkl', '')
                .replace('_', ' ')
                .strip())
            try:
                # Mock loading since actual files are not present in this context
                # To make this runnable in a real environment, files must exist
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                with open(model_file, 'rb') as f:
                    loaded_models[area_name] = pickle.load(f)
            except FileNotFoundError:
                missing_models.append(model_file)
            except Exception as e:
                print(f"❌ Error loading {model_file}: {e}")
        if missing_models:
            print(f"⚠️ Missing models: {len(missing_models)} files could not be found. (This is expected in a sandbox environment without the actual files.)")
        return loaded_models

    def _load_encoder_and_columns(self):
        ohe = None
        train_columns = None
        try:
            # Mock loading since actual files are not present in this context
            if not os.path.exists(OHE_FILE):
                raise FileNotFoundError(f"OHE file not found: {OHE_FILE}")
            with open(OHE_FILE, 'rb') as f:
                ohe = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading One-Hot Encoder: {e}")
        try:
            # Mock loading since actual files are not present in this context
            if not os.path.exists(COLUMNS_FILE):
                raise FileNotFoundError(f"Columns file not found: {COLUMNS_FILE}")
            with open(COLUMNS_FILE, 'rb') as f:
                train_columns = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading Training Columns: {e}")
        return ohe, train_columns

    def _load_training_data(self) -> pd.DataFrame:
        try:
            # Mock loading since actual files are not present in this context
            if not os.path.exists(TRAINING_DATA_FILE):
                raise FileNotFoundError(f"Training data file not found: {TRAINING_DATA_FILE}")

            train_data = pd.read_csv(TRAINING_DATA_FILE)
            train_data['instance_date'] = pd.to_datetime(train_data['instance_date'])
            return train_data
        except Exception as e:
            print(f"❌ Could not load training data for trend analysis: {e}")
            return None

    def _load_forecasting_data(self) -> pd.DataFrame:
        try:
            # Mock loading since actual files are not present in this context
            if not os.path.exists(FORECAST_DATA_FILE):
                raise FileNotFoundError(f"Forecast data file not found: {FORECAST_DATA_FILE}")

            growth_df = pd.read_csv(FORECAST_DATA_FILE)
            return growth_df
        except Exception as e:
            print(f"❌ Error loading forecasting data: {e}")
            return None

    def prepare_input_data(self, area, rooms, floor, pool, balcony_val, elevator_val, metro_val, parking, area_size):
        input_data = pd.DataFrame({
            'rooms_en': [rooms], 'floor_bin': [floor], 'swimming_pool': [pool],
            'balcony': [balcony_val], 'elevator': [elevator_val], 'metro': [metro_val],
            'has_parking': [parking], 'area_name_en': [area], 'procedure_area': [area_size]
        })
        area_name = input_data['area_name_en'].iloc[0]
        input_no_area = input_data.drop(columns=['area_name_en'])
        cat_cols = ['rooms_en', 'floor_bin']
        if self.ohe is None or self.train_columns is None:
            return None, None, None
        try:
            X_cat = self.ohe.transform(input_no_area[cat_cols])
            feature_names = self.ohe.get_feature_names_out(cat_cols)
            X_cat_df = pd.DataFrame(X_cat.toarray() if hasattr(X_cat, 'toarray') else X_cat, columns=feature_names)
            X_numerical = input_no_area.drop(columns=cat_cols)
            X_processed = pd.concat([X_numerical.reset_index(drop=True), X_cat_df.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(f"❌ Error in encoding input: {e}")
            return None, None, None
        final_X = pd.DataFrame(0, index=X_processed.index, columns=self.train_columns)
        for col in X_processed.columns:
             if col in final_X.columns:
                 final_X[col] = X_processed[col]
        return final_X, area_name, input_data

    # Modified filter to ONLY filter by area name (Tier 3)
    def filter_training_data_by_area_only(self, train_data, area_name):
        if train_data is None:
            return pd.DataFrame()

        # Filter only by area name, ignoring all other property features
        filtered_data = train_data[train_data['area_name_en'] == area_name].copy()

        return filtered_data

    # Combined trend calculation, now specialized for Area Trend
    def calculate_area_trend(self, filtered_data):
        """Calculates LOESS trend for the entire area, returning a formatted DataFrame or None."""
        TREND_TYPE = 'Historical Trend (Entire Area)'

        if filtered_data is None or len(filtered_data) < 2:
            return pd.DataFrame({'Month': [], 'Median Price': [], 'Type': []})

        filtered = filtered_data.copy()
        filtered['instance_date'] = pd.to_datetime(filtered['instance_date'])
        filtered['year_month'] = filtered['instance_date'].dt.to_period('M')
        monthly_data = filtered.groupby('year_month')['meter_sale_price'].agg(['median', 'count']).reset_index()
        monthly_data = monthly_data.rename(columns={'median': 'meter_sale_price', 'count': 'data_points'})
        monthly_data['timestamp'] = monthly_data['year_month'].dt.to_timestamp()
        monthly_data = monthly_data.sort_values('timestamp').reset_index(drop=True)

        if len(monthly_data) < 2:
            return pd.DataFrame({'Month': [], 'Median Price': [], 'Type': []})

        try:
            monthly_data['num_index'] = np.arange(len(monthly_data))
            y_values = monthly_data['meter_sale_price'].values
            x_values = monthly_data['num_index'].values
            loess_smoothed = lowess(y_values, x_values, frac=LOESS_FRAC, it=LOESS_IT)
            trend_indices = loess_smoothed[:, 0].astype(int)

            # --- FIX APPLIED: Ensure consistent DD-MM-YYYY format ---
            trend_df = pd.DataFrame({
                # Since 'timestamp' is the 1st of the month, this will result in 01-MM-YYYY
                'Month': monthly_data['timestamp'].iloc[trend_indices].dt.strftime(DATE_FORMAT).values,
                'Median Price': loess_smoothed[:, 1],
                'Type': TREND_TYPE
            })
            # Add temporary key for sorting before dropping it
            trend_df['Sort_Key'] = pd.to_datetime(trend_df['Month'], format=DATE_FORMAT)
            return trend_df.sort_values('Sort_Key').drop(columns=['Sort_Key'])
        except Exception as e:
            print(f"❌ Error during LOESS calculation ({TREND_TYPE}): {e}")
            return pd.DataFrame({'Month': [], 'Median Price': [], 'Type': []})


    def prepare_forecast_data(self, area_name):
        if self.growth_pivot is None:
            return None
        area_growth = self.growth_pivot[self.growth_pivot['area_name_en'] == area_name]
        if area_growth.empty:
            return None
        periods = area_growth['month'].unique()
        forecast_data = {}
        for period in periods:
            period_data = area_growth[area_growth['month'] == period].iloc[0]
            forecast_data[period] = {
                'main': period_data['growth_factor'],
                'upper': period_data['growth_factor_upper'],
                'lower': period_data['growth_factor_lower']
            }
        return forecast_data

    def predict_and_analyze(self, selected_area, rooms_en, floor_bin, swimming_pool, balcony, elevator, metro, has_parking, procedure_area) -> pd.DataFrame:

        X_input, area_name, _ = self.prepare_input_data(
            selected_area, rooms_en, floor_bin, swimming_pool, balcony,
            elevator, metro, has_parking, procedure_area
        )

        if X_input is None:
            return pd.DataFrame({'Error': ["Failed to prepare input data."]})

        if area_name not in self.area_models:
            return pd.DataFrame({'Error': [f"No model found for area: {area_name}"]})

        model = self.area_models[area_name]

        try:
            predicted_price = model.predict(X_input)[0]
        except Exception as e:
            return pd.DataFrame({'Error': [f"Model prediction failed: {e}"]})

        # --- 1. Historical Trend Dataframe (ONLY Entire Area Trend) ---
        area_data = self.filter_training_data_by_area_only(self.train_data, area_name)
        historical_df = self.calculate_area_trend(area_data)


        # --- 2. Future Forecast Dataframe (Non-Cumulative) ---
        forecast_data = self.prepare_forecast_data(area_name)
        forecast_table_data = []

        # Assuming prediction point is always the 1st of the month
        prediction_date = datetime(2025, 8, 1)

        if forecast_data:
            # Add the prediction date as the starting point
            # --- FIX APPLIED: Ensure consistent DD-MM-YYYY format ---
            forecast_table_data.append({
                 'Month': prediction_date.strftime(DATE_FORMAT),
                 'Median Price': predicted_price,
                 'Type': 'Prediction Point'
            })

            sorted_periods = sorted(forecast_data.keys())
            for period in sorted_periods:
                growth_factors = forecast_data[period]
                forecasted_price_main = predicted_price * growth_factors['main']

                # Format forecast period correctly
                clean_period = str(period).strip()
                try:
                    # Convert YYYY-MM to datetime object (defaults to 1st of the month)
                    date_object = datetime.strptime(clean_period, '%Y-%m')
                    # --- FIX APPLIED: Ensure consistent DD-MM-YYYY format ---
                    formatted_period = date_object.strftime(DATE_FORMAT)
                except ValueError:
                    formatted_period = clean_period # Fallback if format is unexpected

                forecast_table_data.append({
                    'Month': formatted_period,
                    'Median Price': forecasted_price_main,
                    'Type': 'Future Forecast'
                })

        forecast_df = pd.DataFrame(forecast_table_data)


        # --- 3. Combine DataFrames and Sort ---

        # Concatenate Historical (Entire Area) and Future Forecast
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)

        # Final sort by date
        # Use the DATE_FORMAT for parsing the Month column
        combined_df['sort_key'] = pd.to_datetime(combined_df['Month'], format=DATE_FORMAT, errors='coerce')
        combined_df = combined_df.sort_values(by='sort_key', na_position='last').drop(columns='sort_key').reset_index(drop=True)
        
        return combined_df
    



    #execution example
if __name__ == "__main__":
    
    # 1. Instantiate the predictor engine
    engine = PropertyPricePredictor()

    print("\n" + "="*70)
    print("🚀 Running Analysis for Sample Input: Al Barsha South Fourth (2 B/R, 60 sqMt)")
    print("="*70)
    
    # --- Define Input Features ---
    selected_area = 'Business Bay'
    rooms_en = '3 B/R'             
    floor_bin = '11-20'            
    swimming_pool = 0              # 1 for Yes, 0 for No
    balcony = 0
    elevator = 1
    metro = 1
    has_parking = 1
    procedure_area = 67 # sqMt    

    # 2. Call the method and receive the combined DataFrame
    results_df = engine.predict_and_analyze(
        selected_area, rooms_en, floor_bin, swimming_pool, balcony, 
        elevator, metro, has_parking, procedure_area
    )
