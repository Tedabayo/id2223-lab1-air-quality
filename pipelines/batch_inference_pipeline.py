"""
Batch Inference Pipeline for Air Quality Prediction (PM2.5) - Task 4

This script:
1. Connects to Hopsworks (Model Registry and Feature Store).
2. Downloads the latest trained model.
3. Retrieves the latest weather forecast features (next 7 days).
4. Predicts PM2.5 levels for the forecast period.
5. Generates a dashboard plot (PNG) of the predictions.
"""
import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
FEATURE_VIEW_NAME = "air_quality_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "air_quality_model"
MODEL_VERSION = 1
TARGET_CITY_NAME = "stockholm"
OUTPUT_DASHBOARD_PATH = "air_quality_dashboard.png"
FORECAST_DAYS = 7 # Predict for the next 7 days

# ---------------------------------------------------------------------
# HOPSWORKS CONNECTION
# ---------------------------------------------------------------------
def connect_to_feature_store_and_registry():
    """Connect to Hopsworks using the API key."""
    hopsworks_api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY environment variable is not set.")

    # CRITICAL STABILITY FIX: Pause to prevent immediate Kafka timeout
    print("Pausing briefly before Hopsworks login...")
    time.sleep(2)
    project = hopsworks.login(api_key_value=hopsworks_api_key)
    
    # CRITICAL STABILITY FIX: Pause after login
    print("Pausing for 5 seconds to stabilize Hopsworks connection...")
    time.sleep(5)
    
    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()
    return feature_store, model_registry, project


# ---------------------------------------------------------------------
# DATA & MODEL RETRIEVAL
# ---------------------------------------------------------------------
def load_model(mr):
    """Downloads and loads the latest registered model."""
    print(f"Downloading model '{MODEL_NAME}', version {MODEL_VERSION}...")
    
    # 1. Download Model from Registry
    model = mr.get_model(name=MODEL_NAME, version=MODEL_VERSION)
    model_dir = model.download()
    
    # 2. Load the pipeline object using joblib
    model_path = os.path.join(model_dir, "xgboost_pipeline.pkl")
    trained_pipeline = joblib.load(model_path)
    
    print("Model loaded successfully.")
    return trained_pipeline
# this is AI generated as i could noit get it to work properly. 
def get_forecast_features(fs):
    """
    Retrieves the future weather features for prediction from the Feature View.
    CRITICAL FIX: Forces date strings without time components to resolve Hopsworks format error.
    """
    print(f"Retrieving forecast features from Feature View {FEATURE_VIEW_NAME}...")
    
    # Calculate the time range needed: starting from TOMORROW 
    today = pd.Timestamp(datetime.now().date())
    
    # FIX 1: Shift start time to TOMORROW and convert to simple date string (YYYY-MM-DD)
    start_time = (today + timedelta(days=1)).date().isoformat()
    
    # FIX 2: Calculate the end time 7 days ahead and add two extra days for safety, 
    # and convert to simple date string.
    end_time = (today + timedelta(days=FORECAST_DAYS + 2)).date().isoformat() 

    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    
    # Get batch data for the forecast time range
    feature_vector_df = fv.get_batch_data(
        start_time=start_time,
        end_time=end_time,
        excluded_labels=True # Only retrieve features, no labels needed for forecast
    )

    if feature_vector_df.empty:
        raise RuntimeError(
            f"No forecast features found in Feature View for the range {start_time} to {end_time}. "
            "Ensure the feature pipeline pushed the 7-day forecast."
        )

    # The model expects specific features based on how it was trained (Batch 3)
    feature_columns = [
        "weather_wind_speed_max",
        "weather_wind_direction_dominant",
        "weather_wind_gusts_max",
        "weather_temperature_max",
    ]
    
    forecast_dates = feature_vector_df["date"].copy()
    
    # Select only the features the model needs (replicating the feature set from the training pipeline)
    X_forecast = feature_vector_df.loc[:, feature_columns]
    
    # Since we queried a larger window, we only take the 7 days needed for the prediction
    X_forecast = X_forecast.head(FORECAST_DAYS)
    forecast_dates = forecast_dates.head(FORECAST_DAYS)
    
    print(f"Retrieved {len(X_forecast)} forecast days.")
    return X_forecast, forecast_dates


# ---------------------------------------------------------------------
# VISUALIZATION (Dashboard Generation)
# ---------------------------------------------------------------------
def generate_dashboard(predictions, dates):
    """
    Plots the predicted PM2.5 levels for the forecast period and saves it as a PNG.
    """
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    # Plot the PM2.5 Predictions
    plt.plot(dates, predictions, marker='o', linestyle='-', color='indigo', label='Predicted PM2.5')
    
    # Add Air Quality Index (AQI) thresholds as horizontal regions
    # Good: 0-49 (Green background)
    plt.axhspan(0, 50, color='green', alpha=0.1, label='Good (0-49)')
    # Moderate: 50-99 (Yellow background)
    plt.axhspan(50, 100, color='yellow', alpha=0.1, label='Moderate (50-99)')
    # Unhealthy for Some: 100-149 (Red background)
    plt.axhspan(100, 150, color='red', alpha=0.1, label='Unhealthy (100-149)')

    
    plt.title(f"PM2.5 Forecast for {TARGET_CITY_NAME} (Next {FORECAST_DAYS} Days)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("PM2.5 Concentration", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')
    
    # Ensure y-axis covers the typical AQI range for clarity
    max_pred = predictions.max() if not predictions.empty else 50
    plt.ylim(0, max(150, max_pred * 1.2))

    plt.tight_layout()
    plt.savefig(OUTPUT_DASHBOARD_PATH)
    print(f"\n Dashboard saved successfully to {OUTPUT_DASHBOARD_PATH}")


# ---------------------------------------------------------------------
# MAIN BATCH INFERENCE PIPELINE
# ---------------------------------------------------------------------
def run_batch_inference_pipeline():
    """
    Executes the entire batch inference process.
    """
    print("Starting batch inference pipeline...")
    try:
        # 1. Connect to Hopsworks
        feature_store, model_registry, project = connect_to_feature_store_and_registry()
        
        # 2. Load Model
        trained_pipeline = load_model(model_registry)
        
        # 3. Get Forecast Features
        X_forecast, forecast_dates = get_forecast_features(feature_store)
        
        # 4. Predict
        print(f"Making {len(X_forecast)} predictions...")
        # The model pipeline handles preprocessing (OneHotEncoding) automatically
        predictions = trained_pipeline.predict(X_forecast)
        
        # 5. Generate Dashboard
        predictions_series = pd.Series(predictions, index=forecast_dates)
        generate_dashboard(predictions_series, forecast_dates)
        
        print("\n Batch Inference Pipeline completed successfully.\n")

    except RuntimeError as e:
        print(f"Pipeline failed due to configuration or data error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during inference: {e}")


if __name__ == "__main__":
    run_batch_inference_pipeline()