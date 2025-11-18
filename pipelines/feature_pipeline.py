"""
Daily Feature Pipeline for ID2223 Lab 1

This script:
1. Connects to Hopsworks using HOPSWORKS_API_KEY
2. Fetches historical data (3 days) AND future forecast data (7 days)
3. Creates or retrieves Feature Groups: 'weather' and 'air_quality'
4. Inserts all collected rows into their respective Feature Groups
"""

import os
from datetime import date, timedelta
import time
import pandas as pd
import hopsworks
import requests


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
TARGET_CITY_NAME = "stockholm"          # example: "stockholm"
TARGET_LATITUDE = 59.3293
TARGET_LONGITUDE = 18.0686
TARGET_SENSOR_ID = "your_sensor_id"     
TARGET_TIMEZONE = "UTC"                 
AQ_CSV_PATH = "data/aqicn_pm25_stockholm_small.csv"
FORECAST_DAYS = 7 # Number of days to fetch forecast for


# ---------------------------------------------------------------------
# Hopsworks connection
# ---------------------------------------------------------------------
def connect_to_feature_store():
    """Connect to Hopsworks and return Feature Store handle."""
    hopsworks_api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not hopsworks_api_key:
        raise RuntimeError("HOPSWORKS_API_KEY environment variable is not set.")

    # Stability fix: Wait before connecting
    time.sleep(2)
    project = hopsworks.login(api_key_value=hopsworks_api_key)
    feature_store = project.get_feature_store()
    return feature_store


# ---------------------------------------------------------------------
# REAL DATA FUNCTIONS — Open-Meteo
# ---------------------------------------------------------------------
def get_weather_features_for_range(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch weather features for a given date range using the Open-Meteo API.
    Handles multiple dates in a single robust API call.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": TARGET_LATITUDE,
        "longitude": TARGET_LONGITUDE,
        "daily": ",".join(
            [
                "wind_speed_10m_max",
                "wind_gusts_10m_max",
                "wind_direction_10m_dominant",
                "temperature_2m_max",
            ]
        ),
        "timezone": TARGET_TIMEZONE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    # --- RETRY LOGIC (Max stability for GitHub Actions) ---
    max_retries = 10
    delay_seconds = 10 

    for attempt in range(max_retries):
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            break 
        except requests.exceptions.ReadTimeout as e:
            if attempt < max_retries - 1:
                print(f"[API] Timeout (Attempt {attempt + 1}). Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
                delay_seconds *= 2 
            else:
                print(f"[API] Final attempt failed after {max_retries} retries.")
                raise e
    # --- END RETRY LOGIC ---

    data = resp.json()

    daily = data.get("daily", {})
    if not daily or len(daily.get("time", [])) == 0:
        raise RuntimeError(f"No weather data returned by Open-Meteo for range {start_date} to {end_date}")

    # Convert the daily dictionary response into a DataFrame
    weather_df = pd.DataFrame({
        "city": TARGET_CITY_NAME,
        "date": pd.to_datetime(daily["time"]),
        "wind_speed_max": daily["wind_speed_10m_max"],
        "wind_direction_dominant": [str(d) for d in daily["wind_direction_10m_dominant"]],
        "wind_gusts_max": daily["wind_gusts_10m_max"],
        "temperature_max": daily["temperature_2m_max"],
    })
    
    return weather_df.dropna() # Drop any rows where data is missing

# ---------------------------------------------------------------------
# AIR QUALITY DATA (CSV) - No change needed here
# ---------------------------------------------------------------------

_AQ_DF_CACHE = None
def load_air_quality_csv() -> pd.DataFrame:
    """Load and cache the AQ CSV data."""
    global _AQ_DF_CACHE
    if _AQ_DF_CACHE is None:
        if not os.path.exists(AQ_CSV_PATH):
            raise FileNotFoundError(f"AQ CSV file not found at {AQ_CSV_PATH}.")
        raw = pd.read_csv(AQ_CSV_PATH, comment="#")
        if "City" not in raw.columns or "Specie" not in raw.columns:
            raise ValueError("Expected columns 'City' and 'Specie' not found in the AQ CSV.")
        mask_city = raw["City"].astype(str).str.lower() == "stockholm"
        mask_spec = raw["Specie"].astype(str).str.lower() == "pm25"
        df = raw[mask_city & mask_spec].copy()
        if df.empty:
            raise ValueError("No rows found in the AQ CSV for Stockholm and pm25.")
        if "Date" not in df.columns or "median" not in df.columns:
            raise ValueError("Missing 'Date' or 'median' column in AQ CSV.")
        df["date"] = pd.to_datetime(df["Date"])
        df["pm2_5"] = df["median"].astype(float)
        df["date_only"] = df["date"].dt.date
        _AQ_DF_CACHE = (df[["date", "date_only", "pm2_5"]].sort_values("date").reset_index(drop=True))
    return _AQ_DF_CACHE


def get_air_quality_features_for_date(input_date: date) -> pd.DataFrame:
    """Look up pm2_5 for this date in the aqicn CSV."""
    df = load_air_quality_csv()
    subset = df[df["date_only"] == input_date]

    if subset.empty:
        # For historical data, this is an error.
        raise ValueError(
            f"No pm2_5 value found in {AQ_CSV_PATH} for date {input_date}. Cannot run pipeline."
        )

    pm25_value = float(subset.iloc[0]["pm2_5"])

    aq_row = {
        "city": TARGET_CITY_NAME,
        "date": pd.to_datetime(input_date),
        "pm2_5": pm25_value,
    }
    return pd.DataFrame([aq_row])


# ---------------------------------------------------------------------
# MAIN PIPELINE LOGIC
# ---------------------------------------------------------------------
def run_daily_feature_pipeline(num_days_backfill: int = 3):
    """
    Main function for the daily feature pipeline.
    Fetches historical data (num_days_backfill) AND forecast data (FORECAST_DAYS).
    """
    today_date = date.today()

    # 1. DEFINE HISTORICAL DATES
    # Historical dates: yesterday, day before yesterday, etc.
    historical_dates = [
        today_date - timedelta(days=offset)
        for offset in range(1, num_days_backfill + 1)
    ]
    
    # 2. DEFINE FORECAST DATE RANGE
    # Forecast starts today and goes forward 7 days
    forecast_start = today_date
    forecast_end = today_date + timedelta(days=FORECAST_DAYS)
    
    print(f" Running feature pipeline for {TARGET_CITY_NAME}")
    print(f"   Fetching {num_days_backfill} days of historical data + {FORECAST_DAYS}-day forecast.")
    print("   Historical Dates:", historical_dates)
    print(f"   Forecast Range: {forecast_start} to {forecast_end}")

    # 3. COLLECT HISTORICAL AIR QUALITY ROWS (One call per date needed for AQ CSV lookup)
    air_quality_rows = []
    for d in historical_dates:
        # We only need AQ for historical days because forecast AQ is what we predict
        air_quality_rows.append(get_air_quality_features_for_date(d))
        time.sleep(1) # Stability sleep

    # 4. COLLECT ALL WEATHER DATA (Historical + Forecast) in one large API call
    # This is more robust than 90 individual calls
    all_weather_df = get_weather_features_for_range(
        start_date=historical_dates[-1], # Oldest historical date
        end_date=forecast_end             # Furthest forecast date
    )
    weather_features_df = all_weather_df

    # 5. CONCATENATE AIR QUALITY ROWS
    air_quality_features_df = pd.concat(air_quality_rows, ignore_index=True)

    # 6. Final Dataframe Cleanup
    weather_features_df["wind_direction_dominant"] = (
        weather_features_df["wind_direction_dominant"].astype(str)
    )

    print("\n Weather features (sample, includes forecast):")
    print(weather_features_df.head())
    print(weather_features_df.tail())

    print("\n Air quality features (sample, historical):")
    print(air_quality_features_df.head())

    # Connect to Hopsworks Feature Store
    feature_store = connect_to_feature_store()

    # CRITICAL STABILITY FIX: Pause after login, before insertion
    print("Pausing for 5 seconds to stabilize Hopsworks connection...")
    time.sleep(5)

    # Create or get Feature Groups
    weather_feature_group = feature_store.get_or_create_feature_group(
        name="weather",
        version=1,
        description="Daily weather features (Historical + Forecast)",
        primary_key=["city"],
        event_time="date",
    )

    air_quality_feature_group = feature_store.get_or_create_feature_group(
        name="air_quality",
        version=1,
        description="Daily PM2.5 air quality measurements (Historical label)",
        primary_key=["city"],
        event_time="date",
    )

    # Insert data (multiple days at once)
    print(
        f"\n Inserting {len(weather_features_df)} weather rows "
        f"into Feature Group 'weather' (Includes {FORECAST_DAYS} forecast days)..."
    )
    weather_feature_group.insert(weather_features_df)

    print(
        f"⬆Inserting {len(air_quality_features_df)} air quality rows "
        f"into Feature Group 'air_quality' (Historical labels)..."
    )
    # The air quality feature group should only contain historical labels (pm2_5)
    air_quality_feature_group.insert(air_quality_features_df)

    print("\n Feature pipeline completed successfully.\n")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Run the feature pipeline for 3 historical days + 7 forecast days
    run_daily_feature_pipeline(num_days_backfill=3)