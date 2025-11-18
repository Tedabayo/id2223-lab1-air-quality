"""
Daily Feature Pipeline for ID2223 Lab 1

This script:
1. Connects to Hopsworks using HOPSWORKS_API_KEY
2. Fetches (placeholder) weather and air quality data for a range of past days
3. Creates or retrieves Feature Groups: 'weather' and 'air_quality'
4. Inserts one row per day into each Feature Group
"""

import os
from datetime import date, timedelta
import pandas as pd
import hopsworks


# ---------------------------------------------------------------------
# CONFIGURATION — CHANGE THESE TO MATCH YOUR SENSOR AND CITY
# ---------------------------------------------------------------------
TARGET_CITY_NAME = "stockholm"          # example: "stockholm"
TARGET_SENSOR_ID = "your_sensor_id"     # example: "stockholm-us-embassy"
TARGET_TIMEZONE = "UTC"                 # not used yet, but helpful later


# ---------------------------------------------------------------------
# Hopsworks connection
# ---------------------------------------------------------------------
def connect_to_feature_store():
    """
    Connect to Hopsworks using the API key stored in environment variables.
    Returns a Feature Store handle.
    """
    hopsworks_api_key = os.environ.get("HOPSWORKS_API_KEY")

    if not hopsworks_api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable is not set.\n"
            "→ In GitHub Actions: Set it as a repository secret.\n"
            "→ Locally: export HOPSWORKS_API_KEY='your-key-here'"
        )

    project = hopsworks.login(api_key_value=hopsworks_api_key)
    feature_store = project.get_feature_store()
    return feature_store


# ---------------------------------------------------------------------
# PLACEHOLDER FUNCTIONS — Replace these with real API calls
# ---------------------------------------------------------------------
def get_weather_features_for_date(input_date: date) -> pd.DataFrame:
    """
    Placeholder that returns a dummy weather row.
    Later, replace with:
    - Open-Meteo weather API call
    """
    dummy_weather_data = {
        "city": [TARGET_CITY_NAME],
        "date": [pd.to_datetime(input_date)],
        "wind_speed_max": [12.3],
        "wind_direction_dominant": ["N"],
        "wind_gusts_max": [20.5],
        "temperature_max": [18.7],
    }

    return pd.DataFrame(dummy_weather_data)


def get_air_quality_features_for_date(input_date: date) -> pd.DataFrame:
    """
    Placeholder that returns a dummy pm2_5 row.
    Later, replace with:
    - aqicn API call
    - OR your downloaded CSV of historical AQI values
    """
    dummy_air_quality_data = {
        "city": [TARGET_CITY_NAME],
        "date": [pd.to_datetime(input_date)],
        "pm2_5": [5.3],  # Replace with real value
    }

    return pd.DataFrame(dummy_air_quality_data)


# ---------------------------------------------------------------------
# MAIN PIPELINE LOGIC
# ---------------------------------------------------------------------
def run_daily_feature_pipeline(num_days_backfill: int = 30):
    """
    Main function for the daily feature pipeline.

    By default, it backfills the last `num_days_backfill` days
    (yesterday, day before, etc.), so that the Feature Groups and
    Feature View have enough rows for training.

    Steps:
      - Build a list of past dates
      - Fetch weather + air quality features for each date
      - Insert into their respective Feature Groups
    """
    today_date = date.today()

    # Build list of dates: yesterday, day before yesterday, ..., N days back
    dates_to_process = [
        today_date - timedelta(days=offset)
        for offset in range(1, num_days_backfill + 1)
    ]

    print(f" Running feature pipeline for {TARGET_CITY_NAME}")
    print(f"   Backfilling last {num_days_backfill} days:")
    print("   Dates:", dates_to_process)

    # Collect all rows into DataFrames
    weather_rows = []
    air_quality_rows = []

    for d in dates_to_process:
        weather_rows.append(get_weather_features_for_date(d))
        air_quality_rows.append(get_air_quality_features_for_date(d))

    weather_features_df = pd.concat(weather_rows, ignore_index=True)
    air_quality_features_df = pd.concat(air_quality_rows, ignore_index=True)

    print("\n Weather features (sample):")
    print(weather_features_df.head())

    print("\n Air quality features (sample):")
    print(air_quality_features_df.head())

    # Connect to Hopsworks Feature Store
    feature_store = connect_to_feature_store()

    # Create or get Feature Groups
    weather_feature_group = feature_store.get_or_create_feature_group(
        name="weather",
        version=1,
        description="Daily weather features",
        primary_key=["city"],
        event_time="date",
    )

    air_quality_feature_group = feature_store.get_or_create_feature_group(
        name="air_quality",
        version=1,
        description="Daily PM2.5 air quality measurements",
        primary_key=["city"],
        event_time="date",
    )

    # Insert data (multiple days at once)
    print(
        f"\n Inserting {len(weather_features_df)} weather rows "
        f"into Feature Group 'weather'..."
    )
    weather_feature_group.insert(weather_features_df)

    print(
        f"⬆️ Inserting {len(air_quality_features_df)} air quality rows "
        f"into Feature Group 'air_quality'..."
    )
    air_quality_feature_group.insert(air_quality_features_df)

    print("\n Feature pipeline completed successfully.\n")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # You can change the number of days here if you want
    run_daily_feature_pipeline(num_days_backfill=30)
