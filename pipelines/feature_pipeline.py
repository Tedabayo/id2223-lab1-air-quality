"""
Daily Feature Pipeline for ID2223 Lab 1

This script:
1. Connects to Hopsworks using HOPSWORKS_API_KEY
2. Fetches real weather (Open-Meteo) and air quality data (from a CSV)
   for a range of past days
3. Creates or retrieves Feature Groups: 'weather' and 'air_quality'
4. Inserts one row per day into each Feature Group
"""

import os
from datetime import date, timedelta

import pandas as pd
import hopsworks
import requests
import time





# ---------------------------------------------------------------------
# CONFIGURATION — CHANGE THESE TO MATCH YOUR SENSOR AND CITY
# ---------------------------------------------------------------------
TARGET_CITY_NAME = "stockholm"          # example: "stockholm"

# Coordinates for Stockholm city center (tweak if you want a different spot)
TARGET_LATITUDE = 59.3293
TARGET_LONGITUDE = 18.0686

TARGET_SENSOR_ID = "your_sensor_id"     # kept for compatibility, unused here
TARGET_TIMEZONE = "UTC"                 # used for Open-Meteo

# Path to your aqicn CSV with historical PM2.5 values.
# In your case this is the WAQI COVID-19 dataset:
#   Date,Country,City,Specie,count,min,max,median,variance
#
# We will convert it to a simple [date, pm2_5] format inside load_air_quality_csv().
AQ_CSV_PATH = "data/aqicn_pm25_stockholm_small.csv"




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
# REAL DATA FUNCTIONS — Open-Meteo + aqicn CSV
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# REAL DATA FUNCTIONS — Open-Meteo + aqicn CSV
# ---------------------------------------------------------------------
def get_weather_features_for_date(input_date: date) -> pd.DataFrame:
    """
    Fetch daily weather features for a given date using the Open-Meteo API.
    Includes robust retry logic for network stability.
    """
    base_url = "https://api.open-meteo.com/v1/forecast"
    date_str = input_date.isoformat()
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
        "start_date": date_str,
        "end_date": date_str,
    }

    # --- ADDED RETRY LOGIC (The definitive fix for GitHub timeouts) ---
    max_retries = 5
    delay_seconds = 5 # Initial delay

    for attempt in range(max_retries):
        try:
            # Use the high 90 second timeout
            resp = requests.get(base_url, params=params, timeout=90)
            resp.raise_for_status()
            # If successful, break the loop and proceed to data processing
            break 
        except requests.exceptions.ReadTimeout as e:
            if attempt < max_retries - 1:
                # Wait longer on each failure
                print(f"[{date_str}] Timeout (Attempt {attempt + 1}). Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
                delay_seconds *= 2 
            else:
                # If all attempts fail, raise the error
                print(f"[{date_str}] Final attempt failed after {max_retries} retries.")
                raise e
    # --- END RETRY LOGIC ---

    data = resp.json()
    # ... (rest of the data extraction is the same) ...

    daily = data.get("daily", {})
    if not daily or len(daily.get("time", [])) == 0:
        raise RuntimeError(f"No weather data returned by Open-Meteo for {date_str}")

    idx = 0
    wind_speed_max = daily["wind_speed_10m_max"][idx]
    wind_gusts_max = daily["wind_gusts_10m_max"][idx]
    wind_dir_dom = daily["wind_direction_10m_dominant"][idx]
    temperature_max = daily["temperature_2m_max"][idx]

    weather_row = {
        "city": TARGET_CITY_NAME,
        "date": pd.to_datetime(input_date),
        "wind_speed_max": wind_speed_max,
        "wind_direction_dominant": str(wind_dir_dom),
        "wind_gusts_max": wind_gusts_max,
        "temperature_max": temperature_max,
    }

    return pd.DataFrame([weather_row])

_AQ_DF_CACHE = None


def load_air_quality_csv() -> pd.DataFrame:
    """
    Load the WAQI COVID-19 CSV and convert it into the format expected
    by the rest of the pipeline.

    Input CSV columns (example):
      Date,Country,City,Specie,count,min,max,median,variance

    We:
      - Filter to City == 'Stockholm' (case-insensitive)
      - Filter to Specie == 'pm25'   (case-insensitive)
      - Create:
          date      : datetime64[ns] parsed from 'Date'
          pm2_5     : float from 'median'
          date_only : date for matching

    Output cached DataFrame columns:
      date        (datetime64[ns])
      date_only   (datetime.date)
      pm2_5       (float)
    """
    global _AQ_DF_CACHE
    if _AQ_DF_CACHE is None:
        if not os.path.exists(AQ_CSV_PATH):
            raise FileNotFoundError(
                f"AQ CSV file not found at {AQ_CSV_PATH}. "
                f"Place your aqicn CSV under resources/ or update AQ_CSV_PATH."
            )

        # Read raw CSV, skipping comment lines at the top
        raw = pd.read_csv(AQ_CSV_PATH, comment="#")

        # Be robust to capitalization in the CSV
        if "City" not in raw.columns or "Specie" not in raw.columns:
            raise ValueError(
                "Expected columns 'City' and 'Specie' not found in the AQ CSV. "
                f"Columns present: {list(raw.columns)}"
            )

        # Filter to Stockholm + PM2.5, case-insensitive
        mask_city = raw["City"].astype(str).str.lower() == "stockholm"
        mask_spec = raw["Specie"].astype(str).str.lower() == "pm25"
        df = raw[mask_city & mask_spec].copy()

        if df.empty:
            raise ValueError(
                "No rows found in the AQ CSV for City == 'Stockholm' and Specie == 'pm25'."
            )

        if "Date" not in df.columns:
            raise ValueError("Expected a 'Date' column in the AQ CSV but did not find it.")

        if "median" not in df.columns:
            raise ValueError("Expected a 'median' column in the AQ CSV but did not find it.")

        # Create proper datetime column called 'date' from 'Date'
        df["date"] = pd.to_datetime(df["Date"])

        # Create pm2_5 from one of the statistic columns (here: 'median').
        df["pm2_5"] = df["median"].astype(float)

        # Helper column for lookups by date
        df["date_only"] = df["date"].dt.date

        # Keep only what we need and cache it
        _AQ_DF_CACHE = (
            df[["date", "date_only", "pm2_5"]]
            .sort_values("date")
            .reset_index(drop=True)
        )

    return _AQ_DF_CACHE


def get_air_quality_features_for_date(input_date: date) -> pd.DataFrame:
    """
    Look up pm2_5 for this date in the aqicn CSV.

    Returns a DataFrame with columns:
      city, date, pm2_5
    """
    df = load_air_quality_csv()
    subset = df[df["date_only"] == input_date]

    if subset.empty:
        raise ValueError(
            f"No pm2_5 value found in {AQ_CSV_PATH} for date {input_date}."
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
def run_daily_feature_pipeline(num_days_backfill: int = 90):
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
        # This pause prevents the API from getting overwhelmed and timing out.
        time.sleep(1)


    weather_features_df = pd.concat(weather_rows, ignore_index=True)
    air_quality_features_df = pd.concat(air_quality_rows, ignore_index=True)

    # 🔑 Make absolutely sure schema matches Hopsworks FG:
    # wind_direction_dominant must be string
    weather_features_df["wind_direction_dominant"] = (
        weather_features_df["wind_direction_dominant"].astype(str)
    )

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
        f"⬆Inserting {len(air_quality_features_df)} air quality rows "
        f"into Feature Group 'air_quality'..."
    )
    air_quality_feature_group.insert(air_quality_features_df)

    print("\n Feature pipeline completed successfully.\n")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # You can change the number of days here if you want
    run_daily_feature_pipeline(num_days_backfill=90)
