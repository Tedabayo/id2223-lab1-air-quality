# KTH ID2223 Lab 1 — Air Quality Prediction

This project predicts PM2.5 using weather data and the Hopsworks Feature Store.

## Feature Pipelines
- Backfill pipeline loads historical PM2.5 and weather data into Hopsworks.
- Daily pipeline loads yesterday’s data and 7-day weather forecast.

## Training Pipeline
Creates a Feature View, splits data, trains an XGBoost model, and saves it to the Hopsworks Model Registry.

## Batch Inference Pipeline
Loads the model and predicts PM2.5 for the coming days.
Saves a dashboard plot here:
dashboards/pm25_hindcast.png

## Hindcast Evaluation
pm25_hindcast.py compares predicted vs actual PM2.5 and saves a graph.

## Running Locally
pip install -r requirements.txt
export HOPSWORKS_API_KEY=your_key
python3 pm25_hindcast.py

## Repository Link
https://github.com/Tedabayo/id2223-lab1-air-quality
