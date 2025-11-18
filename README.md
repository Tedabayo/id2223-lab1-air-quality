# KTH ID2223 Lab 1 — Air Quality Prediction

This project builds an end-to-end pipeline for predicting PM2.5 using Hopsworks Feature Store and GitHub Actions.
Files include feature pipeline, training pipeline, and batch inference pipeline, all running automatically.

The model (XGBoost) is trained using a Feature View that joins weather and air quality data.
A hindcast script (`pm25_hindcast.py`) evaluates the model on historical data and saves the plot in `dashboards/`. Much of this is not totally understood as i use LLM for help and understanding

## To run locally:

```bash
pip install -r requirements.txt
python3 pm25_hindcast.py
```

