"""
Batch Inference Pipeline for ID2223 Lab 1

This script:
1. Connects to Hopsworks Feature Store and Model Registry
2. Loads the 'air_quality_fv' Feature View
3. Loads the latest 'air_quality_model' from the Model Registry
4. Fetches recent feature data
5. Runs batch predictions for pm2_5
6. Creates a hindcast plot (true vs predicted if available) and saves it as a PNG
"""

import os
from datetime import datetime, timedelta

import hopsworks
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
FEATURE_VIEW_NAME = "air_quality_fv"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "air_quality_model"
MODEL_VERSION = 3  #just for training


# Number of past days to show in the hindcast
HINDCAST_DAYS = 14

# Where to save the dashboard PNG
OUTPUT_DIR = "dashboards"
OUTPUT_PNG_PATH = os.path.join(OUTPUT_DIR, "pm25_hindcast.png")


# ---------------------------------------------------------------------
# HOPSWORKS CONNECTION
# ---------------------------------------------------------------------
def connect_to_feature_store_and_registry():
    """
    Connect to Hopsworks using HOPSWORKS_API_KEY.
    Returns (feature_store, model_registry, project).
    """
    hopsworks_api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not hopsworks_api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable is not set.\n"
            "→ In GitHub Actions: set it as a repository secret.\n"
            "→ Locally: export HOPSWORKS_API_KEY='your-key-here'."
        )

    project = hopsworks.login(api_key_value=hopsworks_api_key)
    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()
    return feature_store, model_registry, project


# ---------------------------------------------------------------------
# LOAD FEATURE VIEW & MODEL
# ---------------------------------------------------------------------
def load_feature_view(feature_store):
    """
    Load the Feature View used for training.
    """
    feature_view = feature_store.get_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
    )
    print(f" Loaded Feature View '{FEATURE_VIEW_NAME}', version {FEATURE_VIEW_VERSION}.")
    return feature_view


def load_registered_model(model_registry):
    """
    Load the latest / specified version of the 'air_quality_model'
    from the Model Registry and return the trained sklearn Pipeline.
    """
    if MODEL_VERSION is not None:
        model = model_registry.get_model(MODEL_NAME, version=MODEL_VERSION)
    else:
        model = model_registry.get_model(MODEL_NAME, version=None)  # latest

    print(f" Loaded model '{MODEL_NAME}', version {model.version} from Model Registry.")

    model_dir = model.download()
    model_file_path = os.path.join(model_dir, "xgboost_pipeline.pkl")
    trained_pipeline = joblib.load(model_file_path)

    return trained_pipeline


# ---------------------------------------------------------------------
# BATCH DATA + HINDCAST
# ---------------------------------------------------------------------
def get_recent_batch_data(feature_view) -> pd.DataFrame:
    """
    Fetch recent feature data from the Feature View for hindcast evaluation.
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=HINDCAST_DAYS)

    print(f" Fetching batch data from Feature View between {start_time} and {end_time}...")

    batch_df = feature_view.get_batch_data(start_time=start_time, end_time=end_time)

    print(f"   Retrieved {len(batch_df)} rows for hindcast.")
    print("   Batch columns:", list(batch_df.columns))
    return batch_df


def create_hindcast_plot(hindcast_df: pd.DataFrame, output_path: str):
    """
    Create and save a hindcast plot.

    If true pm2_5 values are available (column 'pm2_5' in hindcast_df),
    we plot both true and predicted.
    Otherwise, we plot only predictions.
    """
    if "date" not in hindcast_df.columns:
        raise ValueError("Expected a 'date' column in hindcast_df.")

    sorted_df = hindcast_df.sort_values("date")

    plt.figure(figsize=(10, 5))

    # Plot true values only if they exist
    if "pm2_5" in sorted_df.columns:
        plt.plot(sorted_df["date"], sorted_df["pm2_5"], label="True pm2_5")

    # Always plot predictions
    plt.plot(
        sorted_df["date"],
        sorted_df["pm2_5_pred"],
        label="Predicted pm2_5",
        linestyle="--",
    )

    plt.xlabel("Date")
    plt.ylabel("pm2_5")
    plt.title("PM2.5 Hindcast (True vs Predicted / Predicted only)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f" Hindcast plot saved to: {output_path}")


# ---------------------------------------------------------------------
# MAIN BATCH INFERENCE PIPELINE
# ---------------------------------------------------------------------
def run_batch_inference_pipeline():
    """
    1. Connect to Hopsworks
    2. Load Feature View and model
    3. Fetch recent batch data
    4. Run predictions
    5. Build hindcast DataFrame
    6. Generate dashboard PNG
    """
    (
        feature_store,
        model_registry,
        project,
    ) = connect_to_feature_store_and_registry()

    feature_view = load_feature_view(feature_store)
    trained_pipeline = load_registered_model(model_registry)

    batch_df = get_recent_batch_data(feature_view)

    if batch_df.empty:
        print(
            "⚠️ No data available for the selected hindcast window. "
            "Make sure your feature pipeline has inserted enough days."
        )
        return

    # Try to detect a pm2_5 label column automatically.
    # In some setups it might be called 'pm2_5', in others 'air_quality_pm2_5', etc.
    pm_cols = [c for c in batch_df.columns if "pm2_5" in c]
    label_col = pm_cols[0] if pm_cols else None

    if label_col is None:
        print(
            "⚠️ No 'pm2_5' label column found in batch_df – will only plot predictions.\n"
            f"    Available columns: {list(batch_df.columns)}"
        )
    else:
        print(f"   Using label column: {label_col}")

    # Use similar feature logic as training: drop non-feature columns.
    # These were problematic or not used as numeric features.
    drop_columns = ["city", "date", "weather_date"]
    if label_col is not None:
        drop_columns.append(label_col)

    feature_columns = [col for col in batch_df.columns if col not in drop_columns]

    print("   Feature columns used for prediction:", feature_columns)

    X_batch = batch_df[feature_columns]

    print("\n🚀 Running batch predictions...")
    y_pred = trained_pipeline.predict(X_batch)

    # Build hindcast dataframe for visualization
    if "date" not in batch_df.columns:
        raise ValueError("Expected 'date' column in batch_df.")

    hindcast_df = batch_df[["date"]].copy()
    hindcast_df["pm2_5_pred"] = y_pred

    if label_col is not None:
        # Rename the label column nicely to 'pm2_5' for plotting
        hindcast_df["pm2_5"] = batch_df[label_col].values

    print("\n Sample of hindcast results:")
    print(hindcast_df.tail())

    # Create hindcast plot (dashboard PNG)
    create_hindcast_plot(hindcast_df, OUTPUT_PNG_PATH)

    print("\n Batch inference pipeline completed successfully.")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_batch_inference_pipeline()
