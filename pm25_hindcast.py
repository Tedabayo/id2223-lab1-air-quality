import os

import hopsworks
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# --- CONFIG (matches your actual project) ---

FEATURE_VIEW_NAME = "air_quality_fv"
FEATURE_VIEW_VERSION = 1

AIR_QUALITY_FG_NAME = "air_quality"
AIR_QUALITY_FG_VERSION = 1

MODEL_NAME = "air_quality_model"   # from your training logs
MODEL_VERSION = None               # None = latest version


def load_model(project):
    """
    Load the trained sklearn pipeline (preprocessor + XGBoost)
    from Hopsworks Model Registry.
    """
    mr = project.get_model_registry()

    if MODEL_VERSION is None:
        model = mr.get_model(MODEL_NAME)
        print(f"Loaded latest version of model '{MODEL_NAME}': version={model.version}")
    else:
        model = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
        print(f"Loaded model '{MODEL_NAME}' version={MODEL_VERSION}")

    model_dir = model.download()
    print(f"Model artifacts downloaded to: {model_dir}")

    # This must match joblib.dump(...) in your training_pipeline.py
    # You logged "xgboost_pipeline.pkl" there.
    model_path = os.path.join(model_dir, "xgboost_pipeline.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model file 'xgboost_pipeline.pkl' in {model_dir}.\n"
            f"Check your training_pipeline.py for the exact filename used in joblib.dump()."
        )

    pipeline = joblib.load(model_path)
    print("Model pipeline loaded successfully.")
    return pipeline


def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # ---------------------------------------------------------
    # 1. Read features from Feature View
    # ---------------------------------------------------------
    print(f"\nReading Feature View '{FEATURE_VIEW_NAME}', version={FEATURE_VIEW_VERSION}...")
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

    # In your setup, get_batch_data() returns a single DataFrame (features only)
    X = fv.get_batch_data()
    print(f"Feature View rows: {len(X)}")
    print("Feature columns:", list(X.columns))

    if "city" not in X.columns or "date" not in X.columns:
        raise ValueError(
            "Expected 'city' and 'date' columns in feature view features.\n"
            f"Available columns: {list(X.columns)}"
        )

    # Ensure 'date' is datetime
    X["date"] = pd.to_datetime(X["date"])

    # ---------------------------------------------------------
    # 2. Read labels from air_quality Feature Group
    # ---------------------------------------------------------
    print(f"\nReading air quality Feature Group '{AIR_QUALITY_FG_NAME}', version={AIR_QUALITY_FG_VERSION}...")
    aq_fg = fs.get_feature_group(AIR_QUALITY_FG_NAME, version=AIR_QUALITY_FG_VERSION)
    aq_df = aq_fg.read()
    print(f"air_quality rows: {len(aq_df)}")
    print("air_quality columns:", list(aq_df.columns))

    # We expect 'pm2_5' as the label and 'city','date' to join on
    expected_cols = ["city", "date", "pm2_5"]
    for col in expected_cols:
        if col not in aq_df.columns:
            raise ValueError(
                f"Expected column '{col}' in air_quality dataframe.\n"
                f"Available columns: {list(aq_df.columns)}"
            )

    aq_df["date"] = pd.to_datetime(aq_df["date"])

    # ---------------------------------------------------------
    # 3. Merge features (X) with labels (air_quality pm2_5) on (city, date)
    # ---------------------------------------------------------
    print("\nMerging features with actual pm2_5 labels on (city, date)...")
    merged = pd.merge(
        X,
        aq_df[["city", "date", "pm2_5"]],
        on=["city", "date"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "Merged dataframe is empty.\n"
            "No overlapping (city, date) rows between Feature View and air_quality FG.\n"
            "Check that both use the same city names and date resolution."
        )

    print(f"Merged rows: {len(merged)}")
    merged = merged.sort_values("date")

    # ---------------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------------
    print("\nLoading model from Model Registry...")
    pipeline = load_model(project)

    # ---------------------------------------------------------
    # 5. Prepare features for prediction
    #    (mirror your training: drop non-feature columns)
    # ---------------------------------------------------------
    drop_cols = ["city", "date", "weather_date", "weather_city", "pm2_5"]
    cols_to_drop = [c for c in drop_cols if c in merged.columns]

    X_features = merged.drop(columns=cols_to_drop)
    print("Columns used for prediction:", list(X_features.columns))

    # ---------------------------------------------------------
    # 6. Predict and build hindcast dataframe
    # ---------------------------------------------------------
    print("Generating predictions for hindcast...")
    y_pred = pipeline.predict(X_features)

    hindcast_df = pd.DataFrame({
        "date": merged["date"],
        "pm2_5_actual": merged["pm2_5"],
        "pm2_5_pred": y_pred,
    })

    # ---------------------------------------------------------
    # 7. Plot hindcast graph
    # ---------------------------------------------------------
    print("Plotting hindcast graph...")

    plt.figure(figsize=(12, 6))
    plt.plot(hindcast_df["date"], hindcast_df["pm2_5_actual"], label="Actual PM2.5")
    plt.plot(hindcast_df["date"], hindcast_df["pm2_5_pred"], label="Predicted PM2.5")
    plt.xlabel("Date")
    plt.ylabel("PM2.5")
    plt.title("Hindcast: Actual vs Predicted PM2.5")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("dashboards", exist_ok=True)
    output_path = os.path.join("dashboards", "pm25_hindcast.png")
    plt.savefig(output_path)
    plt.close()

    print(f"\nHindcast saved to: {output_path}")


if __name__ == "__main__":
    main()
