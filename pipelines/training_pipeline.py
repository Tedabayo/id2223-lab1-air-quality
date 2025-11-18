"""
Training Pipeline for ID2223 Lab 1

This script:
1. Connects to the Hopsworks Feature Store
2. Joins the 'air_quality' and 'weather' Feature Groups into a Feature View
3. Reads the data into train/test splits
4. Trains an XGBoost regression model to predict pm2_5
5. Evaluates the model (RMSE, R^2)
6. Registers the model in the Hopsworks Model Registry
"""

import os
from typing import Tuple
import hopsworks

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
FEATURE_VIEW_NAME = "air_quality_fv"
FEATURE_VIEW_VERSION = 1
TARGET_COLUMN_NAME = "pm2_5"


# ---------------------------------------------------------------------
# HOPSWORKS CONNECTION
# ---------------------------------------------------------------------
def connect_to_feature_store_and_registry():
    """
    Connect to Hopsworks using the HOPSWORKS_API_KEY environment variable.
    Returns:
        feature_store, model_registry, project
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
# FEATURE VIEW CREATION / RETRIEVAL
# ---------------------------------------------------------------------
def get_or_create_feature_view(feature_store):
    """
    Create or fetch a Feature View that joins air_quality and weather feature groups.
    """
    # Get Feature Groups
    air_quality_feature_group = feature_store.get_feature_group(
        name="air_quality", version=1
    )
    weather_feature_group = feature_store.get_feature_group(
        name="weather", version=1
    )

    # Join air quality (label) with weather features
    feature_group_query = air_quality_feature_group.select_all().join(
        weather_feature_group.select_all()
    )

    try:
        feature_view = feature_store.get_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
        )
        if feature_view is None:
            raise ValueError(
                f"feature_store.get_feature_view('{FEATURE_VIEW_NAME}', "
                f"{FEATURE_VIEW_VERSION}) returned None"
            )

        print(
            f"Using existing Feature View '{FEATURE_VIEW_NAME}', "
            f"version {FEATURE_VIEW_VERSION}."
        )
    except Exception as e:
        print(
            f"Feature View '{FEATURE_VIEW_NAME}' not found or invalid "
            f"({e}). Creating a new one..."
        )
        feature_view = feature_store.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            description="Joined air quality (pm2_5) and weather features for training.",
            labels=[TARGET_COLUMN_NAME],
            query=feature_group_query,
        )

    if feature_view is None:
        raise RuntimeError(
            f"Failed to obtain a valid Feature View '{FEATURE_VIEW_NAME}', "
            f"version {FEATURE_VIEW_VERSION}."
        )

    return feature_view


# ---------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------
def create_training_pipeline(
    feature_dataframe: pd.DataFrame, target_series: pd.Series
) -> Tuple[SklearnPipeline, float, float]:
    """
    Build and train a scikit-learn pipeline with:
    - OneHotEncoder for categorical features
    - XGBRegressor for regression
    Returns the trained pipeline, RMSE, and R^2 scores.
    """
    X_train = feature_dataframe.copy()
    y_train = target_series

    # Columns present in X_train (from your logs):
    # ['city', 'date', 'weather_date',
    #  'weather_wind_speed_max', 'weather_wind_direction_dominant',
    #  'weather_wind_gusts_max', 'weather_temperature_max']

    # 1) Categorical column (string)
    categorical_feature_names = ["weather_wind_direction_dominant"]

    # 2) Columns we want to DROP (not numeric features)
    drop_columns = ["city", "date", "weather_date"]

    # 3) Numerical feature columns: all others that are not cat or drop
    numerical_feature_names = [
        col
        for col in X_train.columns
        if col not in categorical_feature_names + drop_columns
    ]

    print("\n Training features columns:")
    print("Categorical:", categorical_feature_names)
    print("Numerical:", numerical_feature_names)
    print("Dropping columns:", drop_columns)

    # Optional: actually drop the unwanted columns from the dataframe to be safe
    X_train = X_train.drop(columns=drop_columns)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical_feature_names,
            ),
            (
                "numerical",
                "passthrough",
                numerical_feature_names,
            ),
        ],
        remainder="drop",  # ensure nothing unexpected leaks through
    )

    xgb_regressor = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    training_pipeline = SklearnPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb_regressor),
        ]
    )

    print("\n Training XGBoost model...")
    training_pipeline.fit(X_train, y_train)

    train_predictions = training_pipeline.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    r2 = r2_score(y_train, train_predictions)

    print(f"\n Training completed.")
    print(f"   RMSE (train): {rmse:.3f}")
    print(f"   R^2   (train): {r2:.3f}")

    return training_pipeline, rmse, r2


# ---------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------
def run_training_pipeline():
    """
    1. Connect to Hopsworks (Feature Store + Model Registry)
    2. Get or create Feature View
    3. Split into train/test
    4. Train model
    5. Evaluate and register model in Hopsworks
    """
    (
        feature_store,
        model_registry,
        project,
    ) = connect_to_feature_store_and_registry()

    feature_view = get_or_create_feature_view(feature_store)

    print("\n Loading data from Feature View...")
    print("   feature_view object type:", type(feature_view))

    if feature_view is None:
        raise RuntimeError(
            "feature_view is None right before train_test_split. "
            "Check get_or_create_feature_view()."
        )

    X_train, X_test, y_train, y_test = feature_view.train_test_split(
        test_size=0.2
    )

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test  shape: {X_test.shape}")

    if len(X_train) == 0:
        raise RuntimeError(
            "No training data found in Feature View. "
            "Run the feature pipeline for more than one day to collect data."
        )

    trained_pipeline, train_rmse, train_r2 = create_training_pipeline(
        X_train, y_train
    )

    print("\nEvaluating on test set...")
    test_predictions = trained_pipeline.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    print(f"\n Evaluation completed.")
    print(f"   RMSE (test): {test_rmse:.3f}")
    print(f"   R^2   (test): {test_r2:.3f}")

    print("\n Preparing to register model in Hopsworks Model Registry...")

    model_directory = "air_quality_model"
    os.makedirs(model_directory, exist_ok=True)
    model_file_path = os.path.join(model_directory, "xgboost_pipeline.pkl")
    joblib.dump(trained_pipeline, model_file_path)

    air_quality_model = model_registry.sklearn.create_model(
        name="air_quality_model",
        metrics={
            "rmse_train": train_rmse,
            "r2_train": train_r2,
            "rmse_test": test_rmse,
            "r2_test": test_r2,
        },
        input_example=X_train.iloc[:1],
        description="XGBoost regression model predicting pm2_5 based on weather features.",
    )

    air_quality_model.save(model_directory)

    print(
        "\n Model registered successfully in Hopsworks Model Registry.\n"
        "   You can explore it in the Hopsworks UI."
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_training_pipeline()
