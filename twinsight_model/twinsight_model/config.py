"""
Configuration file for the regression model.
This file defines the feature columns, target column, and risk threshold,
and allows for environment-based overrides for flexibility.
"""

import os

def get_feature_columns():
    """
    Returns the list of feature columns to be used in the model.
    You can override this by setting the FEATURE_COLUMNS environment variable as a comma-separated list.
    """
    default_features = [
        "age",
        "gender",
        "bmi",
        "smoking_status",
        "exercise_frequency",
        "alcohol_use",
        "comorbidity_score"
    ]
    features = os.getenv("FEATURE_COLUMNS")
    if features:
        return [col.strip() for col in features.split(",") if col.strip()]
    return default_features

def get_target_column():
    """
    Returns the target column name.
    You can override this by setting the TARGET_COLUMN environment variable.
    """
    return os.getenv("TARGET_COLUMN", "disease_outcome")

def get_risk_threshold():
    """
    Returns the risk stratification threshold as a float between 0 and 1.
    You can override this by setting the RISK_THRESHOLD environment variable.
    """
    try:
        threshold = float(os.getenv("RISK_THRESHOLD", "0.5"))
        if not (0 <= threshold <= 1):
            raise ValueError
        return threshold
    except ValueError:
        raise ValueError("RISK_THRESHOLD must be a float between 0 and 1.")

FEATURE_COLUMNS = get_feature_columns()
TARGET_COLUMN = get_target_column()
RISK_THRESHOLD = get_risk_threshold()

# Optional: Print configuration summary for debugging (remove in production)
if __name__ == "__main__":
    print("Feature columns:", FEATURE_COLUMNS)
    print("Target column:", TARGET_COLUMN)
    print("Risk threshold:", RISK_THRESHOLD)
