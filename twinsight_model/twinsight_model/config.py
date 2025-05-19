# List of features to use in the model
FEATURE_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "smoking_status",
    "exercise_frequency",
    "alcohol_use",
    "comorbidity_score"
]

# Target variable
TARGET_COLUMN = "disease_outcome"

# Risk stratification threshold
RISK_THRESHOLD = 0.5
