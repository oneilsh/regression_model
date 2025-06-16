import logging
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd  
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score
)
from twinsight_model import dataloader
from twinsight_model import preprocessing

logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train, **kwargs):
    """
    Train a logistic regression model with configurable parameters.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    **kwargs: Additional parameters for LogisticRegression.

    Returns:
    LogisticRegression: Trained logistic regression model.
    """
    logging.info("Initializing LogisticRegression model...")
    model = LogisticRegression(max_iter=1000, **kwargs)
    try:
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using several metrics.

    Parameters:
        model: Trained model with predict and predict_proba methods.
        X_test (array-like): Test data features.
        y_test (array-like): True test labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and ROC AUC scores.
    """
    preds = model.predict(X_test)
    try:
        probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probas)
    except Exception:
        probas = None
        roc_auc = None
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="binary"),
        "recall": recall_score(y_test, preds, average="binary"),
        "f1_score": f1_score(y_test, preds, average="binary"),
        "roc_auc": roc_auc
    }
    return metrics

def save_model(model, path):
    """
    Save the trained model to a file.

    Parameters:
        model: Trained model object.
        path (str): File path to save the model.
    """
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def load_model(path):
    """
    Load a model from a file.

    Parameters:
        path (str): File path to load the model from.

    Returns:
        The loaded model object.
    """
    try:
        model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def run_end_to_end_pipeline(config_path, model_save_path, test_size=0.2, random_state=42, **model_kwargs):
    """
    Executes the full end-to-end ML pipeline from configuration loading to model saving.

    Parameters:
    config_path (str): Absolute path to the configuration.yaml file.
    model_save_path (str): Absolute path to save the trained scikit-learn model.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed for random number generator for reproducibility.
    **model_kwargs: Additional parameters to pass to the LogisticRegression model.

    Returns:
    tuple: (trained_sklearn_model, evaluation_metrics_dict)
    """
    logging.info("Starting the end-to-end ML pipeline from model.py...")

    # --- 1. Load Configuration ---
    logging.info(f"Loading configuration from: {config_path}")
    try:
        config = dataloader.load_configuration(config_path)
        logging.info("Configuration loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

    # --- 2. Load Data from BigQuery ---
    logging.info("Loading data from BigQuery...")
    try:
        data_df = dataloader.load_data_from_bigquery(config)
        logging.info(f"Data loaded from BigQuery. Initial shape: {data_df.shape}")
        logging.info(f"Columns: {data_df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error loading data from BigQuery: {e}")
        raise

    # --- 3. Prepare Data for Modeling (Feature Engineering & Target Handling) ---

    # Determine the target column name from configuration
    target_condition_name = config.get('outcome', {}).get('name')
    if not target_condition_name:
        raise ValueError("Outcome name not found in configuration.yaml under 'outcome.name'.")

    target_column = target_condition_name
    logging.info(f"Identified target column: '{target_column}'")

    # Handle missing values in the target column
    if data_df[target_column].isnull().any():
        initial_rows = len(data_df)
        data_df[target_column].fillna(0.0, inplace=True)
        logging.warning(f"Filled NaN values in target column '{target_column}' with 0.0.")
        if len(data_df) < initial_rows: # If fillna was called on a series that caused rows to drop, check again
             logging.warning(f"Number of rows changed after fillna: {initial_rows} -> {len(data_df)}")


    # --- DIAGNOSTIC STEPS (for class imbalance and data completeness) ---
    logging.info(f"Unique values in target column ('{target_column}'): {data_df[target_column].unique()}")
    logging.info(f"Value counts in target column ('{target_column}'):\n{data_df[target_column].value_counts()}")

    if data_df[target_column].nunique() < 2:
        logging.error(f"FATAL ERROR: The entire loaded DataFrame's target column ('{target_column}') "
                      "contains less than 2 unique classes. Model cannot be trained.")
        raise ValueError("Target column has only one class in the full dataset.")

    if data_df.empty:
        raise ValueError("DataFrame is empty after target value handling. Cannot proceed.")
    # --- END DIAGNOSTIC STEPS ---


    # --- NEW: Convert Datetime Columns to Numerical Features (Unix Timestamps) ---
    logging.info("Converting datetime columns to numerical features (Unix timestamps)...")
    datetime_features_to_convert = [
        'condition_start_datetimes',
        'condition_end_datetimes'
    ]

    for col in datetime_features_to_convert:
        if col in data_df.columns:
            datetime_series_ns = pd.to_datetime(data_df[col], errors='coerce')
            unix_timestamp_series = (datetime_series_ns.astype(np.int64) // 10**9)
            data_df[col + '_unix'] = unix_timestamp_series.fillna(0.0).astype(float)
            logging.info(f"Created '{col}_unix' from '{col}'.")
        else:
            logging.warning(f"Datetime feature '{col}' not found in data_df. Skipping conversion to unix timestamp.")


    # --- NEW: Calculate 'condition_duration_days' from Unix timestamps ---
    logging.info("Calculating 'condition_duration_days' from min/max condition datetimes...")

    start_col_unix = 'condition_start_datetimes_unix'
    end_col_unix = 'condition_end_datetimes_unix'

    if start_col_unix in data_df.columns and end_col_unix in data_df.columns:
        duration_seconds = data_df[end_col_unix] - data_df[start_col_unix]
        data_df['condition_duration_days'] = (duration_seconds / 86400.0).clip(lower=0.0)
        logging.info(f"Created 'condition_duration_days'. Min: {data_df['condition_duration_days'].min()}, Max: {data_df['condition_duration_days'].max()}")
        logging.info(f"Non-null count for 'condition_duration_days': {data_df['condition_duration_days'].count()}")
    else:
        logging.warning(f"Columns '{start_col_unix}' or '{end_col_unix}' not found. Skipping 'condition_duration_days' calculation.")

    # Identify and exclude non-feature columns
    columns_to_exclude_from_features = [
        'person_id', 'birth_datetime', 'date_of_birth', 'gender_concept_id',
        'race_concept_id', 'ethnicity_concept_id', 'sex_at_birth_concept_id',
        'age_at_consent', 'ehr_consent', 'has_ehr_data', 'year_of_birth',
        'condition_start_datetimes', 'condition_end_datetimes'
    ]

    # --- 4. Split Data ---
    logging.info("Splitting data into training and testing sets...")
    try:
        X_train_raw, X_test_raw, y_train, y_test = preprocessing.split_data(
            df=data_df,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            stratify=data_df[target_column]
        )
        logging.info(f"Raw data split. X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")
    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        raise

    # --- Explicit Type Conversion for Numerical Features in X_train_raw/X_test_raw ---
    logging.info("Converting relevant columns to float64 before preprocessing...")
    for col in X_train_raw.select_dtypes(include=['Int64', 'boolean']).columns:
        X_train_raw[col] = X_train_raw[col].astype(float)
        X_test_raw[col] = X_test_raw[col].astype(float)

    # --- 5. Feature Preprocessing (Imputation, Scaling, Encoding) ---
    logging.info("Applying feature preprocessing pipeline...")

    X_train_clean = X_train_raw.drop(columns=[col for col in columns_to_exclude_from_features if col in X_train_raw.columns], errors='ignore')
    X_test_clean = X_test_raw.drop(columns=[col for col in columns_to_exclude_from_features if col in X_test_raw.columns], errors='ignore')

    logging.info("--- X_train_clean dtypes before preprocessing ---")
    # print(X_train_clean.dtypes) # Removed print for cleaner logs, logging.info sufficient if needed

    try:
        preprocessor = preprocessing.create_preprocessor(X_train_clean)
        X_train_processed, X_test_processed = preprocessing.apply_preprocessing(
            preprocessor, X_train_clean, X_test_clean
        )
        logging.info(f"Features processed. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")
    except Exception as e:
        logging.error(f"Error during feature preprocessing: {e}")
        raise

    # --- NEW: Filter out Constant/Zero-Variance Features for Statsmodels (post-preprocessing) ---
    logging.info("Checking for constant features in X_train_processed (post-preprocessing) for Statsmodels...")
    constant_feature_indices = []
    for i in range(X_train_processed.shape[1]):
        if np.std(X_train_processed[:, i]) < 1e-9:
            constant_feature_indices.append(i)

    if constant_feature_indices:
        logging.warning(f"Found {len(constant_feature_indices)} constant/zero-variance features in processed data. Removing them for Statsmodels stability.")
        non_constant_mask = np.ones(X_train_processed.shape[1], dtype=bool)
        non_constant_mask[constant_feature_indices] = False

        X_train_processed_sm_filtered = X_train_processed[:, non_constant_mask]
        X_test_processed_sm_filtered = X_test_processed[:, non_constant_mask]
        logging.info(f"New X_train_processed shape after dropping constant features for Statsmodels: {X_train_processed_sm_filtered.shape}")
    else:
        logging.info("No constant features found. Proceeding with all features.")
        X_train_processed_sm_filtered = X_train_processed.copy()
        X_test_processed_sm_filtered = X_test_processed.copy()

    # --- Generate Statsmodels Logistic Regression Summary ---
    logging.info("Generating OLS-style summary table for Logistic Regression (using Statsmodels)...")
    try:
        X_train_sm = sm.add_constant(X_train_processed_sm_filtered)
        y_train_sm = y_train.to_numpy().astype(np.float64)

        sm_logit_model = sm.Logit(y_train_sm, X_train_sm)
        sm_logit_result = sm_logit_model.fit(disp=False)
        logging.info("\n--- Statsmodels Logistic Regression Summary ---")
        logging.info(sm_logit_result.summary().as_text()) # Log as text to avoid printing raw object
        logging.info("---------------------------------------------")

    except Exception as e:
        logging.error(f"Error generating Statsmodels summary: {e}")
        # Do not raise here, as scikit-learn model training should still proceed
        # if statsmodels summary fails for some reason.

    # --- 6. Train scikit-learn Model ---
    logging.info("Training the scikit-learn Logistic Regression model...")
    try:
        trained_sklearn_model = train_model(X_train_processed_sm_filtered, y_train, **model_kwargs)
        logging.info("scikit-learn Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during scikit-learn model training: {e}")
        raise

    # --- 7. Evaluate scikit-learn Model ---
    logging.info("Evaluating the scikit-learn model on the test set...")
    try:
        evaluation_metrics = evaluate_model(trained_sklearn_model, X_test_processed_sm_filtered, y_test)
        logging.info("scikit-learn Model evaluation completed.")

        logging.info("\n--- scikit-learn Model Evaluation Metrics ---")
        for k, v in evaluation_metrics.items():
            logging.info(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
        logging.info("------------------------------")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

    # --- 8. Save Model ---
    logging.info(f"Saving the trained scikit-learn model to: {model_save_path}")
    try:
        # Ensure the directory for saving the model exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        save_model(trained_sklearn_model, model_save_path)
        logging.info(f"ML pipeline completed successfully! Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

    return trained_sklearn_model, evaluation_metrics
