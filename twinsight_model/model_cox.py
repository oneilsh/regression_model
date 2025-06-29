
import logging
import joblib # Keep for general saving/loading, but MLflow will manage models
import pandas as pd
import numpy as np
import os # For creating directories for saving

# --- MLflow Imports ---
import mlflow
import mlflow.pyfunc
# --- Lifelines Imports for Cox PH ---
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index # For explicit C-index calculation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Use logger instead of direct logging.info

# Assume dataloader and preprocessing are adjusted to provide necessary columns
from twinsight_model import dataloader
from twinsight_model import preprocessing


# --- MLflow Model Wrapper for CoxPHFitter ---
# This class wraps the lifelines model to make it compatible with mlflow.pyfunc.log_model
class LifelinesCoxPHWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Loads the CoxPHFitter model and its associated metadata from the context.
        """
        self.model = context.artifacts["model"]
        self.duration_col = context.artifacts["duration_col"]
        self.event_col = context.artifacts["event_col"]
        self.feature_names = context.artifacts["feature_names"]
        logger.info("LifelinesCoxPHWrapper loaded successfully.")

    def predict(self, context, model_input):
        """
        Predicts survival probabilities (or other metrics) using the loaded model.
        This method will return the probability of the event occurring by specific times.
        By default, it will return the predicted partial hazard, which is the linear
        predictor component of the Cox model.
        """
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_names)
        
        # Ensure columns match training features
        # Note: In a real scenario, you'd apply the same preprocessing steps here
        # that were used during training (e.g., scaling, one-hot encoding).
        # For simplicity, this example assumes model_input is already processed.
        
        # To get absolute probabilities:
        # Define time points for prediction (e.g., 1 year, 3 years, 5 years)
        # These are in the same units as your 'time_to_event' (e.g., days, years)
        # For demonstration, let's return probabilities at 1, 3, and 5 years (if time_to_event is in years)
        
        # This wrapper can be extended to accept a 'time_points' argument for prediction
        # For now, we'll return the predicted partial hazard as it's a direct output from Cox,
        # or predicted survival if specific time points are desired for probability.
        
        # If you want probabilities by specific times (e.g., 1, 3, 5 years from time_0)
        # you would add a parameter to the predict method or fix the times:
        # Example to get 1, 3, 5 year survival probabilities
        # pred_times = np.array([1, 3, 5])
        # survival_functions = self.model.predict_survival_function(model_input)
        # survival_probs_at_times = survival_functions.loc[pred_times].T
        # event_probs_at_times = 1 - survival_probs_at_times
        # event_probs_at_times.columns = [f"prob_event_by_{t}_year" for t in pred_times]
        # return event_probs_at_times

        # For basic output consistent with what SHAP would operate on: return partial hazard
        return self.model.predict_partial_hazard(model_input)



# --- Training and Evaluation Functions for Cox PH ---
def train_cox_model(X_train, duration_train, event_train, **kwargs):
    """
    Train a Cox Proportional Hazards model with configurable parameters.

    Parameters:
    X_train (pd.DataFrame): Training data features.
    duration_train (pd.Series): Time to event or censoring for training data.
    event_train (pd.Series): Event observed (1) or censored (0) for training data.
    **kwargs: Additional parameters for CoxPHFitter (e.g., penalizer_alpha, l1_ratio).

    Returns:
    CoxPHFitter: Trained Cox Proportional Hazards model.
    """
    logger.info("Initializing CoxPHFitter model...")
    # CoxPHFitter takes regularization parameters directly
    model = CoxPHFitter(**kwargs)
    try:
        # Pass duration and event columns directly or as series
        model.fit(X_train, duration_col=duration_train.name, event_col=event_train.name)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    return model

def evaluate_cox_model(model, X_test, duration_test, event_test):
    """
    Evaluate the Cox model using the Concordance Index (C-index).

    Parameters:
        model: Trained CoxPHFitter model.
        X_test (pd.DataFrame): Test data features.
        duration_test (pd.Series): Time to event or censoring for test data.
        event_test (pd.Series): Event observed (1) or censored (0) for test data.

    Returns:
        dict: Dictionary containing the C-index score.
    """
    try:
        # Ensure X_test contains duration and event cols for model.score()
        # Create a combined DataFrame for scoring if not already done
        test_df_for_score = X_test.copy()
        test_df_for_score[duration_test.name] = duration_test
        test_df_for_score[event_test.name] = event_test

        c_index = model.score(test_df_for_score,
                              duration_col=duration_test.name,
                              event_col=event_test.name)
        metrics = {"c_index": c_index}
        logger.info("Model evaluation completed.")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    return metrics

def save_model_joblib(model, path):
    """
    Save the trained model to a file using joblib (for non-MLflow context if needed).

    Parameters:
        model: Trained model object.
        path (str): File path to save the model.
    """
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise

def load_model_joblib(path):
    """
    Load a model from a file using joblib (for non-MLflow context if needed).

    Parameters:
        path (str): File path to load the model from.

    Returns:
        The loaded model object.
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def run_end_to_end_pipeline(config_path, model_artifact_path="cox_model", registered_model_name="COPD_Prediction_CoxPH",
                            test_size=0.2, random_state=42,
                            preloaded_data_df=None,
                            **model_kwargs):
    logger.info("Starting the end-to-end ML pipeline for Cox PH model...")

    mlflow.set_experiment("COPD_Prediction_CoxPH_Experiment")

    with mlflow.start_run(run_name="COPD_Cox_Run") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        mlflow.log_params(model_kwargs)
        logger.info(f"Logged model parameters: {model_kwargs}")

        # --- 1. Load Configuration ---
        logger.info(f"Loading configuration from: {config_path}")
        try:
            config = dataloader_cox.load_configuration(config_path)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

        # --- 2. Load Data from BigQuery (OR Use Preloaded Data) ---
        if preloaded_data_df is not None:
            data_df = preloaded_data_df
            logger.info(f"Using preloaded data. Shape: {data_df.shape}")
        else:
            logger.info("Loading data from BigQuery...")
            try:
                data_df = dataloader_cox.load_data_from_bigquery(config)
                logger.info(f"Data loaded from BigQuery. Initial shape: {data_df.shape}")
                logger.info(f"Columns: {data_df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error loading data from Bigquery: {e}")
                raise

        # --- Data Preparation for Cox Model (Time, Event, Features) ---
        time_col = 'time_to_event_days'
        event_col = 'event_observed'

        if time_col not in data_df.columns or event_col not in data_df.columns:
            logger.error(f"FATAL ERROR: Required '{time_col}' or '{event_col}' columns not found for Cox model. Dataloader must provide these.")
            raise ValueError(f"Missing time-to-event or event columns.")

        data_df[event_col] = data_df[event_col].astype(bool)
        data_df[time_col] = data_df[time_col].astype(float).clip(lower=0.1)

        columns_to_exclude_from_features = [
            'person_id', 'birth_datetime', 'date_of_birth', 'gender_concept_id',
            'race_concept_id', 'ethnicity_concept_id', 'sex_at_birth_concept_id',
            'age_at_consent', 'ehr_consent', 'has_ehr_data',
            'condition_start_datetimes', 'condition_end_datetimes',
            time_col, event_col, 'copd_status',

            'gender',
            'race',
            'smoking_status',
            'cardiovascular_disease',
        ]

        feature_columns = [col for col in data_df.columns if col not in columns_to_exclude_from_features]
        logger.info(f"Identified {len(feature_columns)} predictor features.")
        logger.info(f"Feature columns: {feature_columns}")


        # --- 3. Split Data ---
        logger.info("Splitting data into training and testing sets...")
        try:
            # --- CORRECTED: Unpack ALL 6 return values directly from preprocessing_cox.split_data ---
            # Your preprocessing_cox.split_data returns: X_train, duration_train, event_train, X_test, duration_test, event_test
            X_train_raw, duration_train, event_train, X_test_raw, duration_test, event_test = preprocessing_cox.split_data(
                df=data_df[feature_columns + [time_col, event_col]], # Pass relevant columns for splitting
                duration_column=time_col,  # Explicitly pass duration column name as per split_data's signature
                event_column=event_col,    # Explicitly pass event column name as per split_data's signature
                test_size=test_size,
                random_state=random_state,
                # Using stratify_by as per your split_data's signature (ensure it's not commented out there if you want stratification)
                stratify_by=data_df[event_col] if event_col in data_df.columns else None
            )
            # The manual assignments (e.g., X_train_raw = train_df[feature_columns]) are now removed as they are directly unpacked above.

            logger.info(f"Raw data split. X_train_raw shape: {X_train_raw.shape}, X_test_raw shape: {X_test_raw.shape}")
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise

        # --- Explicit Type Conversion for Numerical Features ---
        logger.info("Converting relevant columns to float64 before preprocessing...")
        for col in X_train_raw.select_dtypes(include=['Int64', 'boolean']).columns:
            X_train_raw[col] = X_train_raw[col].astype(float)
        for col in X_test_raw.select_dtypes(include=['Int64', 'boolean']).columns:
            X_test_raw[col] = X_test_raw[col].astype(float)


        # --- 4. Feature Preprocessing (Imputation, Scaling, Encoding) ---
        logger.info("Applying feature preprocessing pipeline...")
        try:
            preprocessor = preprocessing_cox.create_preprocessor(X_train_raw)
            X_train_processed_array, X_test_processed_array = preprocessing_cox.apply_preprocessing(
                preprocessor, X_train_raw, X_test_raw
            )
            
            X_train_processed = pd.DataFrame(X_train_processed_array, columns=preprocessor.get_feature_names_out())
            X_test_processed = pd.DataFrame(X_test_processed_array, columns=preprocessor.get_feature_names_out())

            logger.info(f"Features processed. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")
        except Exception as e:
            logger.error(f"Error during feature preprocessing: {e}")
            raise

        # --- CRITICAL: Dynamically drop problematic columns POST-preprocessing (for model fit only) ---
        logger.info("Checking processed features for potential convergence issues before model fit...")
        final_X_train_processed = X_train_processed.copy()
        final_X_test_processed = X_test_processed.copy()
        
        problematic_cols_to_drop_for_fit = []

        for col in final_X_train_processed.columns:
            if final_X_train_processed[col].nunique() <= 1:
                if final_X_train_processed[col].sum() == 0:
                    logger.warning(f"Processed column '{col}' is all zeros (zero variance). Adding to drop list for model fit.")
                else:
                    logger.warning(f"Processed column '{col}' has zero variance (only one unique non-zero value). Adding to drop list for model fit.")
                problematic_cols_to_drop_for_fit.append(col)
            elif pd.api.types.is_numeric_dtype(final_X_train_processed[col]):
                col_variance = final_X_train_processed[col].var() # Corrected from .lower_bound_values
                if col_variance < 1e-8:
                    logger.warning(f"Processed numerical column '{col}' has very low variance ({col_variance:.2e}). Adding to drop list for model fit.")
                    problematic_cols_to_drop_for_fit.append(col)

        problematic_cols_to_drop_for_fit = list(set(problematic_cols_to_drop_for_fit)) # Remove duplicates

        if problematic_cols_to_drop_for_fit:
            logger.info(f"Temporarily dropping {len(problematic_cols_to_drop_for_fit)} problematic processed columns for Cox model fit: {problematic_cols_to_drop_for_fit}")
            final_X_train_processed = final_X_train_processed.drop(columns=problematic_cols_to_drop_for_fit)
            final_X_test_processed = final_X_test_processed.drop(columns=problematic_cols_to_drop_for_fit)
        else:
            logger.info("No additional problematic processed columns identified for temporary drop.")

        logger.info(f"Final features for model fit. X_train shape: {final_X_train_processed.shape}, X_test shape: {final_X_test_processed.shape}")


        # Add duration and event columns back to processed dataframes for lifelines .fit() and .score()
        # This is needed because lifelines expects all relevant columns in one dataframe for fit/score
        train_df_processed_for_fit = final_X_train_processed.copy() # Use the final, cleaned feature set
        train_df_processed_for_fit[duration_train.name] = duration_train
        train_df_processed_for_fit[event_train.name] = event_train

        test_df_processed_for_score = final_X_test_processed.copy() # Use the final, cleaned feature set
        test_df_processed_for_score[duration_test.name] = duration_test
        test_df_processed_for_score[event_test.name] = event_test


        # --- 5. Train Cox PH Model ---
        logger.info("Training the Cox Proportional Hazards model...")
        try:
            # Check if any features remain after dynamic dropping
            if train_df_processed_for_fit.empty or len(train_df_processed_for_fit.columns) == 0: # Ensure at least one feature
                logger.error("No valid features remaining for CoxPHFitter after dynamic column removal. Cannot train model.")
                trained_cox_model = None
            else:
                trained_cox_model = train_cox_model(
                    train_df_processed_for_fit,
                    duration_train,
                    event_train,
                    **model_kwargs # These are the actual CoxPHFitter parameters
                )
                logger.info("Cox PH Model training completed successfully.")
                # Log Cox model summary
                logger.info("\n--- Cox PH Model Summary ---")
                logger.info(trained_cox_model.summary.to_string())
                logger.info("----------------------------")

        except Exception as e:
            logger.error(f"Error during Cox PH model training: {e}", exc_info=True)
            trained_cox_model = None
            raise

        # --- 6. Evaluate Cox PH Model ---
        evaluation_metrics = {}
        if trained_cox_model is not None:
            logger.info("Evaluating the Cox PH model on the test set...")
            try:
                evaluation_metrics = evaluate_cox_model(
                    trained_cox_model,
                    final_X_test_processed,
                    duration_test,
                    event_test
                )
                logger.info("Cox PH Model evaluation completed.")

                logger.info("\n--- Cox PH Model Evaluation Metrics ---")
                for k, v in evaluation_metrics.items():
                    logger.info(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
                logger.info("------------------------------")
                mlflow.log_metrics(evaluation_metrics) # Log evaluation metrics to MLflow

            except Exception as e:
                logger.error(f"Error during model evaluation: {e}", exc_info=True)
                # Do not re-raise, allow logging model even if evaluation fails
        else:
            logger.warning("Model training failed, skipping evaluation.")


        # --- 7. Log Model with MLflow ---
        if trained_cox_model is not None:
            logger.info(f"Logging the trained Cox PH model to MLflow artifact path: {model_artifact_path}")
            try:
                # Define conda environment for MLflow model (must include lifelines)
                conda_env = {
                    "channels": ["defaults", "conda-forge"],
                    "dependencies": [
                        f"python={os.environ.get('PYTHON_VERSION', '3.9')}",
                        "pip",
                        {"pip": ["mlflow", "lifelines", "pandas", "numpy", "scikit-learn"]}
                    ]
                }

                mlflow.pyfunc.log_model(
                    artifact_path=model_artifact_path,
                    python_model=LifelinesCoxPHWrapper(),
                    artifacts={
                        "model": trained_cox_model,
                        "duration_col": duration_train.name,
                        "event_col": event_train.name,
                        # Store feature names that were *actually* used for training
                        "feature_names": final_X_train_processed.columns.tolist()
                    },
                    conda_env=conda_env,
                    registered_model_name=registered_model_name
                )
                logger.info(f"Cox PH model logged to MLflow successfully under run ID: {run_id}")

            except Exception as e:
                logger.error(f"Error logging model to MLflow: {e}", exc_info=True)
                # Do not re-raise, allow overall pipeline to complete with partial success
        else:
            logger.warning("Model training failed, skipping MLflow model logging.")

    logger.info("ML pipeline completed successfully (with potential warnings/errors).")
    return trained_cox_model, evaluation_metrics


if __name__ == "__main__":
    # This is a placeholder for how you would call the pipeline.
    # In a real scenario, config_path and model_save_path would be passed
    # as command-line arguments or from a configuration system.

    # Simulate config_path for testing (replace with actual path to your config.yaml)
    # Ensure config.yaml is in the parent directory, or adjust path
    _config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configuration.yaml')
    # Use a dummy path for joblib save if needed, but MLflow will handle main model logging
    _model_save_path_joblib = "local_cox_model.joblib"

    # Example model parameters for CoxPHFitter
    _model_kwargs = {
        "penalizer_alpha": 0.05,  # L2 regularization strength
        "l1_ratio": 0.0            # 0.0 for L2 (Ridge), 1.0 for L1 (Lasso)
    }

    # IMPORTANT: The dataloader.load_data_from_bigquery(config) function
    # needs to be updated to:
    # 1. Establish the 'time_0' for each participant, ensuring no outcome prior.
    # 2. Derive 'time_to_event_days' (duration from time_0 to event or censoring).
    # 3. Derive 'event_observed' (1 for event, 0 for censored).
    # This current model.py assumes these columns are provided by the dataloader.
    
    # Simulate a dataframe that dataloader.py *should* return for demonstration
    # In a real scenario, this df would come from your dataloader/preprocessing.
    # Ensure it has 'time_to_event_days' and 'event_observed'
    # And other features as defined in your YAML.
    
    # Placeholder for a dummy load_configuration and load_data_from_bigquery
    # Replace with your actual implementation from dataloader.py
    class MockDataloader:
        @staticmethod
        def load_configuration(path):
            # Simplified mock config based on your YAML structure
            return {
                'outcome': {'name': 'copd_status'}, # Example outcome name
                'co_indicators': [
                    {'name': 'obesity', 'domain': 'condition_occurrence', 'concepts_include': [4433736]},
                    {'name': 'diabetes', 'domain': 'condition_occurrence', 'concepts_include': [201826]},
                    {'name': 'cardiovascular_disease', 'domain': 'condition_occurrence', 'concepts_include': [319835]}
                ],
                'features': [
                    {'name': 'smoking_status_obs', 'domain': 'observation', 'type': 'categorical', 'concepts_include': [1585856]},
                    {'name': 'bmi', 'domain': 'measurement', 'type': 'continuous', 'concepts_include': [3038553]},
                    {'name': 'year_of_birth', 'domain': 'person', 'type': 'continuous'}
                ]
            }

        @staticmethod
        def load_data_from_bigquery(config):
            # Simulate data with required time and event columns
            np.random.seed(42)
            n_samples = 1000
            
            # Simulated outcome
            copd_status = np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 0=No COPD, 1=COPD
            
            # Simulated time to event or censoring (in days)
            # Shorter times for those with event=1
            time_to_event = np.where(copd_status == 1, np.random.exponential(365*2, n_samples), np.random.exponential(365*5, n_samples))
            # Simulate some censoring before the actual event time
            censoring_time = np.random.uniform(365, 365*7, n_samples)
            
            # Final time and event status
            final_time = np.minimum(time_to_event, censoring_time)
            final_event = np.where(time_to_event <= censoring_time, copd_status, 0) # 0 if censored, 1 if event occurred within follow-up
            
            # Ensure time is positive
            final_time = np.maximum(final_time, 1) # Minimum 1 day duration

            data = {
                'person_id': range(n_samples),
                'age_at_time_0': np.random.randint(40, 80, n_samples), # Derived age
                'smoking_status_obs': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.5, 0.3, 0.2]),
                'bmi': np.random.normal(28, 5, n_samples),
                'diabetes': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'cardiovascular_disease': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                # These cols are now *NOT* features, but used to derive outcome.
                # Adding them here to show they might be in the raw dataframe.
                'condition_start_datetimes': pd.to_datetime('2000-01-01') + pd.to_timedelta(np.random.randint(0, 365*20, n_samples), unit='D'),
                'condition_end_datetimes': pd.to_datetime('2000-01-01') + pd.to_timedelta(np.random.randint(0, 365*20, n_samples), unit='D'),
                
                # These are the actual outcome variables for Cox model
                'time_to_event_days': final_time,
                'event_observed': final_event,
                'copd_status': copd_status # The original binary outcome label from config
            }
            df = pd.DataFrame(data)
            return df

    # Replace your actual dataloader and preprocessing if you run this directly
    # For testing, we'll temporarily assign the mock
    _original_dataloader_load_config = dataloader.load_configuration
    _original_dataloader_load_data = dataloader.load_data_from_bigquery
    _original_preprocessing_create_preprocessor = preprocessing.create_preprocessor
    _original_preprocessing_apply_preprocessing = preprocessing.apply_preprocessing
    _original_preprocessing_split_data = preprocessing.split_data

    # Mock out dataloader functions for standalone execution
    dataloader.load_configuration = MockDataloader.load_configuration
    dataloader.load_data_from_bigquery = MockDataloader.load_data_from_bigquery

    # Mock out preprocessing functions as well, for simplicity in this demo
    # In a real setup, ensure your preprocessing functions correctly handle the
    # columns and types expected by the CoxPHFitter after this step.
    class MockPreprocessor:
        def __init__(self):
            # A simple passthrough for features in this mock
            self.feature_names_out_ = None
        def fit(self, X):
            self.feature_names_out_ = X.columns.tolist()
            return self
        def transform(self, X):
            return X.values # Return numpy array, assuming no scaling/encoding needed for this mock

        def get_feature_names_out(self):
            return self.feature_names_out_

    class MockPreprocessing:
        @staticmethod
        def create_preprocessor(X):
            # This mock preprocessor just passes features through for demo purposes
            # In your actual code, this would set up Imputation, Scaling, Encoding
            preprocessor = MockPreprocessor()
            preprocessor.fit(X) # "Fit" on the columns
            return preprocessor

        @staticmethod
        def apply_preprocessing(preprocessor, X_train_clean, X_test_clean):
            # This mock just returns the cleaned data as numpy arrays
            return preprocessor.transform(X_train_clean), preprocessor.transform(X_test_clean)

        @staticmethod
        def split_data(df, target_column, test_size, random_state, stratify):
            # Simplified split, ensures time/event cols are kept together
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)
            return train_df, test_df # Return DataFrames directly for easier column separation


    preprocessing.create_preprocessor = MockPreprocessing.create_preprocessor
    preprocessing.apply_preprocessing = MockPreprocessing.apply_preprocessing
    preprocessing.split_data = MockPreprocessing.split_data

    try:
        run_end_to_end_pipeline(
            config_path=_config_path,
            test_size=0.2,
            random_state=42,
            **_model_kwargs
        )
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    finally:
        # Restore original functions after execution
        dataloader.load_configuration = _original_dataloader_load_config
        dataloader.load_data_from_bigquery = _original_dataloader_load_data
        preprocessing.create_preprocessor = _original_preprocessing_create_preprocessor
        preprocessing.apply_preprocessing = _original_preprocessing_apply_preprocessing
        preprocessing.split_data = _original_preprocessing_split_data
