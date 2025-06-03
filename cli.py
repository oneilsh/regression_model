# python cli.py --data prediction_model_file.csv - Running this will pass the provided data file to twinsight-cli and handle errors


"""
CLI script for running the twinsight-cli command with a specified data file.
This script parses the command-line arguments, executes the twinsight-cli command,
and handles any errors that may occur during execution.
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np # For type hinting np.ndarray

# Import modules from your twinsight_model package
from twinsight_model import dataloader
from twinsight_model import preprocessing
from twinsight_model import model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Twinsight CLI - Run end-to-end model pipeline.")

    # Data loading arguments (based on dataloader.py's config-driven approach)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file for data loading (e.g., configuration.yaml).')
    
    # Model saving argument
    parser.add_argument('--model-output', type=str, default='trained_model.joblib',
                        help='Filename to save the trained model (e.g., my_model.joblib).')
    
    # Preprocessing arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2).')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility of data split (default: 42).')

    args = parser.parse_args()

    # --- Pipeline Execution ---
    try:
        logging.info("Step 1: Loading configuration...")
        config = dataloader.load_configuration(args.config)
        
        logging.info("Step 2: Loading data from BigQuery based on configuration...")
        data_df = dataloader.load_data_from_bigquery(config)
        logging.info(f"Initial data loaded. Shape: {data_df.shape}")
        
        # Determine the target column based on the outcome defined in config
        outcome_config = config.get('outcome')
        if not outcome_config or 'name' not in outcome_config or 'domain' not in outcome_config:
            raise ValueError("Outcome not properly defined in the configuration file.")
        
        outcome_name = outcome_config['name']
        outcome_domain = outcome_config['domain']
        target_column = f"{outcome_name}_presence" if outcome_domain != 'measurement' else f"{outcome_name}_value"
        
        if target_column not in data_df.columns:
            # This might happen if the outcome concept isn't found for any person, leading to missing column
            logging.error(f"Target column '{target_column}' not found in the loaded data. "
                          "Ensure outcome concepts are present in the dataset.")
            raise KeyError(f"Target column '{target_column}' not found.")

        # Handle missing values in the target (critical for splitting/training)
        if data_df[target_column].isnull().any():
            initial_rows = len(data_df)
            data_df.dropna(subset=[target_column], inplace=True)
            logging.warning(f"Removed {initial_rows - len(data_df)} rows due to missing values in target column '{target_column}'. "
                            f"Remaining rows: {len(data_df)}")
        
        if data_df.empty:
            raise ValueError("DataFrame is empty after handling missing target values. Cannot proceed with splitting.")

        logging.info("Step 3: Splitting data into training and test sets...")
        # Pass the entire DataFrame to split_data; it will handle dropping the target column
        X_train_raw, X_test_raw, y_train, y_test = preprocessing.split_data(
            df=data_df,
            target_column=target_column,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=data_df[target_column] # Stratify on the target variable
        )
        logging.info(f"Data split. X_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
        logging.info(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")
        
        # Exclude non-feature columns (like person_id) from X_train_raw and X_test_raw
        # This needs to be handled carefully. The dataloader might bring in metadata.
        # It's safest to define 'features' explicitly or to exclude non-feature columns
        # right before preprocessing.
        
        # Example of explicit column removal that are not features from X_train_raw/X_test_raw
        # You should identify these based on your dataloader.py output.
        # Person_id, date_of_birth, gender_concept_id, etc., are base person columns.
        # Only include columns that are meant to be model features.
        
        # Simplistic approach: Assume all columns in X_train_raw are features,
        # but drop typical identifiers/metadata that shouldn't be features.
        
        columns_to_exclude_from_features = [
            'person_id', 'date_of_birth', 'gender_concept_id', 'race_concept_id',
            'ethnicity_concept_id', 'sex_at_birth_concept_id', 'ehr_consent',
            'year_of_birth', # Assuming 'year_of_birth' is not a direct feature
            # Add any other non-feature columns from your dataloader.py output
        ]
        
        # Filter out columns that are not meant to be features from the raw X sets
        X_train_clean = X_train_raw.drop(columns=[col for col in columns_to_exclude_from_features if col in X_train_raw.columns])
        X_test_clean = X_test_raw.drop(columns=[col for col in columns_to_exclude_from_features if col in X_test_raw.columns])

        logging.info("Step 4: Creating and applying feature preprocessing pipeline...")
        # Create and fit the preprocessor on X_train_clean
        preprocessor = preprocessing.create_preprocessor(X_train_clean)
        
        # Apply the fitted preprocessor to both training and test sets
        X_train_processed, X_test_processed = preprocessing.apply_preprocessing(
            preprocessor, X_train_clean, X_test_clean
        )
        logging.info(f"Features processed. X_train_processed shape: {X_train_processed.shape}, X_test_processed shape: {X_test_processed.shape}")

        logging.info("Step 5: Training model...")
        # model.train_model expects array-like inputs, which X_train_processed is
        trained_model = model.train_model(X_train_processed, y_train)
        logging.info("Model training completed.")

        logging.info("Step 6: Evaluating model...")
        # model.evaluate_model expects array-like inputs
        metrics = model.evaluate_model(trained_model, X_test_processed, y_test)
        logging.info("Model evaluation completed.")

        logging.info(f"Step 7: Saving model to {args.model_output}...")
        model.save_model(trained_model, args.model_output)
        logging.info("Model saved successfully.")

        # --- Display Results ---
        print("\n--- Model Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
        print("------------------------------")

    except Exception as e:
        logging.critical(f"An error occurred during the pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
