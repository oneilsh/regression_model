# python cli.py --data prediction_model_file.csv - Running this will pass the provided data file to twinsight-cli and handle errors


"""
CLI script for running the twinsight-cli command with a specified data file.
This script parses the command-line arguments, executes the twinsight-cli command,
and handles any errors that may occur during execution.
"""

import argparse
import logging
import sys
from twinsight_model.data_loader import load_data, filter_columns, stratify_by_risk
from twinsight_model.preprocessing import preprocess_data
from twinsight_model.model import train_model, evaluate_model, save_model

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Twinsight CLI - Run end-to-end model pipeline.")
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV data file.')
    parser.add_argument('--target', type=str, required=True, help='Target column name (e.g., condition_flag).')
    parser.add_argument('--features', type=str, nargs='+', required=True, help='List of feature column names.')
    parser.add_argument('--risk-column', type=str, required=False, help='Column used for risk stratification.')
    parser.add_argument('--risk-threshold', type=float, default=0.5, help='Threshold for stratifying risk (default: 0.5).')
    parser.add_argument('--model-output', type=str, default='model.joblib', help='Filename to save trained model.')
    
    args = parser.parse_args()

    try:
        logging.info(f"Loading data from {args.data}...")
        df = load_data(args.data)

        logging.info("Filtering columns...")
        required_columns = args.features + [args.target]
        df = filter_columns(df, required_columns)

        if args.risk_column:
            logging.info(f"Stratifying by risk column '{args.risk_column}' with threshold {args.risk_threshold}...")
            df = stratify_by_risk(df, args.risk_column, args.risk_threshold)

        logging.info("Preprocessing data and splitting into train/test sets...")
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column=args.target, stratify=df[args.target])

        logging.info("Training model...")
        model = train_model(X_train, y_train)

        logging.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        logging.info("Saving model...")
        save_model(model, args.model_output)

        print("\nModel Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
