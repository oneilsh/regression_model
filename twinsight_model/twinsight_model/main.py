# twinsight_model/main.py

import logging
import argparse
from twinsight_model.config import TARGET_COLUMN, FEATURE_COLUMNS
from twinsight_model.data_loader import load_data, filter_columns
from twinsight_model.preprocessing import preprocess_data
from twinsight_model.model import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="TWINSight CLI")
    parser.add_argument('--data', type=str, required=True, help='Path to the input CSV file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        # Load data
        df = load_data(args.data)

        # Filter only required columns
        df = filter_columns(df, FEATURE_COLUMNS + [TARGET_COLUMN])

        # Preprocess
        X_train, X_test, y_train, y_test = preprocess_data(df, TARGET_COLUMN)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        exit(1)
