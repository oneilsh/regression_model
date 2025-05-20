# twinsight_model/preprocessing.py
import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        stratify (pd.Series or None): If not None, data is split in a stratified fashion based on this variable.

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.

    Raises:
        KeyError: If target_column is not in df.
        ValueError: If df is empty or has insufficient rows for splitting.
    """
    logging.info("Starting data preprocessing...")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' does not exist in the DataFrame.")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    if len(df) < 2:
        raise ValueError("DataFrame must have at least 2 rows to split.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Optional: Check for missing values and log warnings
    if X.isnull().any().any():
        logging.warning("There are missing values in the features (X). Consider handling them before training.")
    if y.isnull().any():
        logging.warning("There are missing values in the target (y). Consider handling them before training.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as e:
        logging.error(f"Error during train/test split: {e}")
        raise

    logging.info(
        f"Data split complete: {len(X_train)} train samples, {len(X_test)} test samples."
    )
    return X_train, X_test, y_train, y_test
