import logging
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np # For type hinting ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)

def split_data(
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
        TypeError: If input df is not a pandas DataFrame.
    """
    logging.info("Starting data splitting...")
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
        logging.warning("There are missing values in the features (X) *before* splitting. Consider handling them.")
    if y.isnull().any():
        logging.warning("There are missing values in the target (y) *before* splitting. Consider handling them.")

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

def create_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Creates and fits a scikit-learn preprocessing pipeline based on the training data.
    This pipeline handles numerical imputation/scaling and categorical one-hot encoding.

    Parameters:
        X_train (pd.DataFrame): The training features DataFrame, used to fit the preprocessor.

    Returns:
        sklearn.pipeline.Pipeline: A fitted preprocessing pipeline (ColumnTransformer).
    """
    logging.info("Creating feature preprocessing pipeline...")

    # Identify numeric and categorical features
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    if not numeric_features and not categorical_features:
        logging.warning("No numeric or categorical features identified for preprocessing.")
        # If no features are identified, return a dummy preprocessor
        return Pipeline(steps=[('passthrough', 'passthrough')])


    # Define preprocessing pipelines for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute with mean
        ('scaler', StandardScaler()) # Scale numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Impute with most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
    ])

    # Create a preprocessor to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, or 'drop'
    )
    logging.info("Feature preprocessing pipeline created.")
    return preprocessor

def apply_preprocessing(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the fitted preprocessing pipeline to both training and test data.

    Parameters:
        preprocessor (ColumnTransformer): The fitted preprocessing pipeline.
        X_train (pd.DataFrame): Raw training features.
        X_test (pd.DataFrame): Raw test features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed X_train and X_test as NumPy arrays.
    """
    logging.info("Applying preprocessing to training data...")
    X_train_processed_sparse = preprocessor.fit_transform(X_train)
    # Convert sparse to dense float64 NumPy array immediately
    X_train_processed = X_train_processed_sparse.toarray().astype(np.float64)

    logging.info("Applying preprocessing to test data...")
    X_test_processed_sparse = preprocessor.transform(X_test)
    # Convert sparse to dense float64 NumPy array immediately
    X_test_processed = X_test_processed_sparse.toarray().astype(np.float64)

    logging.info(f"Processed X_train shape: {X_train_processed.shape}, dtype: {X_train_processed.dtype}")
    logging.info(f"Processed X_test shape: {X_test_processed.shape}, dtype: {X_test_processed.dtype}")
    return X_train_processed, X_test_processed # These are now dense float64 arrays
