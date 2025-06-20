import logging
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np # For type hinting ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Use logger for consistency

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    A custom transformer to cap outliers in numerical features based on quantiles.
    Values below lower_bound_quantile are capped at lower_bound_quantile_value.
    Values above upper_bound_quantile are capped at upper_bound_quantile_value.
    This helps in robust scaling and model stability, especially for features like BMI.
    """
    def __init__(self, lower_bound_quantile=0.01, upper_bound_quantile=0.99):
        self.lower_bound_quantile = lower_bound_quantile
        self.upper_bound_quantile = upper_bound_quantile
        self.lower_bound_values = {}
        self.upper_bound_values = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # Attempt to convert to DataFrame, assuming column names might be lost
            # from previous steps (e.g., from numpy array output of ColumnTransformer).
            # If X is already an array, we need to ensure it has columns for quantile.
            # However, this capper is designed to run *before* OneHotEncoder, so X should be DataFrame.
            logger.warning("OutlierCapper received non-DataFrame input. Attempting conversion.")
            X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])]) # Dummy columns if none

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Only calculate bounds for non-null values
                self.lower_bound_values[col] = X[col].quantile(self.lower_bound_quantile)
                self.upper_bound_values[col] = X[col].quantile(self.upper_bound_quantile)
            else:
                self.lower_bound_values[col] = None # Store None for non-numeric columns
                self.upper_bound_values[col] = None
        logger.info(f"OutlierCapper fitted for numeric columns. Lower bounds: {self.lower_bound_values}, Upper bounds: {self.upper_bound_values}")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if not isinstance(X_transformed, pd.DataFrame):
            logger.warning("OutlierCapper received non-DataFrame input for transform. Attempting conversion.")
            # If X_transformed is a numpy array, it needs to retain column names from fit.
            # This is why OutlierCapper should ideally be applied before ColumnTransformer converts to numpy.
            # Assuming columns from fit are available, or passed via context.
            if hasattr(X, 'columns'):
                X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
            else:
                # This path indicates a potential issue in the pipeline ordering
                logger.error("OutlierCapper received a NumPy array without column names for transform. Skipping outlier capping.")
                return X_transformed # Return unchanged if columns are unknown

        for col in X_transformed.columns:
            if pd.api.types.is_numeric_dtype(X_transformed[col]) and col in self.lower_bound_values and self.lower_bound_values[col] is not None:
                # Apply clipping, handling potential NaNs gracefully
                X_transformed[col] = X_transformed[col].clip(
                    lower=self.lower_bound_values[col],
                    upper=self.upper_bound_values[col]
                )
        logger.info("OutlierCapper transformed data.")
        return X_transformed
        
def split_data(
    df: pd.DataFrame,
    duration_column: str,
    event_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets, explicitly separating features (X),
    duration, and event status for time-to-event modeling.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing features, duration, and event.
        duration_column (str): The name of the column containing time to event/censoring.
        event_column (str): The name of the column indicating event occurrence (1) or censoring (0).
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        stratify_by (pd.Series or None): If not None, data is split in a stratified fashion based on this variable.
                                         Typically, this would be the event_column to ensure similar event rates.

    Returns:
        Tuple: (X_train, duration_train, event_train, X_test, duration_test, event_test)
               where X are DataFrames and duration/event are Series.

    Raises:
        KeyError: If duration_column or event_column are not in df.
        ValueError: If df is empty or has insufficient rows for splitting.
        TypeError: If input df is not a pandas DataFrame.
    """
    logger.info("Starting data splitting for time-to-event data...")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    if duration_column not in df.columns:
        raise KeyError(f"Duration column '{duration_column}' does not exist in the DataFrame.")
    if event_column not in df.columns:
        raise KeyError(f"Event column '{event_column}' does not exist in the DataFrame.")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    if len(df) < 2:
        raise ValueError("DataFrame must have at least 2 rows to split.")

    # Separate features (X), duration, and event status (y)
    # The dataframe passed to this function from model.py should already contain
    # only the feature columns + duration_column + event_column.
    X = df.drop(columns=[duration_column, event_column])
    duration = df[duration_column]
    event = df[event_column]

    # Optional: Check for missing values and log warnings in features
    if X.isnull().any().any():
        logger.warning("There are missing values in the features (X) *before* splitting. Consider handling them.")
    if duration.isnull().any():
        logger.warning("There are missing values in the duration column *before* splitting. This will cause issues for lifelines.")
    if event.isnull().any():
        logger.warning("There are missing values in the event column *before* splitting. This will cause issues for lifelines.")

    try:
        # Perform the split on X, duration, and event simultaneously to keep rows aligned
        X_train, X_test, duration_train, duration_test, event_train, event_test = train_test_split(
            X, duration, event, test_size=test_size, random_state=random_state, stratify=stratify_by
        )
    except ValueError as e:
        logger.error(f"Error during train/test split: {e}")
        raise

    logger.info(
        f"Data split complete: {len(X_train)} train samples, {len(X_test)} test samples."
    )
    # Return features as DataFrames and duration/event as Series
    return X_train, duration_train, event_train, X_test, duration_test, event_test

def create_preprocessor(X_train: pd.DataFrame) -> Pipeline:
    """
    Creates and fits a scikit-learn preprocessing pipeline based on the training data.
    This pipeline handles numerical imputation/scaling, outlier capping, and categorical one-hot encoding.

    Parameters:
        X_train (pd.DataFrame): The training features DataFrame, used to fit the preprocessor.

    Returns:
        sklearn.pipeline.Pipeline: A fitted preprocessing pipeline (ColumnTransformer).
    """
    logger.info("Creating feature preprocessing pipeline...")

    # Identify numeric and categorical features
    # Ensure numeric_features explicitly handles float, int types, and excludes object/category
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Remove any columns from numeric_features that might accidentally be boolean or object if type detection isn't perfect
    numeric_features = [f for f in numeric_features if X_train[f].dtype != 'object' and X_train[f].dtype != 'category' and X_train[f].dtype != 'bool']
    # Add booleans to categorical features for one-hot encoding if needed
    for f in X_train.select_dtypes(include='bool').columns.tolist():
        if f not in categorical_features:
            categorical_features.append(f)


    if not numeric_features and not categorical_features:
        logger.warning("No numeric or categorical features identified for preprocessing. Returning a dummy preprocessor.")
        return Pipeline(steps=[('passthrough', 'passthrough')]) # Return a passthrough pipeline

    # Define preprocessing pipelines for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Impute with mean
        ('outlier_capper', OutlierCapper(lower_bound_quantile=0.01, upper_bound_quantile=0.99)), # Custom outlier capping
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
    
    # Fit the preprocessor to the training data to learn parameters (e.g., means, scales, categories)
    preprocessor.fit(X_train)
    logger.info("Feature preprocessing pipeline created and fitted.")
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
    logger.info("Applying preprocessing to training data...")
    # .fit_transform() for train data (even if preprocessor is already fitted, it re-fits and transforms)
    # This assumes create_preprocessor returns a fitted preprocessor, so we should just transform here.
    X_train_processed_sparse = preprocessor.transform(X_train) # Just transform, not fit_transform

    # Convert sparse (if applicable) to dense float64 NumPy array immediately
    # ColumnTransformer can return sparse matrix if OneHotEncoder is used
    if hasattr(X_train_processed_sparse, 'toarray'):
        X_train_processed = X_train_processed_sparse.toarray().astype(np.float64)
    else:
        X_train_processed = X_train_processed_sparse.astype(np.float64)

    logger.info("Applying preprocessing to test data...")
    X_test_processed_sparse = preprocessor.transform(X_test)
    if hasattr(X_test_processed_sparse, 'toarray'):
        X_test_processed = X_test_processed_sparse.toarray().astype(np.float64)
    else:
        X_test_processed = X_test_processed_sparse.astype(np.float64)


    logger.info(f"Processed X_train shape: {X_train_processed.shape}, dtype: {X_train_processed.dtype}")
    logger.info(f"Processed X_test shape: {X_test_processed.shape}, dtype: {X_test_processed.dtype}")
    return X_train_processed, X_test_processed
