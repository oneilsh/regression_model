import pandas as pd
import numpy as np
from typing import List

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise RuntimeError(f"File not found: {filepath}")
    except pd.errors.ParserError:
        raise RuntimeError(f"Failed to parse CSV file: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def filter_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Filter dataframe to retain only specified columns."""
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in DataFrame: {missing_columns}")
    return df[columns].copy()

def stratify_by_risk(df: pd.DataFrame, risk_column: str, threshold: float) -> pd.DataFrame:
    """Stratify dataset into high vs. low risk groups based on threshold."""
    if risk_column not in df.columns:
        raise KeyError(f"Risk column '{risk_column}' does not exist in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[risk_column]):
        raise ValueError(f"Risk column '{risk_column}' must contain numeric data.")

    df = df.copy()
    df['risk_group'] = np.where(df[risk_column] >= threshold, 'high', 'low')
    return df
