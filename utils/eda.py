import pathlib
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

def dummy_error_to_stop_runs() -> None:
  """A dummy function that raises an error to stop the notebook from running further."""
  raise RuntimeError("This is a dummy error to stop the notebook from running further. Please remove this function call to continue.")

def show_columns_with_missing_values(df: pd.DataFrame) -> List[str]:
  """Print the columns in the DataFrame that contain missing values and the number of missing values in each.
  
  Returns
  -------
    List[str]
        A list of column names that contain missing values.
  """

  missing = df.isna().sum()[df.isna().sum() > 0]
  print("Columns with missing values:")
  print(missing)
  return missing.index.tolist()

def show_count_of_columns_missing_values(df: pd.DataFrame, columns: list) -> None:
    """Display the count of missing values for specified columns.
    
    # any row where *either* is NaN will also produce NaN in a custom feature dependent on these columns.
    
    """
    print(df[columns].isna().sum())

def check_columns_for_nulls(df: pd.DataFrame) -> None:
  """Check for null values in the DataFrame and print the columns that contain them."""
  print(df.isnull().any())

def check_columns_for_na(df: pd.DataFrame) -> None:
  """Check for NA values in the DataFrame and print the columns that contain them."""
  # print(df.isna().any())
  print(df.isna().sum())

def check_columns_for_inf(df: pd.DataFrame) -> List[str]:
  """Check for infinite values in the DataFrame and print the columns that contain them."""
  numeric = df.select_dtypes(include='number')

  print("Positive inf counts:")
  print((numeric == np.inf).sum())

  print("\nNegative inf counts:")
  print((numeric == -np.inf).sum())

  # Build list of numeric columns that contain +inf or -inf
  inf_mask = np.isinf(df.select_dtypes(include='number')).any(axis=0)
  columns_with_inf = inf_mask[inf_mask].index.tolist()

  print("Columns with inf values:", columns_with_inf)

  return columns_with_inf

def check_dataset_size_and_shape(df: pd.DataFrame):
  print(f"Number of rows: {len(df)}")
  print(f"Dataset shape: {df.shape}")
  print(f"Number of rows (alternative): {df.shape[0]}")


