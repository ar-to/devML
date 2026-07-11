from typing import Tuple, Dict, List
import pandas as pd

from utils.eda import check_dataset_size_and_shape

def test_no_missing_values(df: pd.DataFrame, columns: list) -> None:
  """Assert that there are no missing values in the specified columns.
  
  The double .sum() first sums per column, then sums across all specified columns into one total. If that total is not zero, the assert raises an AssertionError with the message. If the cell runs silently, all specified columns are clean.
  """
  assert df[columns].isna().sum().sum() == 0, f"Missing values found in {', '.join(columns)}"

def test_no_inf_columns(columns_with_inf: list) -> None:
  """Assert that no numeric columns contain +inf or -inf."""
  assert len(columns_with_inf) == 0, f"Columns still containing inf values: {columns_with_inf}"

def test_train_test_split_ratio(df: pd.DataFrame, X_train, X_test, expected_test_size: float = 0.25, tolerance: float = 0.01) -> None:
    """Assert that the train/test split is approximately the expected ratio."""
    check_dataset_size_and_shape(df)

    total = len(X_train) + len(X_test)
    actual_test_ratio = len(X_test) / total
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Test ratio: {actual_test_ratio:.2%}")
    assert abs(actual_test_ratio - expected_test_size) <= tolerance, (
        f"Expected test ratio ~{expected_test_size:.0%}, got {actual_test_ratio:.2%}"
    )
  
def test_only_non_numeric_columns(df: pd.DataFrame) -> None:
  """Assert that all columns in df are non-numeric.

  Not used for linear regression, but could be useful for verifying that feature selection has successfully removed all numeric columns, leaving only categorical features.
  
  Useful for verifying that all numeric columns have been dropped during feature selection, leaving only non-numeric columns (e.g., categorical features) in the DataFrame. If any numeric columns are found, the assert will raise an AssertionError with the list of offending columns. If the cell runs without error, it confirms that all columns are non-numeric as expected.
  """
  numeric_cols = df.select_dtypes(include="number").columns.tolist()
  assert len(numeric_cols) == 0, f"Numeric columns found: {numeric_cols}"
  print("All columns are non-numeric.")

def test_only_numeric_columns(df: pd.DataFrame) -> None:
  """Assert that all columns in df are numeric only.
  
  This is important for linear regression, which requires numeric input features. If any non-numeric columns are found, the assert will raise an AssertionError with the list of offending columns. If the cell runs without error, it confirms that all columns are numeric as expected.

  """
  numeric_cols = df.select_dtypes(include="number").columns.tolist()
  assert len(numeric_cols) == len(df.columns), f"Non-numeric columns found: {set(df.columns) - set(numeric_cols)}"
  print("All columns are numeric.")
