"""Preprocessing helpers for tabular regression work.

Copied from ``linear_regression/life_expectancy.ipynb`` (Utils → Mutations) so new
notebooks can import them instead of redefining them. The notebook still keeps its
own copies.
"""
from typing import List

import numpy as np
import pandas as pd


def remove_rows_with_inf(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove rows from the DataFrame where any of the specified columns contain positive or negative infinity values.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to clean.
    columns : List[str]
        A list of column names to check for infinity values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with rows containing infinity values in the specified columns removed.
    """
    for col in columns:
        df = df[df[col] != np.inf]
        df = df[df[col] != -np.inf]
    return df


def apply_iqr_outlier_removal(df: pd.DataFrame, numeric_cols: List[str], exclude: set[str]) -> pd.DataFrame:
    """Apply IQR outlier removal to the DataFrame, excluding specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    numeric_cols : List[str]
        A list of numeric column names to consider for outlier removal.
    exclude : set
        A set of column names to exclude from outlier removal (e.g. the target).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with outliers removed based on the IQR method, excluding specified columns.

    Notes
    -----
    - Only applies IQR to numeric columns that are not in the exclude set.
    - Prints the number of rows removed and the before/after row counts.
    - Applied column by column, so rows drop out cumulatively. On many columns this can
      remove a large share of the data; check the printed counts.
    """
    outlier_cols = [col for col in numeric_cols if col not in exclude]

    rows_before = len(df)

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    rows_after = len(df)
    print(f"Rows removed: {rows_before - rows_after}  ({rows_before} → {rows_after})")
    return df
