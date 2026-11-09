"""Plotting helpers for regression EDA and evaluation.

Copied from ``linear_regression/life_expectancy.ipynb`` (Utils → Plots and the
Evaluate section) so new notebooks can import them instead of redefining them. The
notebook still keeps its own copies.

The evaluation plots take a ``target_label`` for the axis text. In the original
notebook it was hard-coded to "Life Expectancy (years)".
"""
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictors_vs_target(df: pd.DataFrame, target: str) -> None:
    """Plot scatter plots of all numeric predictors against the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    target : str
        The name of the target variable column.

    Notes
    -----
    - Automatically selects all numeric columns except the target
    - Lays them out in a 4-column grid
    - Uses alpha=0.3 and small point size to handle overplotting
    - Hides leftover empty subplot panels
    """
    predictors = df.select_dtypes(include='number').columns.drop(target).tolist()

    n_cols = 4
    n_rows = -(-len(predictors) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(predictors):
        axes[i].scatter(df[col], df[target], alpha=0.3, s=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'{col} vs {target}')

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_boxplots_of_numeric_features(df: pd.DataFrame, subtitle: str = 'Box-Whisker Plots (pre-removal)') -> List[str]:
    """Plot boxplots of all numeric features to visually inspect for outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    subtitle : str
        Title shown above the grid.

    Returns
    -------
    List[str]
        The numeric columns that were plotted, arranged in a 4-column grid.
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    n_cols = 4
    n_rows = -(-len(numeric_cols) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True)  # vertical by default; `vert=` is deprecated in matplotlib 3.10
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(subtitle, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return numeric_cols


def plot_actual_vs_predicted(model, X_test, y_test, target_label: str = 'Target') -> None:
    """Scatter actual vs predicted values with a perfect-fit reference line.

    In a perfect model every point falls on the diagonal.
    """
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=15)

    # Perfect-prediction reference line
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect fit')

    ax.set_xlabel(f'Actual {target_label}')
    ax.set_ylabel(f'Predicted {target_label}')
    ax.set_title('Actual vs Predicted')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(model, X_test, y_test, target_label: str = 'Target') -> None:
    """Scatter residuals (actual − predicted) against predicted values.

    What to look for:
    - Random scatter around zero → model assumptions are met
    - Funnel shape → heteroscedasticity
    - Curved pattern → a non-linear relationship the model isn't capturing
    - Points far from zero → predictions the model got badly wrong
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel(f'Predicted {target_label}')
    ax.set_ylabel('Residual (Actual − Predicted)')
    ax.set_title('Residual Plot')
    plt.tight_layout()
    plt.show()


def plot_residual_histogram(model, X_test, y_test, bins: int = 30) -> None:
    """Histogram of residuals (actual − predicted).

    What to look for:
    - Symmetric bell shape centered at 0 → residuals are roughly normal
    - Skewed left or right → the model systematically over- or under-predicts
    - Bimodal or flat → the model may be missing an important feature
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=bins, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', linewidth=1, label='Zero error')
    ax.set_xlabel('Residual (Actual − Predicted)')
    ax.set_ylabel('Count')
    ax.set_title('Residual Histogram')
    ax.legend()
    plt.tight_layout()
    plt.show()
