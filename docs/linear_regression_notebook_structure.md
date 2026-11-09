# Linear Regression Notebook Structure

The headers to start a new linear regression notebook with, a rule of thumb for the workflow, and the existing notebooks to look at for details.

*Source: a graphify query on 2026-11-08, `/graphify query "find me structure to use for linear regression to start a new ipynb..."`. See [graphify_changelog.md](graphify_changelog.md) for the graph build.*

The best model is [linear_regression/life_expectancy.ipynb](../linear_regression/life_expectancy.ipynb). In the graph it's the center of the "Data Cleaning Notebooks" community. It has the fullest structure in the repo: two rounds of EDA and preprocessing, assumption checks, and a residual-based evaluation. The headers below follow it, cut down to a rule of thumb.

## Headers

1. **Setup**
   - Imports
   - Reproducibility (random seeds, shuffle state)
   - Utils (import from `utils`, see [the note below](#note-on-utils))
2. **Load Dataset**
   - a `load_dataset()` function that returns a DataFrame
3. **Exploratory Data Analysis (EDA)**
   - General: size and shape, data types, `describe()`
   - Missing data: NaN, nulls, infinite values
4. **Preprocessing**
   - Dropping
   - Imputing
   - Confirm: run assert tests such as `test_no_missing_values`
5. **Feature Engineering**
   - group new features by domain (life_expectancy uses Population, Lifestyle, Economy, Death Ratio)
6. **2nd EDA: assumption checks**
   - Linearity check: plot each predictor against the target
   - Multicollinearity
   - Outliers (box plots)
7. **2nd Preprocessing**
   - Scaling and transformation
   - Remove infinite and extreme values
   - IQR outlier removal
   - Feature selection, keeping only numeric, non-collinear features
   - Encoding
8. **Training**
   - Train/test split, checked with `test_train_test_split_ratio`
   - Train
9. **Evaluate**
   - Metrics: MAE, MSE or RMSE, R², MAPE, and a note on why you chose them
   - an `evaluate_model()` function you can reuse
   - Plot of actual vs predicted values
   - Residual scatter plot
   - Residual histogram
10. **Feature Influence**
    - coefficients and what they mean
11. **Next Steps**

## Rule of thumb

EDA → clean → engineer → check assumptions → clean again → split and test → train → evaluate with metrics and residual plots → interpret the coefficients.

Steps 6 and 7 are what set regression apart from the classification notebooks. Linear regression assumes linear relationships, little collinearity and numeric inputs, so check those before training.

## Examples to reference

| Notebook | Use it for |
|---|---|
| [linear_regression/life_expectancy.ipynb](../linear_regression/life_expectancy.ipynb) | The full template above: IQR removal, multicollinearity-based feature selection, residual plots, and the MAE vs MSE vs RMSE write-ups |
| [coursera/c1_w2.ipynb](../coursera/c1_w2.ipynb) | A short, clean version: load → prepare → split → train → MSE → R² → reusable `evaluate_model()` → coefficients. A good starting skeleton |
| [coursera/c1_w1.ipynb](../coursera/c1_w1.ipynb) | Comparing a linear regression baseline with k-NN, plus 5-fold cross-validation (`cross_val_knn`) |
| [coursera/c2_w2.ipynb](../coursera/c2_w2.ipynb) | PCA before regression (`pca_mse()`), for the dimensionality-reduction step |
| [explorations/lr_mse_exploration.ipynb](../explorations/lr_mse_exploration.ipynb) | MAE and MSE worked out by hand, in NumPy and in sklearn, plus residuals added back to the DataFrame |

## Note on utils

The graph found that `life_expectancy.ipynb` defines its own copies of every function in [utils/eda.py](../utils/eda.py) and [utils/tests.py](../utils/tests.py), such as `check_columns_for_nulls`, `check_dataset_size_and_shape` and `test_train_test_split_ratio`. The copies and the originals are separate nodes in different communities, with no link between them. In a new notebook, import them instead, so you don't end up with a third copy:

```python
from utils.eda import check_dataset_size_and_shape, check_columns_for_nulls, check_columns_for_na, check_columns_for_inf, show_columns_with_missing_values
from utils.tests import test_no_missing_values, test_no_inf_columns, test_train_test_split_ratio, test_only_numeric_columns
```

Some functions exist only in `life_expectancy.ipynb` and aren't in `utils` yet. These are the ones worth moving next:

| Function | Possible new home |
|---|---|
| `remove_rows_with_inf` | `utils/preprocessing.py` |
| `apply_iqr_outlier_removal` | `utils/preprocessing.py` |
| `plot_predictors_vs_target` | `utils/plots.py` |
| `plot_boxplots_of_numeric_features` | `utils/plots.py` |
| `plot_actual_vs_predicted` | `utils/plots.py` |
| `plot_residuals` | `utils/plots.py` |
| `plot_residual_histogram` | `utils/plots.py` |
