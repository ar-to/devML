# Logistic Regression Notebook Structure

The headers to start a new logistic regression notebook with, a rule of thumb for the workflow, and the existing notebooks to look at for details.

*Source: a graphify query on 2026-11-08, `/graphify query "find me structure to use for logistic regression to start a new ipynb..."`. See [graphify_changelog.md](graphify_changelog.md) for the graph build and [linear_regression_notebook_structure.md](linear_regression_notebook_structure.md) for the regression version this builds on.*

Unlike linear regression, the repo has **no dedicated logistic regression notebook**. `LogisticRegression` only shows up as a baseline in [coursera/c3_w1.ipynb](../coursera/c3_w1.ipynb) and in the multi-algorithm sections of [first_model/model_case_rate_002.ipynb](../first_model/model_case_rate_002.ipynb). What the repo does have is the classification structure: [classification/support_vector_machine/fradulent/fradulent.ipynb](../classification/support_vector_machine/fradulent/fradulent.ipynb) has the evaluation flow, and the word_embeddings_002 notebook has threshold and probability tools.

The headers below combine those with the linear regression template. Steps that carry over unchanged are marked "same".

A notebook that follows this structure: [capstone/logistic_regression.ipynb](../capstone/logistic_regression.ipynb).

## Headers

1. **Setup** (same)
   - Imports
   - Reproducibility
   - Utils (import from `utils`)
2. **Load Dataset**
3. **Exploratory Data Analysis (EDA)**
   - General: size and shape, data types, `describe()`
   - Missing data
   - **Class balance**: count and share of each target class. This decides how the split and the metrics are set up.
4. **Preprocessing**
   - Dropping
   - Imputing
   - **Encoding**: one-hot encode categorical features (the binary target becomes 0/1; a multiclass target keeps its labels). Done here, not in 2nd Preprocessing, so the encoded columns are numeric and get included in the 2nd EDA checks. Keep a list of continuous columns and a list of encoded columns.
   - Confirm: no missing values, every feature numeric
5. **Feature Engineering**
6. **2nd EDA: assumption checks**
   - Multicollinearity, including the encoded columns
   - Outliers (continuous columns only)
   - Features compared across classes: box plots by class for continuous columns, share of 1s per class for encoded columns
   - The target doesn't need to be linear in the features. What should be roughly linear is the log-odds of the target, so the "plot each predictor against the target" check doesn't apply.
7. **2nd Preprocessing**
   - Remove infinite and extreme values
   - IQR outlier removal (optional), **on continuous columns only**. On a 0/1 column, IQR marks the less common value as an outlier and drops those rows.
   - Feature selection
   - Scaling: fit on the training set only. Scaling matters more here than in linear regression, because regularization (on by default) and the solver are sensitive to feature scale.
8. **Training**
   - **Stratified** train/test split: `stratify=y`, so both sets keep the same class ratio
   - Baseline: always predict the majority class (`DummyClassifier`)
   - Train `LogisticRegression(max_iter=1000, random_state=...)`. Add `class_weight="balanced"` if the classes are imbalanced.
9. **Evaluate**
   - Metrics: accuracy, precision, recall, F1, ROC AUC, and a note on why you chose them. On imbalanced data, accuracy alone is misleading.
   - `classification_report`
   - Confusion matrix, as counts and normalized to %
   - ROC curve and precision-recall curve
   - **Threshold analysis**: sweep the cutoff from 0.1 to 0.9 on `predict_proba` and pick the best by F1 or whatever the problem needs. 0.5 isn't automatically right.
10. **Feature Influence**
    - coefficients as log-odds, and `np.exp(coef_)` as odds ratios
11. **Next Steps**

## Rule of thumb

EDA (check class balance) → clean and encode → engineer → check assumptions (now covering the encoded columns) → clean again → scale → stratified split → baseline → train → evaluate with the confusion matrix, ROC/PR curves and a threshold sweep → interpret the odds ratios.

The main differences from linear regression:
- the class-balance check
- the stratified split
- the baseline to compare against
- probabilities and a threshold, rather than residuals
- coefficients read as odds ratios, not slopes

## Multiclass (single-label)

When the target has more than two categories and each row gets exactly one, the structure stays the same. These parts change:

| Section | Binary | Multiclass |
|---|---|---|
| Class balance tests | exactly 2 classes | at least 2 classes, no missing labels, enough rows per class for a stratified split |
| Target encoding | 0/1 | keep the class labels; sklearn handles strings |
| Model | one set of coefficients | multinomial (softmax): one set of coefficients per class |
| Metrics | precision, recall, F1, AUC for the positive class | accuracy against chance (`1 / n_classes`); **macro** and weighted precision/recall/F1; one-vs-rest macro AUC; log loss |
| Curves | one ROC and one PR curve | one ROC and PR curve per class (one-vs-rest) |
| Threshold sweep | tune the 0.5 cutoff | no single threshold (the top-probability class wins), so check **prediction confidence** buckets instead |
| Feature influence | one coefficient and odds ratio per feature | a coefficient table of features × classes; read it per class |

`utils.classification.evaluate_multiclass` and `plot_confusion_matrix` cover the multiclass metrics and plot. [capstone/logistic_regression.ipynb](../capstone/logistic_regression.ipynb) is set up as multiclass on `domain_context`.

## Examples to reference

| Notebook | Use it for |
|---|---|
| [classification/support_vector_machine/fradulent/fradulent.ipynb](../classification/support_vector_machine/fradulent/fradulent.ipynb) | The closest classification template: EDA with class counts, encoding categoricals, `class_weight='balanced'`, `classification_report`, and the confusion matrix as counts and as %. Swap `LinearSVC` for `LogisticRegression`. Its README links the Keras imbalanced-classification example. |
| [coursera/c3_w1.ipynb](../coursera/c3_w1.ipynb) | A `LogisticRegression` baseline on scaled data, compared with a neural net, plus a hand-written `sigmoid()` and manual forward pass. Shows what logistic regression computes. |
| [nlp/word_embeddings_002/word_embeddings_classification.ipynb](../nlp/word_embeddings_002/word_embeddings_classification.ipynb) | `evaluate_binary()` (TP/TN/FP/FN, precision, recall, F1, AUC) and `best_threshold_by_f1()`, a 0.10–0.90 sweep. `binary_entropy_bits()` scores how uncertain a prediction is. The notes are in [prompt_threshold.md](../nlp/word_embeddings_002/prompt_threshold.md) and [prompt_entropy.md](../nlp/word_embeddings_002/prompt_entropy.md). |
| [nlp/word_embeddings_001/prompt_class_eval.md](../nlp/word_embeddings_001/prompt_class_eval.md) | Notes on reading a confusion matrix, precision/recall/F1 and confidence buckets |
| [first_model/model_case_rate_002.ipynb](../first_model/model_case_rate_002.ipynb) | `LogisticRegression` alongside other classifiers, a regression vs classification comparison, and SHAP analysis for a linear model (feature influence) |
| [classification/decision_tree/decision_tree_001.ipynb](../classification/decision_tree/decision_tree_001.ipynb) | Synthetic data with a seeded generator, and plots of predicted probabilities. Useful for a quick sanity test. |

## Note on utils

Everything from the linear regression notebook carries over: `utils.eda`, `utils.tests`, `utils.preprocessing`, and the box-plot function in `utils.plots`.

Classification evaluation is in [utils/classification.py](../utils/classification.py). These are copies; the source notebooks keep their own versions.

| Function | Comes from | Change |
|---|---|---|
| `evaluate_binary(y_true, y_prob, threshold)` | word_embeddings_002 notebook | Takes arrays instead of a Keras model and dataset; AUC from `sklearn.metrics.roc_auc_score` instead of TensorFlow |
| `best_threshold_by_f1(y_true, y_prob)` | word_embeddings_002 notebook | None (pure NumPy) |
| `plot_confusion_matrix(y_true, y_pred, labels)` | fradulent notebook | Turned into a function; counts and % side by side |
