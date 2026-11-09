"""Binary classification evaluation helpers.

Adapted from ``nlp/word_embeddings_002/word_embeddings_classification.ipynb``
(``evaluate_binary``, ``best_threshold_by_f1``) and
``classification/support_vector_machine/fradulent/fradulent.ipynb`` (confusion
matrix plots). Those notebooks keep their own copies.

Differences from the originals:
- ``evaluate_binary`` takes ``y_true`` / ``y_prob`` arrays instead of a Keras model
  and ``tf.data`` dataset, and uses ``sklearn.metrics.roc_auc_score`` instead of
  ``tf.keras.metrics.AUC``, so it works with any model that has ``predict_proba``.
- The confusion matrix plot is a function, drawn with matplotlib only, and works for
  any number of classes.

``evaluate_multiclass`` is new (no source notebook): metrics for single-label
multiclass problems, where ``evaluate_binary`` and the threshold sweep don't apply.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Confusion-matrix counts and metrics for a binary classifier.

    Parameters
    ----------
    y_true : array-like of 0/1
        True labels.
    y_prob : array-like of float
        Predicted probability of the positive class, e.g. ``model.predict_proba(X)[:, 1]``.
    threshold : float
        Probabilities at or above this are predicted as the positive class.

    Returns
    -------
    dict
        tp, tn, fp, fn, accuracy, precision, recall, f1 and auc. AUC uses the
        probabilities, so it does not depend on the threshold.
    """
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp, tn, fp, fn = _counts(y_true, y_pred)

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(float(accuracy), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "auc": round(float(auc), 3),
    }


def best_threshold_by_f1(
    y_true, y_prob, thresholds: Sequence[float] = np.arange(0.1, 0.91, 0.05)
) -> Tuple[Tuple[float, float, float, float, float], List[Tuple[float, float, float, float, float]]]:
    """Sweep decision thresholds and return the one with the best F1.

    Returns
    -------
    (best_row, rows)
        Each row is ``(threshold, accuracy, precision, recall, f1)``.
        ``pd.DataFrame(rows, columns=[...])`` makes the sweep easy to read.
    """
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp, tn, fp, fn = _counts(y_true, y_pred)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(y_true), 1)
        rows.append((float(t), float(accuracy), float(precision), float(recall), float(f1)))
    return max(rows, key=lambda x: x[4]), rows


def evaluate_multiclass(y_true, y_pred, y_prob=None, classes: Optional[Sequence] = None) -> Dict[str, float]:
    """Metrics for a single-label multiclass classifier (also works for binary).

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted class labels (strings or ints).
    y_prob : array-like of shape (n_samples, n_classes), optional
        ``model.predict_proba(X)``. Columns must follow ``classes`` (use ``model.classes_``).
        Needed for ROC AUC and log loss.
    classes : sequence, optional
        Class order matching the columns of ``y_prob``.

    Returns
    -------
    dict
        accuracy; macro-averaged precision, recall and F1 (every class counts equally,
        so rare classes matter); weighted F1 (averaged by class size); and, if
        ``y_prob`` is given, one-vs-rest macro ROC AUC and log loss.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision_macro": round(float(precision), 3),
        "recall_macro": round(float(recall), 3),
        "f1_macro": round(float(f1), 3),
        "f1_weighted": round(float(f1_weighted), 3),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        labels = list(classes) if classes is not None else None
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
        out["roc_auc_ovr_macro"] = round(float(auc), 3)
        out["log_loss"] = round(float(log_loss(y_true, y_prob, labels=labels)), 3)
    return out


def plot_confusion_matrix(y_true, y_pred, labels: Optional[Sequence] = None) -> np.ndarray:
    """Plot the confusion matrix as counts and as row-normalized percentages.

    Works for binary and multiclass labels. Rows are the true class and columns the
    predicted class, so each row of the normalized matrix shows how that true class
    was split across predictions.

    Parameters
    ----------
    labels : sequence, optional
        Class order for the rows and columns (e.g. ``model.classes_``). Defaults to the
        sorted classes found in ``y_true`` and ``y_pred``.

    Returns
    -------
    np.ndarray
        The count matrix.
    """
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()), key=str)
    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    size = max(4.5, 0.9 * len(labels) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * size, size))
    for ax, matrix, title, fmt in (
        (axes[0], cm, "Confusion Matrix (counts)", "{:d}"),
        (axes[1], cm_norm, "Confusion Matrix (% of true class)", "{:.1%}"),
    ):
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), [str(l) for l in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), [str(l) for l in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        threshold = matrix.max() / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()

    return cm
