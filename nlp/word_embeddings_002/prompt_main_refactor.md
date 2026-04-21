User: how do i take the classification task model and evaluation from nlp/word_embeddings_001/word_embeddings_classification.ipynb and apply to this notebook that runs multiple embedding models so I can compare how contemporary and shakespeare embeddings perform on the moview review classification task.

GitHub Copilot: I’ll inspect both notebooks to map the training/evaluation flow from the single-model version into your multi-embedding notebook, then give you exact cell-level code you can paste to compare contemporary vs Shakespeare embeddings on movie reviews.

Retrieving Notebook summary.

Retrieving Notebook summary.

I found the structure: your target notebook already has multi-model loops and plotting sections, so next I’m pulling the exact code blocks around training, prediction, and metrics in both notebooks to map them cleanly.

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_001/word_embeddings_classification.ipynb#480-480), lines 480 to 760

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#360-360), lines 360 to 738

I’ve confirmed where your multi-embedding results dictionary is built; now I’m checking the earlier setup cells to ensure the classification code I propose uses your existing imports/config objects without duplication.

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#1-1), lines 1 to 360

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_001/word_embeddings_classification.ipynb#1-1), lines 1 to 480

Next I’m pulling the remaining evaluation cells from the source notebook so your comparison notebook includes the same confusion-matrix/ROC style outputs, not just test accuracy.

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_001/word_embeddings_classification.ipynb#760-760), lines 760 to 911

Use the same classification architecture and evaluation logic from word_embeddings_classification.ipynb, word_embeddings_classification.ipynb, and word_embeddings_classification.ipynb, then loop it over the two embedding results you already produce in word_embeddings_classification.ipynb.

Add the following cells right after word_embeddings_classification.ipynb.

    # Movie review classification comparison using pretrained embeddings
    from pathlib import Path
    import tarfile
    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from tensorflow.keras import layers

    SEED = 42
    AUTOTUNE = tf.data.AUTOTUNE
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Map corpus names to config so each embedding uses its own token standardization
    config_by_name = {
        "Shakespeare": shakespeare_config,
        "Contemporary": contemporary_config,
    }

    def load_imdb_raw_datasets(seed=42):
        imdb_cache_dir = Path.cwd() / ".tf_data"
        imdb_cache_dir.mkdir(parents=True, exist_ok=True)

        dataset_archive = Path(tf.keras.utils.get_file(
            fname="aclImdb_v1.tar.gz",
            origin="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            cache_dir=str(imdb_cache_dir),
            cache_subdir="",
            extract=False,
        ))

        dataset_dir = imdb_cache_dir / "aclImdb"
        if not dataset_dir.exists():
            with tarfile.open(dataset_archive, "r:gz") as tar:
                tar.extractall(path=imdb_cache_dir)

        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        def make_labeled_review_ds(split_dir, shuffle=False):
            pos_files = sorted((split_dir / "pos").glob("*.txt"))
            neg_files = sorted((split_dir / "neg").glob("*.txt"))
            file_paths = [str(p) for p in pos_files + neg_files]
            labels = [1] * len(pos_files) + [0] * len(neg_files)

            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            if shuffle and len(file_paths) > 1:
                ds = ds.shuffle(len(file_paths), seed=seed, reshuffle_each_iteration=False)

            ds = ds.map(
                lambda path, label: (tf.io.read_file(path), tf.cast(label, tf.float32)),
                num_parallel_calls=AUTOTUNE
            )
            return ds

        full_train_ds = make_labeled_review_ds(train_dir, shuffle=True)
        test_ds = make_labeled_review_ds(test_dir, shuffle=False)

        val_size = 5000
        val_ds = full_train_ds.take(val_size)
        train_ds = full_train_ds.skip(val_size)

        batch_size = 64
        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)
        return train_ds, val_ds, test_ds

    def build_classifier_from_embedding(results, cfg, sequence_length=200, trainable=False):
        vocab = results["vocab"]
        weights = results["weights"]

        standardize = build_standardizer(
            lowercase=cfg.lowercase,
            strip_punctuation=cfg.strip_punctuation,
            keep_apostrophes=cfg.keep_apostrophes,
        )

        review_vectorizer = layers.TextVectorization(
            standardize=standardize,
            output_mode="int",
            output_sequence_length=sequence_length,
            vocabulary=vocab,
        )

        model = tf.keras.Sequential([
            review_vectorizer,
            layers.Embedding(
                input_dim=len(vocab),
                output_dim=weights.shape[1],
                embeddings_initializer=tf.keras.initializers.Constant(weights),
                trainable=trainable,
                mask_zero=True,
                name="pretrained_embedding",
            ),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        return model

    def evaluate_binary(model, test_ds, threshold=0.5):
        y_true_batches, y_prob_batches = [], []
        for x_batch, y_batch in test_ds:
            y_prob = model.predict(x_batch, verbose=0).reshape(-1)
            y_true_batches.append(y_batch.numpy().reshape(-1))
            y_prob_batches.append(y_prob)

        y_true = np.concatenate(y_true_batches).astype(np.int32)
        y_prob = np.concatenate(y_prob_batches)
        y_pred = (y_prob >= threshold).astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(y_true), 1)

        auc_metric = tf.keras.metrics.AUC()
        auc_metric.update_state(y_true, y_prob)
        auc = float(auc_metric.result().numpy())

        return {
            "y_true": y_true,
            "y_prob": y_prob,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": auc,
        }

    def best_threshold_by_f1(y_true, y_prob, thresholds=np.arange(0.1, 0.91, 0.05)):
        rows = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(np.int32)
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            accuracy = (tp + tn) / max(len(y_true), 1)
            rows.append((float(t), float(accuracy), float(precision), float(recall), float(f1)))
        return max(rows, key=lambda x: x[4]), rows

    train_ds, val_ds, test_ds = load_imdb_raw_datasets(seed=SEED)

    classification_results = {}
    summary_rows = []

    for name, w2v_results in results_by_name.items():
        print("\n" + "=" * 72)
        print(f"Training classifier with {name} embeddings")

        model = build_classifier_from_embedding(
            results=w2v_results,
            cfg=config_by_name[name],
            sequence_length=200,
            trainable=False,  # keep frozen for clean embedding-quality comparison
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=8,
            callbacks=callbacks,
            verbose=1
        )

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        eval_stats = evaluate_binary(model, test_ds, threshold=0.5)
        best_t, _ = best_threshold_by_f1(eval_stats["y_true"], eval_stats["y_prob"])

        classification_results[name] = {
            "model": model,
            "history": history,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "eval": eval_stats,
            "best_threshold": best_t[0],
            "best_f1": best_t[4],
        }

        summary_rows.append((
            name,
            float(test_acc),
            float(test_auc),
            eval_stats["precision"],
            eval_stats["recall"],
            eval_stats["f1"],
            float(best_t[0]),
            float(best_t[4]),
        ))

    print("\nComparison Summary")
    print("name         test_acc  test_auc  precision  recall   f1@0.5   best_thr  best_f1")
    print("-" * 88)
    for r in summary_rows:
        print(f"{r[0]:12s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:9.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.2f} {r[7]:8.4f}")

    # Validation curves comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, payload in classification_results.items():
        h = payload["history"].history
        axes[0].plot(h.get("val_accuracy", []), marker="o", label=name)
        axes[1].plot(h.get("val_loss", []), marker="o", label=name)

    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle("IMDB Classification with Different Pretrained Embeddings")
    plt.tight_layout()
    plt.show()

How this maps to your current notebooks:
1. You already create the two embedding sets in word_embeddings_classification.ipynb, so the loop uses that dictionary directly.
2. The classifier architecture and metrics are the same pattern as word_embeddings_classification.ipynb.
3. Confusion style metrics and threshold sweep follow word_embeddings_classification.ipynb and word_embeddings_classification.ipynb.

Important for fair comparison:
1. Keep identical IMDB train, validation, and test splits for both runs.
2. Keep identical classifier head and training settings.
3. Keep embeddings frozen first. If you later unfreeze, compare that in a separate experiment since it changes the question from embedding quality to fine tuning behavior.

If you want, I can also give you a short follow-up cell that plots side by side confusion matrices for Shakespeare vs Contemporary.

User: yes please but before that can i make the training and eval similar to #sym:train_word2vec_pipeline and having config for classification too. How can i refactor so I can run the training, eval and plot in separate cells to reduce time and debug better. 

GitHub Copilot: Perfect—I'll refactor the classification into a config-driven pipeline pattern matching your word2vec structure. This lets you run training independently from eval/plotting and swap configs easily. 

Retrieving Notebook summary.

Now I'll replace the classification section with a clean config-driven pipeline. Here's the refactored structure:

**Replace cells starting after the "Classification Task" markdown cell with this pattern:**

```python
## Classification Config & Utilities

from dataclasses import dataclass

@dataclass
class ClassificationConfig:
    """Configuration for movie review classification task."""
    sequence_length: int = 200
    trainable_embedding: bool = False
    dense_units: int = 64
    dropout_rate: float = 0.3
    batch_size: int = 64
    epochs: int = 8
    early_stopping_patience: int = 2
    threshold: float = 0.5
    val_split: int = 5000  # number of examples to use for validation

classification_config = ClassificationConfig()

# Constants for IMDB dataset
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_CACHE_SUBDIR = ".tf_data"

def load_imdb_raw_datasets(seed=42, batch_size=64):
    """Load and prepare IMDB train/val/test splits."""
    imdb_cache_dir = Path.cwd() / IMDB_CACHE_SUBDIR
    imdb_cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_archive = Path(tf.keras.utils.get_file(
        fname="aclImdb_v1.tar.gz",
        origin=IMDB_URL,
        cache_dir=str(imdb_cache_dir),
        cache_subdir="",
        extract=False,
    ))

    dataset_dir = imdb_cache_dir / "aclImdb"
    if not dataset_dir.exists():
        with tarfile.open(dataset_archive, "r:gz") as tar:
            tar.extractall(path=imdb_cache_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"IMDB dataset directory not found after extraction: {dataset_dir}")

    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    def make_labeled_review_ds(split_dir, shuffle=False):
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        
        pos_files = sorted((split_dir / "pos").glob("*.txt"))
        neg_files = sorted((split_dir / "neg").glob("*.txt"))
        file_paths = [str(p) for p in pos_files + neg_files]
        labels = [1] * len(pos_files) + [0] * len(neg_files)

        if not file_paths:
            raise ValueError(
                f"No review files found under {split_dir}. Expected pos/*.txt and neg/*.txt."
            )

        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        if shuffle and len(file_paths) > 1:
            ds = ds.shuffle(len(file_paths), seed=seed, reshuffle_each_iteration=False)

        ds = ds.map(
            lambda path, label: (tf.io.read_file(path), tf.cast(label, tf.float32)),
            num_parallel_calls=AUTOTUNE
        )
        return ds

    full_train_ds = make_labeled_review_ds(train_dir, shuffle=True)
    test_ds = make_labeled_review_ds(test_dir, shuffle=False)

    val_size = classification_config.val_split
    val_ds = full_train_ds.take(val_size)
    train_ds = full_train_ds.skip(val_size)

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def build_classifier_from_embedding(results, cfg, config):
    """Build classification model using pretrained embeddings."""
    vocab = results["vocab"]
    weights = results["weights"]

    standardize = build_standardizer(
        lowercase=cfg.lowercase,
        strip_punctuation=cfg.strip_punctuation,
        keep_apostrophes=cfg.keep_apostrophes,
    )

    review_vectorizer = layers.TextVectorization(
        standardize=standardize,
        output_mode="int",
        output_sequence_length=config.sequence_length,
        vocabulary=vocab,
    )

    model = tf.keras.Sequential([
        review_vectorizer,
        layers.Embedding(
            input_dim=len(vocab),
            output_dim=weights.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(weights),
            trainable=config.trainable_embedding,
            mask_zero=True,
            name="pretrained_embedding",
        ),
        layers.GlobalAveragePooling1D(),
        layers.Dense(config.dense_units, activation="relu"),
        layers.Dropout(config.dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def evaluate_binary(model, test_ds, threshold=0.5):
    """Compute confusion matrix and metrics."""
    y_true_batches, y_prob_batches = [], []
    for x_batch, y_batch in test_ds:
        y_prob = model.predict(x_batch, verbose=0).reshape(-1)
        y_true_batches.append(y_batch.numpy().reshape(-1))
        y_prob_batches.append(y_prob)

    y_true = np.concatenate(y_true_batches).astype(np.int32)
    y_prob = np.concatenate(y_prob_batches)
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    eps = 1e-8
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps)

    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y_true, y_prob)
    auc = float(auc_metric.result().numpy())

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
    }


def best_threshold_by_f1(y_true, y_prob, thresholds=np.arange(0.1, 0.91, 0.05)):
    """Sweep thresholds and return best by F1."""
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(len(y_true), 1)
        rows.append((float(t), float(accuracy), float(precision), float(recall), float(f1)))
    return max(rows, key=lambda x: x[4]), rows


def train_classification_pipeline(results_by_name, config_by_name, imdb_datasets, clf_config):
    """
    Train classifiers using multiple embeddings on shared IMDB data.
    
    Args:
        results_by_name: dict keyed by corpus name -> word2vec results dict
        config_by_name: dict keyed by corpus name -> CorpusConfig
        imdb_datasets: tuple of (train_ds, val_ds, test_ds)
        clf_config: ClassificationConfig instance
    
    Returns:
        dict keyed by corpus name -> classification results dict
    """
    train_ds, val_ds, test_ds = imdb_datasets
    classification_results = {}
    summary_rows = []

    for name, w2v_results in results_by_name.items():
        print("\n" + "=" * 72)
        print(f"Training classifier with {name} embeddings")
        print("=" * 72)

        model = build_classifier_from_embedding(
            results=w2v_results,
            cfg=config_by_name[name],
            config=clf_config,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=clf_config.early_stopping_patience,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=clf_config.epochs,
            callbacks=callbacks,
            verbose=1
        )

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        eval_stats = evaluate_binary(model, test_ds, threshold=clf_config.threshold)
        best_t, threshold_sweep = best_threshold_by_f1(eval_stats["y_true"], eval_stats["y_prob"])

        classification_results[name] = {
            "model": model,
            "history": history,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "eval": eval_stats,
            "threshold_sweep": threshold_sweep,
            "best_threshold": best_t[0],
            "best_f1": best_t[4],
        }

        summary_rows.append((
            name,
            float(test_acc),
            float(test_auc),
            eval_stats["precision"],
            eval_stats["recall"],
            eval_stats["f1"],
            float(best_t[0]),
            float(best_t[4]),
        ))

    return classification_results, summary_rows
```

Now add these **separate execution cells**:

```python
## Train Classifiers (this runs the pipeline)

# Load IMDB dataset once
train_ds, val_ds, test_ds = load_imdb_raw_datasets(seed=SEED, batch_size=classification_config.batch_size)
print(f"Dataset loaded: train {len(train_ds)} batches, val {len(val_ds)}, test {len(test_ds)}")

# Run full classification pipeline
classification_results, summary_rows = train_classification_pipeline(
    results_by_name=results_by_name,
    config_by_name=config_by_name,
    imdb_datasets=(train_ds, val_ds, test_ds),
    clf_config=classification_config,
)

# Save summary
print("\n" + "=" * 88)
print("CLASSIFICATION COMPARISON SUMMARY")
print("=" * 88)
print("name         test_acc  test_auc  precision  recall   f1@0.5   best_thr  best_f1")
print("-" * 88)
for r in summary_rows:
    print(f"{r[0]:12s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:9.4f} {r[4]:8.4f} {r[5]:8.4f} {r[6]:8.2f} {r[7]:8.4f}")
```

```python
## Plot Validation Curves

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, payload in classification_results.items():
    h = payload["history"].history
    axes[0].plot(h.get("val_accuracy", []), marker="o", label=name, linewidth=2)
    axes[1].plot(h.get("val_loss", []), marker="o", label=name, linewidth=2)

axes[0].set_title("Validation Accuracy by Epoch", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Loss by Epoch", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Classification: Validation Performance on IMDB Reviews", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

```python
## Plot Confusion Matrices Side-by-Side

fig, axes = plt.subplots(1, len(classification_results), figsize=(6 * len(classification_results), 5))
if len(classification_results) == 1:
    axes = [axes]

for idx, (name, payload) in enumerate(classification_results.items()):
    eval_stats = payload["eval"]
    tp, tn, fp, fn = eval_stats["tp"], eval_stats["tn"], eval_stats["fp"], eval_stats["fn"]
    
    cm = np.array([[tn, fp], [fn, tp]])
    ax = axes[idx]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_title(f"{name}\n(F1={payload['best_f1']:.4f})", fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_yticklabels(["neg", "pos"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", 
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax)

fig.suptitle("Confusion Matrices by Embedding", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

```python
## Plot Threshold Sweep & Metrics

fig, axes = plt.subplots(2, len(classification_results), figsize=(7 * len(classification_results), 10))
if len(classification_results) == 1:
    axes = axes.reshape(2, 1)

for idx, (name, payload) in enumerate(classification_results.items()):
    threshold_sweep = payload["threshold_sweep"]
    thresholds = [t[0] for t in threshold_sweep]
    
    # Parse metrics
    accuracies = [t[1] for t in threshold_sweep]
    precisions = [t[2] for t in threshold_sweep]
    recalls = [t[3] for t in threshold_sweep]
    f1s = [t[4] for t in threshold_sweep]
    
    # Top plot: all metrics
    ax = axes[0, idx]
    ax.plot(thresholds, accuracies, marker="o", label="Accuracy")
    ax.plot(thresholds, precisions, marker="s", label="Precision")
    ax.plot(thresholds, recalls, marker="^", label="Recall")
    ax.plot(thresholds, f1s, marker="D", label="F1", linewidth=2)
    ax.axvline(x=payload["best_threshold"], color="red", linestyle="--", alpha=0.7, label=f"Best F1 ({payload['best_threshold']:.2f})")
    ax.set_title(f"{name} - Threshold Sweep", fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom plot: F1 only
    ax = axes[1, idx]
    ax.plot(thresholds, f1s, marker="D", linewidth=2, color="green")
    ax.axvline(x=payload["best_threshold"], color="red", linestyle="--", alpha=0.7)
    ax.scatter(x=[payload["best_threshold"]], y=[payload["best_f1"]], color="red", s=100, zorder=5)
    ax.set_title(f"{name} - F1 by Threshold", fontweight="bold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.grid(True, alpha=0.3)

fig.suptitle("Threshold Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

Key improvements:
1. **Config-driven**: `ClassificationConfig` lets you adjust hyperparameters without touching code
2. **Reusable pipeline**: `train_classification_pipeline()` mirrors `train_word2vec_pipeline()` pattern
3. **Separate cells**: Run training once, then eval/plotting independently (saves time debugging visualizations)
4. **Clean separation**: Config → utilities → training → evaluation → plotting
5. **Deterministic**: IMDB datasets loaded once, shared across both embeddings for fair comparison

You can now modify `classification_config` and re-run just the training cell without re-plotting everything!