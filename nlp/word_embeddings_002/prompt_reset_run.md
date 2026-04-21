User: can you tell me main differences in hyperparameters between using `for name, w2v_results in results_by_name.items():` and `#sym:train_classification_pipeline `? I'm seeing a slight difference in the accuracy between them.

GitHub Copilot: You’re comparing the original inline training loop versus the refactored `train_classification_pipeline`, so I’ll quickly verify the latest notebook cells and then map hyperparameters side-by-side to pinpoint what can change accuracy.

Retrieving Notebook summary.

I’ve confirmed the notebook state; next I’m checking the two training blocks directly to extract a precise hyperparameter diff list rather than guesswork.

Read [](file:///Users/aricode0/devML/nlp/word_embeddings_002/word_embeddings_classification.ipynb#930-930), lines 930 to 1360

Main hyperparameter difference is:

1. **Epochs**
- Inline loop (for name, w2v_results in results_by_name.items()): uses `epochs=8`
- train_classification_pipeline path: you instantiate `ClassificationConfig(epochs=10)`, so it trains up to 10 epochs

Everything else appears matched by default config values:

1. sequence length: 200
2. embedding trainable: False
3. dense units: 64
4. dropout: 0.3
5. batch size: 64
6. early stopping patience: 2
7. threshold for eval: 0.5
8. validation split size: 5000
9. seed value: 42

So if accuracy changed, the biggest direct cause is almost certainly **8 vs 10 max epochs** (plus usual stochastic variance).

Also worth noting: even with same hyperparameters, you can still get slight differences if you ran one experiment after another without resetting seeds immediately before training each model (weight initialization and training order randomness can drift).

User: what is stochastic variance and what do you mean by resetting seeds?

GitHub Copilot: Stochastic variance is the natural run-to-run noise you get in training, even when code and hyperparameters are “the same.”

In your notebook, variance can come from:
1. Random weight initialization in the classifier.
2. Random minibatch/shuffle order in the dataset.
3. Stochastic layers/ops during training.
4. Nondeterministic GPU kernels (some TF ops can differ slightly run to run).

What “resetting seeds” means:
1. Set the same random seed right before each experiment.
2. Rebuild the model and data pipeline after setting the seed.
3. Optionally force deterministic TensorFlow ops.

Why this matters for you:
You set seeds once near the top (SEED = 42), but if you run cells multiple times, random state keeps advancing. So a later run can start from a different RNG state and end with slightly different accuracy.

A solid reproducible pattern before each comparison run:
1. Restart kernel (best clean reset), or at least clear session.
2. Set seed again.
3. Recreate datasets.
4. Recreate model.
5. Train.

If you want, I can give you a tiny “run_once(seed)” cell you can paste so both approaches are compared under the exact same random state.

User: sure

GitHub Copilot: Perfect. Paste this as a new cell to compare both paths under the same seed and same data split.

```python
# Reproducible A/B compare: inline loop vs train_classification_pipeline
import copy
import random
import numpy as np
import tensorflow as tf

def reset_everything(seed=42, deterministic_ops=True):
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)  # Python, NumPy, TF together
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

def train_inline_loop(results_by_name, config_by_name, imdb_datasets, epochs=8, threshold=0.5):
    train_ds, val_ds, test_ds = imdb_datasets
    classification_results = {}
    summary_rows = []

    for name, w2v_results in results_by_name.items():
        model = build_classifier_from_embedding(
            results=w2v_results,
            cfg=config_by_name[name],
            config=ClassificationConfig(
                sequence_length=200,
                trainable_embedding=False,
                dense_units=64,
                dropout_rate=0.3,
                batch_size=64,
                epochs=epochs,
                early_stopping_patience=2,
                threshold=threshold,
                val_split=5000,
                seed=42,
            ),
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
        eval_stats = evaluate_binary(model, test_ds, threshold=threshold)
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

# --------------------------
# 1) Build ONE shared dataset split (fixed seed)
# --------------------------
compare_epochs = 8  # set 8 to match old loop exactly, or 10 if you want both at 10
shared_cfg = ClassificationConfig(epochs=compare_epochs, seed=42)
reset_everything(seed=shared_cfg.seed)
train_ds, val_ds, test_ds = load_imdb_raw_datasets(shared_cfg)  # current refactor signature
shared_datasets = (train_ds, val_ds, test_ds)

# --------------------------
# 2) Run inline-style path
# --------------------------
reset_everything(seed=shared_cfg.seed)
inline_results, inline_summary = train_inline_loop(
    results_by_name=results_by_name,
    config_by_name=config_by_name,
    imdb_datasets=shared_datasets,
    epochs=compare_epochs,
    threshold=0.5,
)

# --------------------------
# 3) Run refactor pipeline path
# --------------------------
reset_everything(seed=shared_cfg.seed)
refactor_cfg = copy.deepcopy(shared_cfg)
refactor_results, refactor_summary = train_classification_pipeline(
    results_by_name=results_by_name,
    config_by_name=config_by_name,
    imdb_datasets=shared_datasets,
    clf_config=refactor_cfg,
)

print("INLINE SUMMARY")
for r in inline_summary:
    print(r)

print("\nREFACTOR SUMMARY")
for r in refactor_summary:
    print(r)
```

If you still see small deltas after this, it is usually residual nondeterminism from backend ops. If you want, I can give you a second cell that runs this comparison 5 times and reports mean/std accuracy so you can quantify the variance directly.

User: sure

GitHub Copilot: Paste this as one new cell. It runs 5 seeds, compares inline-vs-refactor, and reports mean/std accuracy plus average delta.

    import random
    import numpy as np
    import tensorflow as tf

    def reset_everything(seed=42, deterministic_ops=True):
        tf.keras.backend.clear_session()
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass

    # Inline-style builder matching your earlier non-config version
    def build_classifier_inline(results, cfg, sequence_length=200, trainable=False):
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

    def train_inline_loop(results_by_name, config_by_name, imdb_datasets, epochs=8, threshold=0.5):
        train_ds, val_ds, test_ds = imdb_datasets
        out = {}

        for name, w2v_results in results_by_name.items():
            model = build_classifier_inline(
                results=w2v_results,
                cfg=config_by_name[name],
                sequence_length=200,
                trainable=False,
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                )
            ]

            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )

            test_loss, test_acc, test_auc = model.evaluate(test_ds, verbose=0)
            out[name] = {
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "test_auc": float(test_auc),
            }

        return out

    def summarize_runs(run_records):
        # run_records: dict[name][path] -> list of accuracies
        print("\nMean ± std over seeds (accuracy)")
        print("-" * 72)
        print(f'{"Corpus":14s} {"Inline":20s} {"Refactor":20s} {"Delta(ref-inline)":16s}')
        print("-" * 72)

        for name in run_records:
            inline_vals = np.array(run_records[name]["inline"], dtype=float)
            ref_vals = np.array(run_records[name]["refactor"], dtype=float)
            delta = ref_vals - inline_vals

            inline_str = f"{inline_vals.mean():.4f} ± {inline_vals.std(ddof=1):.4f}"
            ref_str = f"{ref_vals.mean():.4f} ± {ref_vals.std(ddof=1):.4f}"
            delta_str = f"{delta.mean():+.4f} ± {delta.std(ddof=1):.4f}"
            print(f"{name:14s} {inline_str:20s} {ref_str:20s} {delta_str:16s}")

    # ---- experiment settings ----
    seeds = [42, 43, 44, 45, 46]
    compare_epochs = 8   # set 8 for apples-to-apples with old loop
    deterministic_ops = True

    run_records = {
        name: {"inline": [], "refactor": []}
        for name in results_by_name.keys()
    }

    for seed in seeds:
        print(f"\nRunning seed={seed}")

        # Build one shared split for both methods at this seed
        reset_everything(seed=seed, deterministic_ops=deterministic_ops)
        cfg = ClassificationConfig(
            sequence_length=200,
            trainable_embedding=False,
            dense_units=64,
            dropout_rate=0.3,
            batch_size=64,
            epochs=compare_epochs,
            early_stopping_patience=2,
            threshold=0.5,
            val_split=5000,
            seed=seed,
        )
        train_ds, val_ds, test_ds = load_imdb_raw_datasets(cfg)
        shared_datasets = (train_ds, val_ds, test_ds)

        # Inline path
        reset_everything(seed=seed, deterministic_ops=deterministic_ops)
        inline_out = train_inline_loop(
            results_by_name=results_by_name,
            config_by_name=config_by_name,
            imdb_datasets=shared_datasets,
            epochs=compare_epochs,
            threshold=0.5,
        )

        # Refactor path
        reset_everything(seed=seed, deterministic_ops=deterministic_ops)
        ref_out, _ = train_classification_pipeline(
            results_by_name=results_by_name,
            config_by_name=config_by_name,
            imdb_datasets=shared_datasets,
            clf_config=cfg,
        )

        # Collect test accuracy
        for name in results_by_name.keys():
            run_records[name]["inline"].append(inline_out[name]["test_acc"])
            run_records[name]["refactor"].append(ref_out[name]["test_acc"])

    summarize_runs(run_records)

If you want, I can also give a tiny follow-up cell that does a paired t-test on the per-seed deltas to tell you whether the observed difference is likely just noise.