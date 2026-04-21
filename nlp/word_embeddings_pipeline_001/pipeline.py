import numpy as np
import tensorflow as tf
from .model import build_model
from .preprocessing import build_vectorizer

def load_text_dataset(config):
    if config.source_type == "url":
        local_path = tf.keras.utils.get_file(
            fname=f"{config.corpus_name}.txt",
            origin=config.source,
        )
        path = local_path
    elif config.source_type == "local_file":
        path = config.source
    else:
        raise ValueError(f"Unsupported source_type: {config.source_type}")

    text_ds = tf.data.TextLineDataset(path)
    text_ds = text_ds.filter(lambda x: tf.strings.length(x) >= config.min_line_length)

    if config.max_lines is not None:
        text_ds = text_ds.take(config.max_lines)

    return text_ds, path

def prepare_training_dataset(text_ds, config, generate_training_data):
    autotune = tf.data.AUTOTUNE

    vectorize_layer = build_vectorizer(config)
    vectorize_layer.adapt(text_ds.batch(1024))

    vocab = vectorize_layer.get_vocabulary()
    actual_vocab_size = len(vocab)

    text_vector_ds = (
        text_ds
        .batch(1024)
        .prefetch(autotune)
        .map(vectorize_layer)
        .unbatch()
    )

    sequences = list(text_vector_ds.as_numpy_iterator())

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=config.window_size,
        num_ns=config.num_negative_samples,
        vocab_size=actual_vocab_size,
        seed=config.seed,
    )

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(config.buffer_size).batch(config.batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=autotune)

    return {
        "dataset": dataset,
        "vectorize_layer": vectorize_layer,
        "vocab": vocab,
        "actual_vocab_size": actual_vocab_size,
        "targets": targets,
        "contexts": contexts,
        "labels": labels,
        "sequences": sequences,
    }

def train_word2vec_pipeline(config, generate_training_data):
    text_ds, resolved_path = load_text_dataset(config)

    prepared = prepare_training_dataset(
        text_ds=text_ds,
        config=config,
        generate_training_data=generate_training_data,
    )

    model = build_model(
        vocab_size=prepared["actual_vocab_size"],
        embedding_dim=config.embedding_dim,
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f"logs/{config.corpus_name}"
    )

    history = model.fit(
        prepared["dataset"],
        epochs=config.epochs,
        callbacks=[tensorboard_callback],
    )

    weights = model.get_layer("w2v_embedding").get_weights()[0]

    return {
        "model": model,
        "history": history,
        "weights": weights,
        "vocab": prepared["vocab"],
        "vectorize_layer": prepared["vectorize_layer"],
        "dataset": prepared["dataset"],
        "source_path": resolved_path,
        "targets": prepared["targets"],
        "contexts": prepared["contexts"],
        "labels": prepared["labels"],
        "sequences": prepared["sequences"],
    }