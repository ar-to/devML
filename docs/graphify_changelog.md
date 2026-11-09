# graphify Changelog

A record of each [graphify](https://docs.graphify.com/open-source/quickstart) knowledge-graph build for this repo. Add the newest run at the top.

---

## 2026-11-08: First full build

**Command:** `/graphify` on the repo root (`.`), default mode, undirected graph.

**Result:** 575 nodes · 1,010 edges · 38 communities

### Corpus

| Type | Files | Notes |
|---|---|---|
| Code | 54 | `.py`, `.sh`, `.toml`, `.json`. Includes 39 notebooks (see below). Read structurally (AST), so no LLM tokens. |
| Docs | 16 | `.md` / `.txt`. Read by 2 semantic-extraction subagents, every file to its last line. |

**What was changed or left out:**
- **Notebooks (`.ipynb`):** this graphify version can't read them. Before extraction, their code and markdown cells were copied into `.py` files under `graphify-out/notebooks/`, named after the original path with `/` replaced by `__`. `model_template/data_analysis_visualizations.ipynb` and `model_template/pre_processing.ipynb` were empty and skipped.
- **Left out on purpose:**
  - the 4 `.mp3` files in `deep_learning/Generative_Deep_Learning_2nd_Edition/outputs/music_gen/` (generated music with no speech to transcribe)
  - `nlp/word_segmentation_001/one-grams.txt` (a 333k-line word-frequency list: data, not prose)

**Token cost:** about 165k subagent tokens for doc extraction. Only a combined total is available, not an input/output split.

### Health check

The integrity check flagged these:
- 509 edges pointing at nodes that aren't in the graph
- 2 edges that connect a node to itself
- 20 duplicate edges between the same two nodes, merged into one

The cause of the 509 is likely imports of outside libraries (pandas, TensorFlow and so on), which have no node of their own in the graph. This wasn't checked. The graph is still usable.

### God nodes (most connected)

1. `train_classification_pipeline()`: 10 edges
2. `load_dataset()`: 9 edges
3. `check_dataset_size_and_shape()`: 8 edges
4. `manual_forward_pass_001()`: 8 edges
5. `requirements.txt Python Dependencies`: 8 edges
6. `check_columns_for_nulls()`: 7 edges
7. `Monitoring`: 7 edges
8. `VAE`: 7 edges
9. `RealNVP`: 7 edges
10. `test_train_test_split_ratio()`: 6 edges

### Surprising connections

All five are INFERRED links between files that never reference each other directly.

- `Coursera Machine Learning Specialization` → `Deep Learning: How to Run (GPU / Google Colab)`: semantically similar (`coursera/README.md` → `deep_learning/README.md`). Both READMEs explain why the code can't run locally as-is.
- `Fraudulent Transaction SVM Data` → `scikit-learn`: conceptually related (`classification/support_vector_machine/fradulent/README.md` → `requirements.txt`).
- `segment() recursive word segmentation` → `nltk`: conceptually related, low confidence 0.55 (`nlp/word_segmentation_001/prompt.md` → `requirements.txt`).
- `Shannon Entropy` → `Prediction Confidence Buckets`: semantically similar (`nlp/word_embeddings_002/prompt_entropy.md` → `nlp/word_embeddings_001/prompt_class_eval.md`).
- `CorpusConfig` → `ClassificationConfig`: semantically similar (`nlp/word_embeddings_002/prompt_optimization.md` → `prompt_main_refactor.md`).

### Group relationships (hyperedges)

These are sets of three or more nodes that work together as one flow.

- **Config-driven classification train/eval flow:** `ClassificationConfig`, `load_imdb_raw_datasets`, `build_classifier_from_embedding`, `evaluate_binary`, `best_threshold_by_f1`, `train_classification_pipeline`
- **Reproducible inline vs refactor seed comparison:** `reset_everything`, `train_inline_loop`, `summarize_runs`, multi-seed A/B comparison, stochastic variance
- **Word2Vec preprocessing speed tuning:** skip-gram preprocessing bottleneck, `max_lines`, speed/quality presets, `CorpusConfig`, `load_text_dataset`
- **Unigram-scored memoized word segmentation:** cached `_segment`, `splitPairs`, `wordSeqFitness`, `OneGramDist`, unknown-word penalty
- **devML environment setup workflow:** venv setup, `requirements.txt`, pip-review upgrade, `utils` internal module

No import cycles were detected.

### Communities (38)

Each node is assigned to one community, a group of closely linked nodes. Labels were chosen by hand from each group's nodes. Cohesion runs from 0 to 1: the higher it is, the more the nodes in a group link to each other.

| ID | Label | Nodes | Cohesion |
|---|---|---|---|
| 0 | Word2Vec Classification Concepts | 52 | 0.05 |
| 1 | Data Cleaning Notebooks | 48 | 0.06 |
| 2 | Embedding Classifier Pipeline v2 | 41 | 0.07 |
| 3 | Music Transformer | 33 | 0.08 |
| 4 | Char-Level Text Generation | 28 | 0.08 |
| 5 | Word Segmentation Concepts | 27 | 0.08 |
| 6 | PixelCNN v1 | 27 | 0.09 |
| 7 | EDA Utilities | 25 | 0.11 |
| 8 | GAN & Memory Monitoring | 24 | 0.10 |
| 9 | Energy-Based Model | 21 | 0.13 |
| 10 | Word2Vec Classification v1 | 19 | 0.14 |
| 11 | Sentiment & Fraud Classification | 18 | 0.14 |
| 12 | Coursera PCA Housing | 17 | 0.13 |
| 13 | Coursera Course 3 | 16 | 0.22 |
| 14 | First Model Experiments | 14 | 0.19 |
| 15 | Vanilla Autoencoder | 14 | 0.21 |
| 16 | Reputational Risk NLP | 14 | 0.15 |
| 17 | Horoscope Fine-Tuning | 13 | 0.17 |
| 18 | Word Segmentation Code | 13 | 0.24 |
| 19 | Coursera KNN Distances | 13 | 0.21 |
| 20 | PixelCNN v2 | 11 | 0.22 |
| 21 | PixelCNN v3 | 11 | 0.22 |
| 22 | Tech Risk NLP | 11 | 0.20 |
| 23 | Manual Forward Pass | 10 | 0.24 |
| 24 | Word2Vec Pipeline Package | 10 | 0.33 |
| 25 | Coursera Auto Regression | 7 | 0.29 |
| 26 | GDL Autoencoder | 7 | 0.29 |
| 27 | GDL MLP | 6 | 0.33 |
| 28 | Linked List Scratch | 5 | 0.50 |
| 29 | Course & Book References | 4 | 0.50 |
| 30 | CIFAR-10 Loader | 4 | 0.50 |
| 31 | CIFAR Image Reshape | 3 | 0.67 |
| 32 | Word Embeddings Versions | 3 | 1.00 |
| 33 | Bach Cello Download | 2 | 1.00 |
| 34 | GAN Experiment | 1 | 1.00 |
| 35 | Pipeline Package Init | 1 | 1.00 |
| 36 | devml Package | 1 | 1.00 |
| 37 | Utils Package Init | 1 | 1.00 |

### Suggested questions

- **What connects `devml`, `Combined Multi-Embedding Classification Notebook` and `word_embeddings_pipeline_001` to the rest of the system?** 25 nodes are weakly connected, which may mean missing docs or missing links.
- **Should these communities be split into smaller ones?** Their nodes are only loosely linked to each other:
  - `Word2Vec Classification Concepts`: 0.052
  - `Data Cleaning Notebooks`: 0.059
  - `Embedding Classifier Pipeline v2`: 0.068
  - `Music Transformer`: 0.078
  - `Word Segmentation Concepts`: 0.080
  - `Char-Level Text Generation`: 0.085

The first question cuts across the most communities. The concepts from the `word_embeddings_002` refactor prompts (communities 0 and 5) and the notebook code they describe (communities 2 and 10) are in separate groups, with few edges between them.

### Outputs

Everything below is in `graphify-out/`.

| File | Tracked in git | Purpose |
|---|---|---|
| `graph.json` | yes | Raw graph that queries and subagents read |
| `GRAPH_REPORT.md` | yes | The full report |
| `graph.html` | no | Interactive graph; open in a browser |
| `notebooks/`, `cache/`, `manifest.json`, `cost.json`, `.graphify_*` | no | Local working files and build state; rebuilt as needed |

### Sample command

Used inside Claude.

```
/graphify query "find me structure to use for logistic regression to start a new ipynb. Need a list of top sections to act as headers and use as a rule of thumb. Also provide existing examples to use as references for details and further work."  
```