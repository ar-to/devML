# Graph Report - devML  (2026-11-08)

## Corpus Check
- Large corpus: 36 files · ~710,619 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 575 nodes · 1010 edges · 38 communities (24 shown, 14 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.85)
- Token cost: 164,665 input · 0 output

## Community Hubs (Navigation)
- Word2Vec Classification Concepts
- Data Cleaning Notebooks
- Embedding Classifier Pipeline v2
- Music Transformer
- Char-Level Text Generation
- Word Segmentation Concepts
- PixelCNN v1
- EDA Utilities
- GAN & Memory Monitoring
- Energy-Based Model
- Word2Vec Classification v1
- Sentiment & Fraud Classification
- Coursera PCA Housing
- Coursera Course 3
- Vanilla Autoencoder
- Reputational Risk NLP
- Horoscope Fine-Tuning
- Word Segmentation Code
- Coursera KNN Distances
- PixelCNN v2
- PixelCNN v3
- Tech Risk NLP
- Manual Forward Pass
- Word2Vec Pipeline Package
- Coursera Auto Regression
- GDL Autoencoder
- GDL MLP
- Linked List Scratch
- Course & Book References
- CIFAR-10 Loader
- CIFAR Image Reshape
- Word Embeddings Versions
- Bach Cello Download
- devml Package

## God Nodes (most connected - your core abstractions)
1. `train_classification_pipeline()` - 10 edges
2. `load_dataset()` - 9 edges
3. `check_dataset_size_and_shape()` - 8 edges
4. `manual_forward_pass_001()` - 8 edges
5. `requirements.txt Python Dependencies` - 8 edges
6. `check_columns_for_nulls()` - 7 edges
7. `Monitoring` - 7 edges
8. `VAE` - 7 edges
9. `RealNVP` - 7 edges
10. `test_train_test_split_ratio()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Coursera Machine Learning Specialization` --semantically_similar_to--> `Deep Learning: How to Run (GPU / Google Colab)`  [INFERRED] [semantically similar]
  coursera/README.md → deep_learning/README.md
- `Fraudulent Transaction SVM Data (pre-split train/test CSV)` --conceptually_related_to--> `scikit-learn`  [INFERRED]
  classification/support_vector_machine/fradulent/README.md → requirements.txt
- `segment() recursive word segmentation` --conceptually_related_to--> `nltk`  [INFERRED]
  nlp/word_segmentation_001/prompt.md → requirements.txt
- `Shannon Entropy` --semantically_similar_to--> `Prediction Confidence Buckets`  [INFERRED] [semantically similar]
  nlp/word_embeddings_002/prompt_entropy.md → nlp/word_embeddings_001/prompt_class_eval.md
- `CorpusConfig` --semantically_similar_to--> `ClassificationConfig`  [INFERRED] [semantically similar]
  nlp/word_embeddings_002/prompt_optimization.md → nlp/word_embeddings_002/prompt_main_refactor.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Config-driven classification train/eval flow** — nlp_word_embeddings_002_prompt_main_refactor_classificationconfig, nlp_word_embeddings_002_prompt_main_refactor_load_imdb_raw_datasets, nlp_word_embeddings_002_prompt_main_refactor_build_classifier_from_embedding, nlp_word_embeddings_002_prompt_main_refactor_evaluate_binary, nlp_word_embeddings_002_prompt_main_refactor_best_threshold_by_f1, nlp_word_embeddings_002_prompt_main_refactor_train_classification_pipeline [EXTRACTED 1.00]
- **Reproducible inline vs refactor seed comparison** — nlp_word_embeddings_002_prompt_reset_run_reset_everything, nlp_word_embeddings_002_prompt_reset_run_train_inline_loop, nlp_word_embeddings_002_prompt_reset_run_summarize_runs, nlp_word_embeddings_002_prompt_reset_run_multi_seed_ab_comparison, nlp_word_embeddings_002_prompt_reset_run_stochastic_variance [EXTRACTED 1.00]
- **Word2Vec preprocessing speed tuning** — nlp_word_embeddings_002_prompt_optimization_skipgram_preprocessing_bottleneck, nlp_word_embeddings_002_prompt_optimization_max_lines, nlp_word_embeddings_002_prompt_optimization_speed_quality_presets, nlp_word_embeddings_002_prompt_optimization_corpusconfig, nlp_word_embeddings_002_prompt_optimization_load_text_dataset [EXTRACTED 1.00]
- **Unigram-scored memoized word segmentation** — nlp_word_segmentation_001_prompt_cached_segment, nlp_word_segmentation_001_prompt_splitpairs, nlp_word_segmentation_001_prompt_wordseqfitness, nlp_word_segmentation_001_prompt_onegramdist, nlp_word_segmentation_001_prompt_unknown_word_penalty [EXTRACTED 1.00]
- **devML environment setup workflow** — readme_virtual_environment_setup, requirements_dependencies, readme_pip_review_upgrade, readme_utils_internal_module [EXTRACTED 1.00]

## Communities (38 total, 14 thin omitted)

### Community 0 - "Word2Vec Classification Concepts"
Cohesion: 0.05
Nodes (49): Prompt: Applying Word2Vec to Classification and Evaluation, Prediction Confidence Buckets, Confusion Matrix, Precision, Recall, F1, Shakespeare-to-IMDB Domain Mismatch, Embedding Fine-Tuning (unfreeze, Adam 1e-4), Frozen Pretrained Embedding Classifier (GlobalAveragePooling + Dense), Padding Token Leakage in Skip-gram Training, Pretrained Embedding Transfer to Sentiment Classifier (+41 more)

### Community 1 - "Data Cleaning Notebooks"
Cohesion: 0.06
Nodes (22): biplot(), visualize_history(), apply_iqr_outlier_removal(), check_columns_for_inf(), check_columns_for_na(), check_columns_for_nulls(), check_dataset_size_and_shape(), dummy_error_to_stop_runs() (+14 more)

### Community 2 - "Embedding Classifier Pipeline v2"
Cohesion: 0.07
Nodes (20): analyze_sample_predictions(), best_threshold_by_f1(), build_classifier_from_embedding(), build_classifier_inline(), build_model(), build_standardizer(), build_vectorizer(), ClassificationConfig (+12 more)

### Community 3 - "Music Transformer"
Cohesion: 0.08
Nodes (8): causal_attention_mask(), get_midi_note(), load_parsed_files(), MusicGenerator, parse_midi_files(), SinePositionEncoding, TokenAndPositionEmbedding, TransformerBlock

### Community 4 - "Char-Level Text Generation"
Cohesion: 0.08
Nodes (6): CharGenModel, clean_logs(), download_and_read(), generate_text(), display(), VAE

### Community 5 - "Word Segmentation Concepts"
Cohesion: 0.08
Nodes (21): Fraudulent Transaction SVM Data (pre-split train/test CSV), Keras Imbalanced Classification Example (credit card fraud), one-grams.txt unigram counts, OneGramDist (unigram frequency distribution), segment() recursive word segmentation, nlp/word_segmentation.ipynb, wordSeqFitness (sum of log10 probabilities), config.yaml model parameters (+13 more)

### Community 6 - "PixelCNN v1"
Cohesion: 0.09
Nodes (7): display(), ImageGenerator, negative_log_likelihood(), preprocess(), Coupling(), ImageGenerator, RealNVP

### Community 7 - "EDA Utilities"
Cohesion: 0.11
Nodes (11): check_columns_for_inf(), check_columns_for_na(), check_dataset_size_and_shape(), dummy_error_to_stop_runs(), show_columns_with_missing_values(), show_count_of_columns_missing_values(), test_no_inf_columns(), test_no_missing_values() (+3 more)

### Community 8 - "GAN & Memory Monitoring"
Cohesion: 0.10
Nodes (6): check_memory(), plotLoss(), saveGeneratedImages(), train(), check_file_size(), Monitoring

### Community 9 - "Energy-Based Model"
Cohesion: 0.13
Nodes (7): Buffer, display(), EBM, generate_samples(), ImageGenerator, preprocess(), SaveModel

### Community 10 - "Word2Vec Classification v1"
Cohesion: 0.14
Nodes (7): clean_text(), custom_standardization(), generate_training_data(), make_labeled_review_ds(), Word2Vec, build_standardizer(), generate_training_data()

### Community 11 - "Sentiment & Fraud Classification"
Cohesion: 0.14
Nodes (3): fit_and_evaluate(), load_dataset(), check_columns_for_nulls()

### Community 12 - "Coursera PCA Housing"
Cohesion: 0.13
Nodes (5): load_housing(), debug_print(), pca_mse(), load_data(), load_data()

### Community 15 - "Vanilla Autoencoder"
Cohesion: 0.21
Nodes (6): Autoencoder, Decoder, Encoder, loss(), train(), train_loop()

### Community 16 - "Reputational Risk NLP"
Cohesion: 0.15
Nodes (3): encode_categorical_columns(), evaluate_model(), normalize_categorical_value()

### Community 18 - "Word Segmentation Code"
Cohesion: 0.24
Nodes (6): OneGramDist, segment(), _segment(), segment_v2(), splitPairs(), wordSeqFitness()

### Community 19 - "Coursera KNN Distances"
Cohesion: 0.21
Nodes (4): knn_rmse(), euclidean_distance(), manhattan_distance(), evaluate_model()

### Community 20 - "PixelCNN v2"
Cohesion: 0.22
Nodes (4): display(), ImageGenerator, negative_log_likelihood(), preprocess()

### Community 21 - "PixelCNN v3"
Cohesion: 0.22
Nodes (4): display(), ImageGenerator, negative_log_likelihood(), preprocess()

### Community 22 - "Tech Risk NLP"
Cohesion: 0.20
Nodes (3): clean_text(), evaluate_model(), extract_topics()

### Community 23 - "Manual Forward Pass"
Cohesion: 0.24
Nodes (5): manual_forward_pass(), manual_forward_pass_001(), relu(), sigmoid(), predict_input()

### Community 24 - "Word2Vec Pipeline Package"
Cohesion: 0.33
Nodes (6): build_model(), Word2Vec, load_text_dataset(), prepare_training_dataset(), train_word2vec_pipeline(), build_vectorizer()

### Community 29 - "Course & Book References"
Cohesion: 0.50
Nodes (3): Coursera Machine Learning Specialization, Generative Deep Learning 2nd Edition (davidADSP), Google Colab Runtime

### Community 32 - "Word Embeddings Versions"
Cohesion: 1.00
Nodes (3): Word Embeddings Pipeline 001 (multi-file production structure), nlp/word_embeddings_001/word_embeddings_classification.ipynb, nlp/word_embeddings_002 (single notebook)

## Knowledge Gaps
- **25 isolated node(s):** `devml`, `Combined Multi-Embedding Classification Notebook`, `word_embeddings_pipeline_001`, `Prompt: Measuring Entropy`, `Prediction Uncertainty Distribution Plot` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 242 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `devml`, `Combined Multi-Embedding Classification Notebook`, `word_embeddings_pipeline_001` to the rest of the system?**
  _25 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Word2Vec Classification Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.05203619909502263 - nodes in this community are weakly interconnected._
- **Should `Data Cleaning Notebooks` be split into smaller, more focused modules?**
  _Cohesion score 0.05939716312056738 - nodes in this community are weakly interconnected._
- **Should `Embedding Classifier Pipeline v2` be split into smaller, more focused modules?**
  _Cohesion score 0.06829268292682927 - nodes in this community are weakly interconnected._
- **Should `Music Transformer` be split into smaller, more focused modules?**
  _Cohesion score 0.07765151515151515 - nodes in this community are weakly interconnected._
- **Should `Char-Level Text Generation` be split into smaller, more focused modules?**
  _Cohesion score 0.08465608465608465 - nodes in this community are weakly interconnected._
- **Should `Word Segmentation Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.07977207977207977 - nodes in this community are weakly interconnected._