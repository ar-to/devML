from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

@dataclass
class CorpusConfig:
    corpus_name: str
    source_type: Literal["local_file", "url"]
    source: str

    vocab_size: int = 4096
    sequence_length: int = 10
    window_size: int = 2
    num_negative_samples: int = 4
    embedding_dim: int = 128
    batch_size: int = 1024
    buffer_size: int = 10000
    epochs: int = 20
    seed: int = 42

    lowercase: bool = True
    strip_punctuation: bool = False
    keep_apostrophes: bool = True
    min_line_length: int = 1
    max_lines: Optional[int] = None #cuts the expensive outer loop before skip-gram generation starts.