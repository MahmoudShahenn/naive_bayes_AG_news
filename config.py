"""Configuration settings for AG News text classification with Naive Bayes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# AG News dataset class mapping (0-indexed as per Hugging Face ag_news)
LABEL_MAPPING: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

CLASS_NAMES = [LABEL_MAPPING[i] for i in range(len(LABEL_MAPPING))]


@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 1


@dataclass
class VectorizerConfig:
    min_df: int = 2
    max_features: int = 30000
    ngram_range: Tuple[int, int] = (1, 1)


@dataclass
class ModelConfig:
    alpha: float = 1.0


@dataclass
class ExperimentConfig:
    random_seed: int = 42
    val_split_ratio: float = 0.1  # 10% of training set for hyperparameter validation
    alphas_to_test: Tuple[float, ...] = (0.1, 1.0, 2.0)
    vocab_limits_to_test: Tuple[int, ...] = (10000, 30000, 50000)
    ngram_ranges_to_test: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 2))


@dataclass
class ProjectConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

