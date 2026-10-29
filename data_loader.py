"""Data loading utilities for the AG News dataset.

Loads official splits via Hugging Face `datasets` and creates a reproducible
train/validation split from the training data for model selection and experiments.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def load_ag_news_dataset() -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """Loads AG News dataset from Hugging Face datasets.

    Combines 'title' and 'description' (or uses pre-combined 'text') into a single
    input text for each article.

    Returns
    -------
    texts : Dict[str, List[str]]
        Dictionary with keys 'train' and 'test' mapping to list of combined text strings.
    labels : Dict[str, np.ndarray]
        Dictionary with keys 'train' and 'test' mapping to numpy arrays of integer labels.

    Raises
    ------
    RuntimeError
        If dataset downloading fails or huggingface datasets is unavailable.
    """
    import os
    import socket

    # Detect network connectivity before importing datasets to prevent long DNS timeout retries
    try:
        socket.gethostbyname("huggingface.co")
    except (socket.gaierror, OSError):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["DATASETS_OFFLINE"] = "1"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' library is required to load AG News. "
            "Install it via `pip install datasets`."
        ) from exc

    raw_dataset = None
    # Try canonical repo IDs
    for repo_id in ["fancyzhx/ag_news", "ag_news"]:
        try:
            raw_dataset = load_dataset(repo_id)
            break
        except Exception:
            continue

    if raw_dataset is None:
        # Try offline mode from cache
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        for repo_id in ["fancyzhx/ag_news", "ag_news"]:
            try:
                raw_dataset = load_dataset(repo_id)
                break
            except Exception:
                continue

    if raw_dataset is None:
        raise RuntimeError(
            "Failed to load the AG News dataset from Hugging Face or local cache.\n"
            "Please check your internet connection or verify that 'fancyzhx/ag_news' is cached."
        )

    texts: Dict[str, List[str]] = {}
    labels: Dict[str, np.ndarray] = {}

    for split in ["train", "test"]:
        split_data = raw_dataset[split]
        combined_texts: List[str] = []
        split_labels: List[int] = []

        for item in split_data:
            if "text" in item and item["text"]:
                combined = item["text"]
            else:
                title = (item.get("title") or "").strip()
                description = (item.get("description") or "").strip()
                combined = f"{title} {description}".strip()
            combined_texts.append(combined)
            split_labels.append(int(item["label"]))

        texts[split] = combined_texts
        labels[split] = np.array(split_labels, dtype=np.int32)

    return texts, labels


def create_train_val_split(
    texts: List[str],
    labels: np.ndarray,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Reproducibly splits training data into train and validation sets.

    Parameters
    ----------
    texts : List[str]
        List of training text documents.
    labels : np.ndarray
        Array of training target labels.
    val_ratio : float, default=0.1
        Proportion of training data to use for validation.
    random_seed : int, default=42
        Seed for pseudo-random number generator to guarantee reproducibility.

    Returns
    -------
    train_texts : List[str]
    train_labels : np.ndarray
    val_texts : List[str]
    val_labels : np.ndarray
    """
    n_samples = len(texts)
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_size = int(n_samples * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = labels[train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_labels = labels[val_idx]

    return train_texts, train_labels, val_texts, val_labels
