"""Evaluation metrics, qualitative analysis, and visualization utilities.

Computes classification metrics (accuracy, precision, recall, F1, confusion matrix),
generates publication-quality plots, and outputs detailed diagnostic reports.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Set safe writable matplotlib config dir
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, LABEL_MAPPING, PLOTS_DIR, REPORTS_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Union[float, Dict[str, Dict[str, float]], np.ndarray]]:
    """Calculates comprehensive classification metrics from scratch with sklearn verification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer labels.
    y_pred : np.ndarray
        Model predicted integer labels.
    class_names : List[str]
        Human-readable class names.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, per-class metrics, macro-f1, weighted-f1,
        and confusion matrix.
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Manual per-class metrics
    per_class: Dict[str, Dict[str, float]] = {}
    supports = []
    f1s = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        supports.append(support)
        f1s.append(f1)

    total_samples = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(sum(f1 * s for f1, s in zip(f1s, supports)) / total_samples)

    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "report_str": report_str,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[Path] = None,
    title: str = "AG News Test Set Confusion Matrix",
) -> None:
    """Plots and saves a styled confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix array.
    class_names : List[str]
        List of class label strings.
    save_path : Optional[Path]
        Destination file path.
    title : str
        Figure title.
    """
    if save_path is None:
        save_path = PLOTS_DIR / "confusion_matrix.png"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0.8,
    )
    plt.title(title, fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_class_metrics(
    per_class: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    title: str = "Precision, Recall, and F1-Score by Class",
) -> None:
    """Plots a grouped bar chart comparing precision, recall, and F1 per class.

    Parameters
    ----------
    per_class : Dict[str, Dict[str, float]]
        Per-class metrics mapping.
    save_path : Optional[Path]
        Destination file path.
    title : str
        Figure title.
    """
    if save_path is None:
        save_path = PLOTS_DIR / "class_metrics.png"

    classes = list(per_class.keys())
    precisions = [per_class[c]["precision"] * 100 for c in classes]
    recalls = [per_class[c]["recall"] * 100 for c in classes]
    f1s = [per_class[c]["f1-score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3470a3")
    rects2 = ax.bar(x, recalls, width, label="Recall", color="#e27c38")
    rects3 = ax.bar(x + width, f1s, width, label="F1-Score", color="#4ba16a")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Label heights
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_experiment_comparisons(
    alpha_results: List[Dict],
    ngram_results: List[Dict],
    vocab_results: List[Dict],
    save_path: Optional[Path] = None,
) -> None:
    """Plots side-by-side experiment comparisons for alpha, n-grams, and vocab limits.

    Parameters
    ----------
    alpha_results : List[Dict]
    ngram_results : List[Dict]
    vocab_results : List[Dict]
    save_path : Optional[Path]
    """
    if save_path is None:
        save_path = PLOTS_DIR / "experiments_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Laplace Smoothing Alpha
    alphas = [str(r["alpha"]) for r in alpha_results]
    alpha_acc = [r["val_accuracy"] * 100 for r in alpha_results]
    axes[0].plot(alphas, alpha_acc, marker="o", color="#2b5c8f", linewidth=2, markersize=8)
    axes[0].set_title("Effect of Laplace Smoothing (alpha)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("alpha", fontsize=10)
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    for i, txt in enumerate(alpha_acc):
        axes[0].annotate(f"{txt:.2f}%", (alphas[i], alpha_acc[i] + 0.1), ha="center")

    # 2. N-gram Features
    ngrams = [r["name"] for r in ngram_results]
    ngram_acc = [r["val_accuracy"] * 100 for r in ngram_results]
    axes[1].bar(ngrams, ngram_acc, color=["#3f88c5", "#d00000"], width=0.45)
    axes[1].set_title("Unigram vs Unigram+Bigram", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Feature Representation", fontsize=10)
    axes[1].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[1].set_ylim(min(ngram_acc) - 2, max(ngram_acc) + 2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for i, txt in enumerate(ngram_acc):
        axes[1].annotate(f"{txt:.2f}%", (i, txt + 0.2), ha="center")

    # 3. Vocabulary Size Limit
    vocabs = [f"{r['max_features'] // 1000}k" if r['max_features'] else "Full" for r in vocab_results]
    vocab_acc = [r["val_accuracy"] * 100 for r in vocab_results]
    axes[2].plot(vocabs, vocab_acc, marker="s", color="#38b000", linewidth=2, markersize=8)
    axes[2].set_title("Effect of Vocabulary Size Limit", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Max Features", fontsize=10)
    axes[2].set_ylabel("Validation Accuracy (%)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    for i, txt in enumerate(vocab_acc):
        axes[2].annotate(f"{txt:.2f}%", (vocabs[i], vocab_acc[i] + 0.1), ha="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_qualitative_examples(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_examples: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """Extracts samples of correctly and incorrectly classified articles with details.

    Parameters
    ----------
    texts : List[str]
    y_true : np.ndarray
    y_pred : np.ndarray
    y_proba : np.ndarray
    n_examples : int, default=5

    Returns
    -------
    correct_samples : List[Dict]
    misclassified_samples : List[Dict]
    """
    correct_idx = np.where(y_true == y_pred)[0]
    error_idx = np.where(y_true != y_pred)[0]

    correct_samples = []
    for idx in correct_idx[:n_examples]:
        correct_samples.append({
            "index": int(idx),
            "text": texts[idx],
            "true_class": LABEL_MAPPING[int(y_true[idx])],
            "pred_class": LABEL_MAPPING[int(y_pred[idx])],
            "confidence": float(y_proba[idx, y_pred[idx]]),
            "probabilities": {LABEL_MAPPING[k]: float(y_proba[idx, k]) for k in range(4)},
        })

    misclassified_samples = []
    for idx in error_idx[:n_examples]:
        misclassified_samples.append({
            "index": int(idx),
            "text": texts[idx],
            "true_class": LABEL_MAPPING[int(y_true[idx])],
            "pred_class": LABEL_MAPPING[int(y_pred[idx])],
            "pred_confidence": float(y_proba[idx, y_pred[idx]]),
            "true_confidence": float(y_proba[idx, y_true[idx]]),
            "probabilities": {LABEL_MAPPING[k]: float(y_proba[idx, k]) for k in range(4)},
        })

    return correct_samples, misclassified_samples


def generate_evaluation_report(
    metrics: Dict,
    train_size: int,
    test_size: int,
    vocab_size: int,
    train_time_sec: float,
    pred_time_sec: float,
    correct_samples: List[Dict],
    misclassified_samples: List[Dict],
    save_path: Optional[Path] = None,
) -> str:
    """Generates and writes a comprehensive text and markdown evaluation report.

    Parameters
    ----------
    metrics : Dict
    train_size : int
    test_size : int
    vocab_size : int
    train_time_sec : float
    pred_time_sec : float
    correct_samples : List[Dict]
    misclassified_samples : List[Dict]
    save_path : Optional[Path]

    Returns
    -------
    report_text : str
    """
    if save_path is None:
        save_path = REPORTS_DIR / "final_evaluation_report.md"

    lines = []
    lines.append("# AG News Classification - Multinomial Naive Bayes Evaluation Report\n")
    lines.append("## 1. Dataset & Model Summary")
    lines.append(f"- **Training Dataset Size**: {train_size:,} articles")
    lines.append(f"- **Test Dataset Size**: {test_size:,} articles")
    lines.append(f"- **Vocabulary Size**: {vocab_size:,} tokens")
    lines.append(f"- **Training Time**: {train_time_sec:.3f} seconds")
    lines.append(f"- **Prediction Time (Test Split)**: {pred_time_sec:.3f} seconds")
    lines.append(f"- **Throughput**: {test_size / pred_time_sec:.1f} articles/second\n")

    lines.append("## 2. Overall Performance Metrics")
    lines.append(f"- **Test Accuracy**: {metrics['accuracy'] * 100:.2f}%")
    lines.append(f"- **Macro-Averaged F1**: {metrics['macro_f1'] * 100:.2f}%")
    lines.append(f"- **Weighted-Averaged F1**: {metrics['weighted_f1'] * 100:.2f}%\n")

    lines.append("## 3. Per-Class Performance Breakdown")
    lines.append("| Class | Precision (%) | Recall (%) | F1-Score (%) | Support |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for cls_name, pcm in metrics["per_class"].items():
        lines.append(
            f"| {cls_name} | {pcm['precision'] * 100:.2f} | {pcm['recall'] * 100:.2f} | "
            f"{pcm['f1-score'] * 100:.2f} | {pcm['support']:,} |"
        )
    lines.append("")

    lines.append("## 4. Confusion Matrix")
    cm = metrics["confusion_matrix"]
    lines.append("| True \\ Pred | " + " | ".join(CLASS_NAMES) + " |")
    lines.append("| :--- | " + " | ".join([":---:"] * len(CLASS_NAMES)) + " |")
    for i, name in enumerate(CLASS_NAMES):
        row_str = " | ".join(f"{cm[i, j]:,}" for j in range(len(CLASS_NAMES)))
        lines.append(f"| **{name}** | {row_str} |")
    lines.append("")

    lines.append("## 5. Classification Report (Scikit-Learn Standard Format)")
    lines.append("```text")
    lines.append(metrics["report_str"].strip())
    lines.append("```\n")

    lines.append("## 6. Qualitative Analysis: Correctly Classified Articles")
    for i, ex in enumerate(correct_samples, 1):
        lines.append(f"### Example {i} (Index: {ex['index']})")
        lines.append(f"- **Article Text**: \"{ex['text']}\"")
        lines.append(f"- **Ground Truth**: `{ex['true_class']}`")
        lines.append(f"- **Prediction**: `{ex['pred_class']}` (Confidence: {ex['confidence'] * 100:.2f}%)")
        lines.append(f"- **Class Probabilities**: {ex['probabilities']}\n")

    lines.append("## 7. Qualitative Analysis: Misclassified Articles")
    for i, ex in enumerate(misclassified_samples, 1):
        lines.append(f"### Error {i} (Index: {ex['index']})")
        lines.append(f"- **Article Text**: \"{ex['text']}\"")
        lines.append(f"- **Ground Truth**: `{ex['true_class']}` (Confidence: {ex['true_confidence'] * 100:.2f}%)")
        lines.append(f"- **Predicted Class**: `{ex['pred_class']}` (Confidence: {ex['pred_confidence'] * 100:.2f}%)")
        lines.append(f"- **Class Probabilities**: {ex['probabilities']}\n")

    report_content = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content
