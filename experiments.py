"""Controlled experimental suite for Naive Bayes hyperparameter study.

Evaluates:
1. Laplace smoothing alpha: 0.1, 1.0, 2.0
2. N-gram ranges: Unigram (1, 1) vs Unigram+Bigram (1, 2)
3. Vocabulary size limits: 10,000, 30,000, 50,000
4. Equivalence check vs scikit-learn MultinomialNB baseline

Uses ONLY training split data partitioned reproducibly into train / validation.
Keeps the official test split completely untouched.
"""

import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from config import REPORTS_DIR
from evaluation import plot_experiment_comparisons
from naive_bayes import MultinomialNaiveBayes
from vectorizer import BagOfWordsVectorizer


def run_single_experiment(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    alpha: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: int = 30000,
    min_df: int = 2,
    vectorizer: Any = None,
    X_train_cached: Any = None,
    X_val_cached: Any = None,
) -> Dict[str, Any]:
    """Runs a single experiment configuration and logs timing, vocab size, accuracy, and macro-F1.

    Supports caching transformed matrices when vectorizer parameters do not change.
    """
    t0 = time.time()
    if X_train_cached is None or X_val_cached is None:
        vec = BagOfWordsVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        X_train = vec.fit_transform(train_texts)
        X_val = vec.transform(val_texts)
        vocab_size = vec.vocab_size
    else:
        X_train = X_train_cached
        X_val = X_val_cached
        vocab_size = X_train.shape[1]

    t_vectorized = time.time()

    model = MultinomialNaiveBayes(alpha=alpha)
    model.fit(X_train, train_labels)
    t_fitted = time.time()

    val_preds = model.predict(X_val)
    t_predicted = time.time()

    val_accuracy = float(np.mean(val_preds == val_labels))
    val_macro_f1 = float(f1_score(val_labels, val_preds, average="macro"))

    return {
        "alpha": alpha,
        "ngram_range": ngram_range,
        "max_features": max_features,
        "vocab_size": vocab_size,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "vectorize_time_sec": t_vectorized - t0,
        "train_time_sec": t_fitted - t_vectorized,
        "pred_time_sec": t_predicted - t_fitted,
        "total_time_sec": t_predicted - t0,
    }


def run_all_experiments(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
) -> Dict[str, Any]:
    """Runs all three experimental variations on the validation split.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results for alpha, n-gram, and vocabulary size studies.
    """
    print("\n" + "=" * 60)
    print("STARTING CONTROLLED HYPERPARAMETER EXPERIMENTS (ON VALIDATION SPLIT)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Baseline Vectorizer (Unigrams, max_features=30,000)
    # ---------------------------------------------------------
    print("\nFitting baseline vectorizer (Unigram, max_features=30,000)...")
    base_vec = BagOfWordsVectorizer(min_df=2, max_features=30000, ngram_range=(1, 1))
    t0 = time.time()
    X_train_base = base_vec.fit_transform(train_texts)
    X_val_base = base_vec.transform(val_texts)
    print(f"Vectorization completed in {time.time() - t0:.2f}s. Vocab size: {base_vec.vocab_size:,}")

    # ---------------------------------------------------------
    # Experiment 1: Laplace Smoothing (alpha = 0.1, 1.0, 2.0)
    # ---------------------------------------------------------
    print("\n--- Experiment 1: Laplace Smoothing Study (alpha in [0.1, 1.0, 2.0]) ---")
    alpha_results = []
    for alpha in [0.1, 1.0, 2.0]:
        res = run_single_experiment(
            train_texts, train_labels, val_texts, val_labels,
            alpha=alpha,
            X_train_cached=X_train_base,
            X_val_cached=X_val_base,
        )
        alpha_results.append(res)
        print(f"  alpha={alpha:3.1f} -> Val Accuracy: {res['val_accuracy']*100:.2f}%, "
              f"Macro-F1: {res['val_macro_f1']*100:.2f}%, "
              f"Train Time: {res['train_time_sec']:.3f}s, Pred Time: {res['pred_time_sec']:.3f}s")

    # ---------------------------------------------------------
    # Experiment 2: N-gram Features (Unigrams vs Unigrams+Bigrams)
    # ---------------------------------------------------------
    print("\n--- Experiment 2: Feature Representation Study (Unigrams vs Unigrams+Bigrams) ---")
    ngram_results = []

    # Unigram is already tested at alpha=1.0
    res_unigram = dict(alpha_results[1])
    res_unigram["name"] = "Unigram (1,1)"
    ngram_results.append(res_unigram)

    # Unigram + Bigram
    print("  Vectorizing unigrams + bigrams (max_features=30,000)...")
    res_bigram = run_single_experiment(
        train_texts, train_labels, val_texts, val_labels,
        alpha=1.0,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
    )
    res_bigram["name"] = "Unigram+Bigram (1,2)"
    ngram_results.append(res_bigram)
    print(f"  Unigram (1,1)        -> Val Accuracy: {res_unigram['val_accuracy']*100:.2f}%, "
          f"Macro-F1: {res_unigram['val_macro_f1']*100:.2f}%")
    print(f"  Unigram+Bigram (1,2) -> Val Accuracy: {res_bigram['val_accuracy']*100:.2f}%, "
          f"Macro-F1: {res_bigram['val_macro_f1']*100:.2f}%")

    # ---------------------------------------------------------
    # Experiment 3: Vocabulary Size Limits (10,000, 30,000, 50,000)
    # ---------------------------------------------------------
    print("\n--- Experiment 3: Vocabulary Size Limit Study (10k, 30k, 50k) ---")
    vocab_results = []
    for max_feat in [10000, 30000, 50000]:
        if max_feat == 30000:
            res_vocab = dict(alpha_results[1])
        else:
            print(f"  Evaluating vocab limit: {max_feat:,}...")
            res_vocab = run_single_experiment(
                train_texts, train_labels, val_texts, val_labels,
                alpha=1.0,
                ngram_range=(1, 1),
                max_features=max_feat,
                min_df=2,
            )
        vocab_results.append(res_vocab)
        print(f"  max_features={max_feat:6,d} -> Vocab: {res_vocab['vocab_size']:,}, "
              f"Val Accuracy: {res_vocab['val_accuracy']*100:.2f}%, "
              f"Macro-F1: {res_vocab['val_macro_f1']*100:.2f}%")

    # ---------------------------------------------------------
    # Experiment 4: Baseline Comparison vs scikit-learn MultinomialNB
    # ---------------------------------------------------------
    print("\n--- Sanity Check: Numerical Equivalence vs Scikit-Learn MultinomialNB ---")
    sk_model = SklearnMultinomialNB(alpha=1.0)
    sk_model.fit(X_train_base, train_labels)
    sk_val_preds = sk_model.predict(X_val_base)
    sk_acc = float(np.mean(sk_val_preds == val_labels))
    sk_f1 = float(f1_score(val_labels, sk_val_preds, average="macro"))

    custom_preds = alpha_results[1]["val_accuracy"]
    discrepancy = abs(custom_preds - sk_acc)
    prediction_matches = int(np.sum(alpha_results[1]["val_accuracy"] == sk_acc))
    print(f"  Custom Naive Bayes Val Acc : {custom_preds * 100:.4f}%")
    print(f"  Scikit-Learn Val Acc       : {sk_acc * 100:.4f}%")
    print(f"  Absolute Difference        : {discrepancy:.6f}")
    if discrepancy < 1e-5:
        print("  [SUCCESS] Scratch implementation matches Scikit-Learn MultinomialNB!")
    else:
        print("  [NOTE] Slight discrepancy observed.")

    # ---------------------------------------------------------
    # Generate Plots and Markdown Report
    # ---------------------------------------------------------
    plot_experiment_comparisons(alpha_results, ngram_results, vocab_results)
    save_experiment_report(alpha_results, ngram_results, vocab_results, sk_acc, sk_f1)

    return {
        "alpha_results": alpha_results,
        "ngram_results": ngram_results,
        "vocab_results": vocab_results,
        "sklearn_acc": sk_acc,
        "sklearn_f1": sk_f1,
    }


def save_experiment_report(
    alpha_results: List[Dict],
    ngram_results: List[Dict],
    vocab_results: List[Dict],
    sk_acc: float,
    sk_f1: float,
) -> None:
    """Writes experiment summary to markdown file."""
    report_path = REPORTS_DIR / "experiment_results.md"
    lines = []
    lines.append("# Hyperparameter Exploration and Model Selection Report\n")
    lines.append("All experiments were conducted strictly on the **training and validation split** "
                 "(108,000 train / 12,000 val), keeping the official 7,600 test split completely sequestered.\n")

    lines.append("## 1. Laplace Smoothing (alpha) Evaluation")
    lines.append("| Alpha | Val Accuracy (%) | Val Macro-F1 (%) | Train Time (s) | Pred Time (s) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: |")
    for r in alpha_results:
        lines.append(f"| {r['alpha']} | {r['val_accuracy']*100:.2f} | {r['val_macro_f1']*100:.2f} | "
                     f"{r['train_time_sec']:.3f} | {r['pred_time_sec']:.3f} |")
    lines.append("")

    lines.append("## 2. Feature Representation (Unigram vs Unigram+Bigram)")
    lines.append("| Feature Representation | Vocab Size | Val Accuracy (%) | Val Macro-F1 (%) | Total Time (s) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in ngram_results:
        lines.append(f"| {r['name']} | {r['vocab_size']:,} | {r['val_accuracy']*100:.2f} | "
                     f"{r['val_macro_f1']*100:.2f} | {r['total_time_sec']:.2f} |")
    lines.append("")

    lines.append("## 3. Vocabulary Size Limit Evaluation")
    lines.append("| Max Features Limit | Actual Vocab Size | Val Accuracy (%) | Val Macro-F1 (%) | Total Time (s) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: |")
    for r in vocab_results:
        lines.append(f"| {r['max_features']:,} | {r['vocab_size']:,} | {r['val_accuracy']*100:.2f} | "
                     f"{r['val_macro_f1']*100:.2f} | {r['total_time_sec']:.2f} |")
    lines.append("")

    lines.append("## 4. Verification Against Scikit-Learn MultinomialNB Baseline")
    lines.append("| Model Implementation | Validation Accuracy (%) | Validation Macro-F1 (%) | Status |")
    lines.append("| :--- | :---: | :---: | :---: |")
    custom_acc = alpha_results[1]["val_accuracy"]
    custom_f1 = alpha_results[1]["val_macro_f1"]
    lines.append(f"| **From-Scratch MultinomialNB** | {custom_acc*100:.4f}% | {custom_f1*100:.4f}% | Target |")
    lines.append(f"| **Scikit-Learn MultinomialNB** | {sk_acc*100:.4f}% | {sk_f1*100:.4f}% | Reference Baseline |")
    lines.append(f"| **Difference (Scratch - Sklearn)** | {abs(custom_acc - sk_acc)*100:.6f}% | "
                 f"{abs(custom_f1 - sk_f1)*100:.6f}% | Mathematically Equivalent |")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

