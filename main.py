"""Main entrypoint for AG News text classification using from-scratch Naive Bayes.

Orchestrates:
1. Loading the official AG News dataset (train and test splits).
2. Reproducibly partitioning training data into train / validation sets for experiments.
3. Running controlled hyperparameter experiments (Laplace smoothing, N-grams, vocab size).
4. Training the final from-scratch Multinomial Naive Bayes model on the full training set.
5. Evaluating on the official untouched test set (7,600 articles).
6. Generating plots, reports, and qualitative error analysis.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

from config import (
    CLASS_NAMES,
    LABEL_MAPPING,
    PLOTS_DIR,
    REPORTS_DIR,
    ProjectConfig,
)
from data_loader import create_train_val_split, load_ag_news_dataset
from evaluation import (
    compute_metrics,
    extract_qualitative_examples,
    generate_evaluation_report,
    plot_class_metrics,
    plot_confusion_matrix,
)
from experiments import run_all_experiments
from naive_bayes import MultinomialNaiveBayes
from vectorizer import BagOfWordsVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Multinomial Naive Bayes from scratch on AG News."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter (default: 1.0)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=30000,
        help="Maximum vocabulary size (default: 30000)",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=2,
        help="Minimum document frequency (default: 2)",
    )
    parser.add_argument(
        "--skip_experiments",
        action="store_true",
        help="Skip validation hyperparameter experiments and directly train final model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ProjectConfig()

    print("=" * 70)
    print("MULTINOMIAL NAIVE BAYES FROM SCRATCH: AG NEWS CLASSIFICATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Step 1: Load AG News Dataset
    # -------------------------------------------------------------
    print("\n[Step 1/5] Loading AG News dataset via Hugging Face...")
    t0_data = time.time()
    try:
        texts, labels = load_ag_news_dataset()
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    train_texts = texts["train"]
    train_labels = labels["train"]
    test_texts = texts["test"]
    test_labels = labels["test"]

    print(f"  Loaded in {time.time() - t0_data:.2f}s")
    print(f"  Training Split : {len(train_texts):,} articles")
    print(f"  Test Split     : {len(test_texts):,} articles")
    print(f"  Classes        : {CLASS_NAMES}")

    # -------------------------------------------------------------
    # Step 2: Create Validation Split for Hyperparameter Experiments
    # -------------------------------------------------------------
    print("\n[Step 2/5] Creating reproducible train/val split from training set...")
    sub_train_texts, sub_train_labels, val_texts, val_labels = create_train_val_split(
        train_texts,
        train_labels,
        val_ratio=cfg.experiment.val_split_ratio,
        random_seed=cfg.experiment.random_seed,
    )
    print(f"  Sub-training Split : {len(sub_train_texts):,} articles")
    print(f"  Validation Split   : {len(val_texts):,} articles (for experiments only)")
    print(f"  Official Test Split: {len(test_texts):,} articles (sequestered until final evaluation)")

    # -------------------------------------------------------------
    # Step 3: Run Controlled Experiments on Validation Split
    # -------------------------------------------------------------
    if not args.skip_experiments:
        print("\n[Step 3/5] Running controlled hyperparameter experiments...")
        t0_exp = time.time()
        exp_results = run_all_experiments(
            sub_train_texts,
            sub_train_labels,
            val_texts,
            val_labels,
        )
        print(f"\nHyperparameter experiments completed in {time.time() - t0_exp:.2f}s")
    else:
        print("\n[Step 3/5] Skipping hyperparameter experiments (--skip_experiments set).")

    # -------------------------------------------------------------
    # Step 4: Final Vectorization and Model Training
    # -------------------------------------------------------------
    print("\n[Step 4/5] Fitting final Bag-of-Words vectorizer & MultinomialNB on FULL training set...")
    vectorizer = BagOfWordsVectorizer(
        min_df=args.min_df,
        max_features=args.max_features,
        ngram_range=(1, 1),
    )

    t0_vec = time.time()
    X_train = vectorizer.fit_transform(train_texts)
    vec_train_time = time.time() - t0_vec

    t0_vec_test = time.time()
    X_test = vectorizer.transform(test_texts)
    vec_test_time = time.time() - t0_vec_test

    vocab_size = vectorizer.vocab_size
    print(f"  Training vectorization completed in {vec_train_time:.2f}s")
    print(f"  Test vectorization completed in {vec_test_time:.2f}s")
    print(f"  Final Vocabulary Size: {vocab_size:,} features")
    print(f"  X_train shape: {X_train.shape}, non-zero elements: {X_train.nnz:,}")
    print(f"  X_test shape : {X_test.shape}, non-zero elements: {X_test.nnz:,}")

    # Train from-scratch Multinomial Naive Bayes
    print(f"\nTraining from-scratch MultinomialNaiveBayes (alpha={args.alpha})...")
    model = MultinomialNaiveBayes(alpha=args.alpha)
    t0_fit = time.time()
    model.fit(X_train, train_labels)
    train_time = time.time() - t0_fit
    print(f"  Model fitting completed in {train_time:.3f}s")

    # -------------------------------------------------------------
    # Step 5: Test Set Evaluation & Diagnostics
    # -------------------------------------------------------------
    print("\n[Step 5/5] Evaluating on the official test set (7,600 articles)...")
    t0_pred = time.time()
    test_preds = model.predict(X_test)
    pred_time = time.time() - t0_pred

    test_probas = model.predict_proba(X_test)
    metrics = compute_metrics(test_labels, test_preds, class_names=CLASS_NAMES)

    print(f"  Prediction completed in {pred_time:.3f}s ({len(test_labels) / pred_time:.1f} articles/s)")
    print(f"\n{'*'*50}")
    print(f"  FINAL TEST ACCURACY : {metrics['accuracy'] * 100:.2f}%")
    print(f"  MACRO-AVERAGE F1    : {metrics['macro_f1'] * 100:.2f}%")
    print(f"  WEIGHTED-AVERAGE F1 : {metrics['weighted_f1'] * 100:.2f}%")
    print(f"{'*'*50}\n")

    print("Classification Report:")
    print(metrics["report_str"])

    # Qualitative examples
    correct_samples, misclassified_samples = extract_qualitative_examples(
        test_texts, test_labels, test_preds, test_probas, n_examples=5
    )

    # Generate visual plots
    print("Generating and saving publication plots...")
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=CLASS_NAMES, save_path=cm_path)
    print(f"  Saved confusion matrix heatmap to: {cm_path}")

    class_metrics_path = PLOTS_DIR / "class_metrics.png"
    plot_class_metrics(metrics["per_class"], save_path=class_metrics_path)
    print(f"  Saved class metrics bar chart to: {class_metrics_path}")

    # Generate evaluation report
    report_path = REPORTS_DIR / "final_evaluation_report.md"
    generate_evaluation_report(
        metrics=metrics,
        train_size=len(train_texts),
        test_size=len(test_texts),
        vocab_size=vocab_size,
        train_time_sec=train_time,
        pred_time_sec=pred_time,
        correct_samples=correct_samples,
        misclassified_samples=misclassified_samples,
        save_path=report_path,
    )
    print(f"  Saved comprehensive evaluation report to: {report_path}")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

