# Multinomial Naive Bayes from Scratch for AG News Text Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

An educational, mathematically rigorous, and high-performance implementation of **Multinomial Naive Bayes from scratch** in Python and NumPy. This project classifies news articles from the AG News benchmark dataset into four categories (*World*, *Sports*, *Business*, *Sci/Tech*) and adheres strictly to academic assignment and ML engineering standards.

> **Key Guarantee**: The core text classification algorithm, feature extraction, and probability calculations are implemented completely **from scratch**. `sklearn.naive_bayes` is never used during model training or prediction; it is imported strictly in unit tests and validation checks to verify that our manual implementation achieves identical numerical results down to machine precision.

---

## Table of Contents
1. [Project Overview and Objectives](#1-project-overview-and-objectives)
2. [Dataset Description and Acquisition](#2-dataset-description-and-acquisition)
3. [Mathematical Foundations of Multinomial Naive Bayes](#3-mathematical-foundations-of-multinomial-naive-bayes)
4. [Laplace Smoothing and Numerical Stability](#4-laplace-smoothing-and-numerical-stability)
5. [Feature Representation: Bag-of-Words vs. TF-IDF](#5-feature-representation-bag-of-words-vs-tf-idf)
6. [Step-by-Step Implementation Architecture](#6-step-by-step-implementation-architecture)
7. [Installation and Execution Instructions](#7-installation-and-execution-instructions)
8. [Experimental Hyperparameter Exploration](#8-experimental-hyperparameter-exploration)
9. [Final Evaluation on the Official AG News Test Split](#9-final-evaluation-on-the-official-ag-news-test-split)
10. [Qualitative Error Analysis](#10-qualitative-error-analysis)
11. [Verification Against Scikit-Learn Baseline](#11-verification-against-scikit-learn-baseline)
12. [Theoretical Limitations and Assumptions](#12-theoretical-limitations-and-assumptions)
13. [Project Directory Layout](#13-project-directory-layout)

---

## 1. Project Overview and Objectives

The objective of this project is to build an end-to-end text classification system from fundamental mathematical principles:
- **Zero Black-Box Classifiers**: All prior estimates, smoothed likelihood calculations, sparse dot products, log-posterior scores, and softmax probability distributions are manually implemented using NumPy and SciPy sparse linear algebra.
- **Strict Separation of Splits**: The vocabulary is learned *strictly* on training documents. The official test set of 7,600 documents is sequestered until final evaluation. Hyperparameter tuning is conducted on a dedicated validation split (108,000 train / 12,000 validation).
- **High Computational Efficiency**: By leveraging SciPy `csr_matrix` representations and vectorized matrix multiplications, training on 120,000 articles executes in **under 0.05 seconds**, with test inference throughput exceeding **3,700,000 articles per second**.
- **Complete Reproducibility**: Includes automated unit testing (`pytest`), synthetic hand-calculated validation tests, automated metric generation, and visualization scripts.

---

## 2. Dataset Description and Acquisition

The project utilizes the **AG News** corpus, a standard academic text classification benchmark compiled by Antonio Gulli. The dataset comprises news headlines and descriptions across four balanced topic categories:

| Class ID | Category Name | Description | Example Keywords |
| :---: | :---: | :--- | :--- |
| `0` | **World** | International news, foreign relations, diplomacy, geopolitics | *un, minister, peace, war, iraq, president* |
| `1` | **Sports** | Athletic events, tournaments, leagues, athlete news | *olympics, cup, game, win, season, victory* |
| `2` | **Business** | Financial markets, corporate earnings, trade, stocks | *shares, market, oil, company, sales, economy* |
| `3` | **Sci/Tech** | Technology, computer science, space exploration, software | *software, microsoft, internet, space, google* |

### Official Splits
- **Training Set**: 120,000 articles (30,000 per class, exactly balanced).
- **Test Set**: 7,600 articles (1,900 per class, exactly balanced).
- **Inputs**: For each document, the `title` and `description` fields are concatenated: `f"{title} {description}"`.

### Automated Loading
Data is automatically fetched using Hugging Face's `datasets` library:
```python
from datasets import load_dataset
dataset = load_dataset("fancyzhx/ag_news")
```
If network access is unavailable, the loader automatically detects cached versions in `~/.cache/huggingface/datasets/` or sets `HF_HUB_OFFLINE=1` to guarantee fast, zero-delay offline execution.

---

## 3. Mathematical Foundations of Multinomial Naive Bayes

In text classification, a document $d$ is represented as a bag-of-words vector $\mathbf{x} = [x_1, x_2, \dots, x_{|V|}]$, where $x_w$ is the count of word $w$ from a fixed vocabulary $V$. We wish to determine the most probable class $c \in \{0, 1, 2, 3\}$.

By **Bayes' Theorem**:
$$P(c \mid d) = \frac{P(c) P(d \mid c)}{P(d)}$$

Since the evidence $P(d)$ is constant across all candidate classes $c$, maximizing the posterior is equivalent to maximizing the joint probability:
$$c^* = \arg\max_{c \in C} P(c) P(d \mid c)$$

### The Multinomial Event Model
Under the Multinomial distribution, a document is viewed as a sequence of independent draws from a categorical distribution over vocabulary words:
$$P(d \mid c) = \frac{\left(\sum_{w \in V} x_w\right)!}{\prod_{w \in V} x_w!} \prod_{w \in V} P(w \mid c)^{x_w}$$

Because the multinomial coefficient depends only on document token counts and is identical for all classes $c$, it can be dropped during classification:
$$P(c \mid d) \propto P(c) \prod_{w \in d} P(w \mid c)^{x_w}$$

### 1. Class Prior Probabilities
For each class $c$, the Maximum Likelihood Estimate (MLE) of the prior is the empirical document frequency:
$$P(c) = \frac{N_c}{N}$$
where $N_c$ is the number of training articles in class $c$, and $N = \sum_c N_c$ is the total number of training articles.

### 2. Word Likelihoods with Laplace Smoothing
The MLE for the probability of observing word $w$ given class $c$ is the fraction of total word occurrences in class $c$ that are word $w$:
$$P(w \mid c) = \frac{N_{w,c} + \alpha}{\sum_{v \in V} N_{v,c} + \alpha |V|}$$
where:
- $N_{w,c} = \sum_{i: y_i = c} X_{i, w}$ is the total count of word $w$ in all class-$c$ training documents.
- $\sum_{v \in V} N_{v,c}$ is the total word count across all features for class $c$.
- $|V|$ is the size of the vocabulary.
- $\alpha \ge 0$ is the additive smoothing parameter (default $\alpha = 1.0$ for Laplace smoothing).

---

## 4. Laplace Smoothing and Numerical Stability

### Why Laplace Smoothing is Crucial
If a word $w$ appears in the test document but never appeared in class $c$ during training ($N_{w,c} = 0$), the unsmoothed likelihood would be zero:
$$P(w \mid c) = 0 \implies \prod_{w' \in d} P(w' \mid c)^{x_{w'}} = 0$$
A single zero likelihood would wipe out all other evidence in document $d$, completely ruling out class $c$ regardless of how many other strong signals were present. 

Setting $\alpha = 1.0$ (Laplace smoothing) or $\alpha \in (0, 1)$ (Lidstone smoothing) assigns a small, principled pseudo-count to every vocabulary term, preventing the zero-frequency problem and acting as a uniform Dirichlet prior over the multinomial parameters.

### Log-Space Computation (Underflow Prevention)
A document may contain dozens or hundreds of tokens. Multiplying hundreds of probabilities $P(w \mid c) \ll 1$ causes immediate floating-point **arithmetic underflow** (rounding to 0.0 in IEEE 754 float64).

To eliminate underflow, calculations are performed in **logarithmic space**:
$$\log P(c \mid d) \propto \log P(c) + \sum_{w \in d} x_w \log P(w \mid c)$$

### Numerically Stable Softmax (Log-Sum-Exp Trick)
When class posterior probabilities $P(c \mid d) \in [0, 1]$ are required, exponentiating raw log-posterior scores directly can cause overflow if scores are positive or underflow if scores are large negative values. We apply the **log-sum-exp trick**:
$$M = \max_{k \in C} \log P(k \mid d)$$
$$P(c \mid d) = \frac{\exp\left(\log P(c \mid d) - M\right)}{\sum_{j \in C} \exp\left(\log P(j \mid d) - M\right)}$$
Subtracting the maximum score ensures that the largest exponent is $\exp(0) = 1.0$, guaranteeing numerical stability without altering the resulting probability distribution.

---

## 5. Feature Representation: Bag-of-Words vs. TF-IDF

| Characteristic | Bag-of-Words (BoW) | TF-IDF |
| :--- | :--- | :--- |
| **Values** | Non-negative integers: $x_{d,w} \in \{0, 1, 2, \dots\}$ | Continuous real weights: $w_{d,w} \in \mathbb{R}^+$ |
| **Generative Assumption** | Matches Multinomial distribution: draws of discrete tokens | Violates discrete multinomial distribution |
| **Weighting Philosophy** | Direct term frequency count | Penalizes frequent words via $\log(N / \text{df}_w)$ |
| **Suitability for MNB** | **Mathematically rigorous and correct** | Ad-hoc heuristic when paired with Multinomial NB |

**Conclusion**: Multinomial Naive Bayes is derived directly from a multinomial count distribution. Bag-of-Words count vectors are the theoretically consistent representation and are used as our primary feature pipeline.

---

## 6. Step-by-Step Implementation Architecture

The project is structured into modular components:

```
naive-bayes-ag-news/
├── config.py           # Hyperparameters, directory constants, class mappings
├── preprocessing.py    # Sanitization, regex tokenization, stopword removal
├── vectorizer.py       # Custom Bag-of-Words vectorizer & CSR sparse builder
├── naive_bayes.py      # From-scratch Multinomial Naive Bayes implementation
├── data_loader.py      # Hugging Face dataset loading & validation split logic
├── evaluation.py       # Accuracy, precision, recall, F1, confusion matrix, plots
├── experiments.py      # Reproducible validation study (alpha, n-grams, vocab)
├── main.py             # Orchestration script for full training & evaluation
├── tests/
│   └── test_naive_bayes.py # 13 unit tests covering math, edge cases, equivalence
└── outputs/
    ├── plots/          # Generated heatmaps and bar charts
    └── reports/        # Markdown evaluation and experiment reports
```

### 1. `TextPreprocessor` ([`preprocessing.py`](file:///d:/naive_bayes_ag_news/preprocessing.py))
- Converts all text to lowercase.
- Strips any residual HTML tags (`<[^>]+>`).
- Tokenizes words using regex boundary matching `\b[a-zA-Z0-9_]+\b`.
- Removes standard English stopwords using a self-contained set of 150+ words (preventing external NLTK downloads).
- Handles empty strings, whitespace, and `None` safely.

### 2. `BagOfWordsVectorizer` ([`vectorizer.py`](file:///d:/naive_bayes_ag_news/vectorizer.py))
- `fit(raw_documents)`:
  - Iterates exclusively over training documents.
  - Computes document frequencies and collection frequencies.
  - Filters terms by `min_df` (default 2) and `max_features` (top 30,000 by corpus frequency).
  - Assigns unique integer indices to vocabulary terms: `vocabulary_: Dict[str, int]`.
- `transform(raw_documents)`:
  - Constructs a `scipy.sparse.csr_matrix` via COO indexing `(data, (rows, cols))` in $O(N_{\text{tokens}})$.
  - Prevents RAM exhaustion (a dense $120,000 \times 30,000$ float32 matrix would require $\approx 14.4\text{ GB}$; the CSR matrix requires only $\approx 35\text{ MB}$).

### 3. `MultinomialNaiveBayes` ([`naive_bayes.py`](file:///d:/naive_bayes_ag_news/naive_bayes.py))
- `fit(X, y)`:
  - Computes class counts $N_c$ and class log priors: $\log P(c) = \log(N_c) - \log(N)$.
  - Constructs a sparse indicator matrix to sum word occurrences per class in a single sparse operation:
    $$\mathbf{N}_{\text{class}} = \mathbf{Y}_{\text{indicator}}^T \mathbf{X} \in \mathbb{R}^{|C| \times |V|}$$
  - Computes Laplace-smoothed feature log probabilities:
    $$\log P(w \mid c) = \log(N_{w,c} + \alpha) - \log\left(\sum_v N_{v,c} + \alpha |V|\right)$$
- `predict_log_scores(X)`:
  - Performs a single matrix-matrix dot product:
    $$\text{Scores} = \mathbf{X} \cdot (\log \mathbf{P})^T + \log \mathbf{P}_{\text{prior}}$$
- `predict(X)`:
  - Returns `classes_[np.argmax(scores, axis=1)]`.
- `predict_proba(X)`:
  - Applies numerically stable softmax across class scores.

---

## 7. Installation and Execution Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux

### 1. Setup Environment
Clone or navigate to the repository directory and install the required dependencies:
```bash
cd d:/naive_bayes_ag_news
pip install -r requirements.txt
```

### 2. Run Unit Tests
Execute the comprehensive test suite:
```bash
python -m pytest -v tests/
```
All 13 unit tests will execute and pass, verifying class priors, smoothed likelihoods, synthetic hand calculations, empty inputs, unseen words, probability axioms, and exact equivalence with Scikit-Learn.

### 3. Run Complete Training and Evaluation Pipeline
Execute the full workflow:
```bash
python main.py
```
This single command automatically:
1. Loads the AG News corpus.
2. Partitions the training data into a 108,000 / 12,000 train/validation split.
3. Runs all 3 hyperparameter experiments on the validation set.
4. Trains the final Naive Bayes model on the full 120,000-sample training set.
5. Evaluates on the official 7,600-sample test set.
6. Generates confusion matrices, metric charts, and markdown diagnostic reports under `outputs/`.

---

## 8. Experimental Hyperparameter Exploration

All experiments were conducted **strictly on the training and validation split** (108,000 train / 12,000 val). The official 7,600 test split was sequestered.

### Experiment 1: Effect of Laplace Smoothing ($\alpha$)
Evaluated with unigram features ($|V| = 30,000$):

| Smoothing Parameter ($\alpha$) | Validation Accuracy | Validation Macro-F1 | Fit Time (s) | Inference Time (s) |
| :---: | :---: | :---: | :---: | :---: |
| $\alpha = 0.1$ | **90.92%** | **90.87%** | 0.042 s | 0.002 s |
| $\alpha = 1.0$ (Standard) | 90.83% | 90.78% | 0.042 s | 0.003 s |
| $\alpha = 2.0$ | 90.72% | 90.67% | 0.042 s | 0.003 s |

*Observation*: Smaller smoothing ($\alpha = 0.1$) provides a modest performance advantage ($+0.09\%$) because the training set is very large (108,000 docs), meaning empirical frequencies are already well-estimated. However, $\alpha = 1.0$ remains robust across all classes.

### Experiment 2: Feature Representation (Unigram vs. Unigram+Bigram)
Evaluated with $\alpha = 1.0$ and $\text{max\_features} = 30,000$:

| Feature Representation | Vocabulary Size | Validation Accuracy | Validation Macro-F1 | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Unigram (1, 1)** | 30,000 | **90.83%** | **90.78%** | 6.34 s |
| **Unigram + Bigram (1, 2)** | 30,000 | 90.77% | 90.71% | 24.12 s |

*Observation*: While bigrams capture word collocations (e.g., "Wall Street", "space flight"), under a fixed vocabulary cap of 30,000, bigrams displace informative unigrams, yielding slightly lower accuracy while increasing feature extraction time $4\times$. Pure unigrams represent the superior speed-to-accuracy trade-off.

### Experiment 3: Vocabulary Size Limit
Evaluated with unigrams and $\alpha = 1.0$:

| Max Features Limit | Actual Vocabulary Size | Validation Accuracy | Validation Macro-F1 | Total Time (s) |
| :---: | :---: | :---: | :---: | :---: |
| 10,000 | 10,000 | 90.22% | 90.17% | 5.81 s |
| **30,000 (Default)** | **30,000** | **90.83%** | **90.78%** | **6.34 s** |
| 50,000 | 40,176* | **90.88%** | **90.83%** | 6.55 s |

*\*Note: With $\text{min\_df} = 2$, only 40,176 distinct unigrams meet the frequency threshold in the sub-training corpus.*

*Observation*: Expanding vocabulary from 10,000 to 30,000 yields a significant $+0.61\%$ accuracy jump. Increasing further to 40,176 yields marginal gain ($+0.05\%$) at the cost of larger parameter tables. 30,000 represents an optimal balance.

---

## 9. Final Evaluation on the Official AG News Test Split

The final model was fitted on the **complete 120,000-article training set** using $\alpha = 1.0$ and unigrams ($\text{max\_features} = 30,000$) and evaluated on the official **7,600-article test set**.

### Overall Performance Summary
- **Training Set Size**: 120,000 articles
- **Test Set Size**: 7,600 articles
- **Vocabulary Size**: 30,000 features
- **Model Training Time**: **0.045 seconds**
- **Test Prediction Time**: **0.002 seconds** (Throughput: **3,785,383 articles/sec**)
- **Final Test Accuracy**: **90.18%**
- **Macro-Averaged F1**: **90.14%**
- **Weighted-Averaged F1**: **90.14%**

### Per-Class Performance Breakdown

| Class Name | Precision (%) | Recall (%) | F1-Score (%) | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 90.89% | 89.84% | **90.37%** | 1,900 |
| **Sports** | 94.91% | 98.11% | **96.48%** | 1,900 |
| **Business** | 87.57% | 84.53% | **86.02%** | 1,900 |
| **Sci/Tech** | 87.16% | 88.26% | **87.71%** | 1,900 |
| **Macro Average** | **90.13%** | **90.18%** | **90.14%** | **7,600** |

### Confusion Matrix

| True \\ Predicted | World | Sports | Business | Sci/Tech | Total True |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **World** | **1,707** | 60 | 91 | 42 | 1,900 |
| **Sports** | 20 | **1,864** | 8 | 8 | 1,900 |
| **Business** | 78 | 19 | **1,606** | 197 | 1,900 |
| **Sci/Tech** | 73 | 21 | 129 | **1,677** | 1,900 |
| **Total Predicted** | 1,878 | 1,964 | 1,834 | 1,924 | 7,600 |

*Key Findings*:
1. **Sports** achieves the highest classification F1 (96.48%) due to distinct, specialized athletic vocabulary (*olympics, tournament, medal, coach, points*).
2. The greatest confusion occurs between **Business** and **Sci/Tech** (197 Business articles misclassified as Sci/Tech; 129 Sci/Tech articles misclassified as Business). This reflects genuine topical intersection in modern business-technology reporting (tech company quarterly earnings, tech stocks, antitrust lawsuits, IPOs).

---

## 10. Qualitative Error Analysis

To understand model behavior, individual predictions were analyzed with their posterior probability distributions:

### High-Confidence Correct Classifications
1. **Article**: *"Fears for T N pension after talks Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."*
   - Ground Truth: `Business` | Prediction: `Business` (Confidence: **100.00%**)
2. **Article**: *"The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)... contest for privately funded suborbital space flight..."*
   - Ground Truth: `Sci/Tech` | Prediction: `Sci/Tech` (Confidence: **100.00%**)
3. **Article**: *"Prediction Unit Helps Forecast Wildfires (AP)... Lightning will strike in places he expects. Winds will pick up, moist places will dry and flames will roar."*
   - Ground Truth: `Sci/Tech` | Prediction: `Sci/Tech` (Confidence: **95.07%**)

### Representative Misclassifications and Analysis
1. **Article (Index 19)**: *"Storage, servers bruise HP earnings update. Earnings per share rise compared with a year ago, but company misses analysts' expectations by a long shot."*
   - **Ground Truth**: `Sci/Tech`
   - **Model Prediction**: `Business` (Confidence: **97.97%**)
   - **Reason**: The article discusses Hewlett-Packard (a hardware company), but the vocabulary is overwhelmingly financial: *earnings, shares, analysts, expectations*. The model's prediction of `Business` is entirely rational given the word evidence.
2. **Article (Index 23)**: *"Some People Not Eligible to Get in on Google IPO. Google has billed its IPO as a way for everyday people to get in on the process, denying Wall Street the usual stranglehold it's had on IPOs. Public bidding, a minimum of just five shares, an open process with 28 underwriters..."*
   - **Ground Truth**: `Sci/Tech`
   - **Model Prediction**: `Business` (Confidence: **67.74%**)
   - **Reason**: While Google is a tech enterprise, the article focuses exclusively on Wall Street, underwriters, IPO bidding, and shares.
3. **Article (Index 41)**: *"Retailers Vie for Back-To-School Buyers (Reuters)... Apparel retailers are hoping their back-to-school fashions will make the grade among style-conscious teens... keeping a tighter hold on their wallets."*
   - **Ground Truth**: `Business`
   - **Model Prediction**: `Sci/Tech` (Confidence: **79.41%**)
   - **Reason**: Words like *grade*, *buyers*, and *wallets* overlapped with consumer-electronics review articles in the Sci/Tech category.

---

## 11. Verification Against Scikit-Learn Baseline

To confirm mathematical correctness, our from-scratch `MultinomialNaiveBayes` was compared head-to-head with `sklearn.naive_bayes.MultinomialNB` under identical input data ($\alpha = 1.0$, identical sparse Bag-of-Words matrix on the validation split):

| Metric | From-Scratch Implementation | Scikit-Learn MultinomialNB | Absolute Discrepancy |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **90.8333%** | **90.8333%** | **0.000000** |
| **Validation Macro-F1** | **90.7812%** | **90.7812%** | **0.000000** |
| **Class Log Prior ($\log P(c)$)** | Identical array | Identical array | $< 10^{-12}$ |
| **Feature Log Prob ($\log P(w \mid c)$)**| Identical matrix | Identical matrix | $< 10^{-12}$ |
| **Predicted Labels** | Exact match | Exact match | **0 mismatched samples** |

**Conclusion**: The manual implementation is mathematically equivalent to the standard reference implementation down to machine precision.

---

## 12. Theoretical Limitations and Assumptions

While Multinomial Naive Bayes achieves >90% accuracy and millisecond inference speeds on AG News, it relies on several structural assumptions:

1. **Conditional Independence Assumption**:
   Naive Bayes assumes that word occurrences are mutually independent given the class:
   $$P(w_1, w_2 \mid c) = P(w_1 \mid c) P(w_2 \mid c)$$
   In natural language, this assumption is frequently violated (e.g., "Wall" strongly correlates with "Street"; "Hong" strongly correlates with "Kong"). While this does not harm classification if the argmax remains stable, it results in **poorly calibrated, overconfident posterior probabilities** (many probabilities appear as $>99.9\%$ or $<0.01\%$).
2. **Loss of Word Order and Syntax**:
   The Bag-of-Words representation ignores grammar, syntax, negation scope, and semantic nuance. "Not bad, quite good" and "Not good, quite bad" have identical word counts.
3. **Linear Decision Boundary in Log Space**:
   Naive Bayes is equivalent to a linear classifier in log-odds space ($\mathbf{w}^T \mathbf{x} + b$). It cannot learn non-linear feature interactions unless explicitly augmented with n-grams or kernel methods.

---

## 13. Project Directory Layout

```text
naive-bayes-ag-news/
├── README.md                      # Comprehensive documentation and project report
├── requirements.txt               # Dependency specifications
├── config.py                      # Global configuration and hyperparameter settings
├── preprocessing.py               # Text cleaning, regex tokenization, stopwords
├── vectorizer.py                  # Custom Bag-of-Words sparse vectorizer
├── naive_bayes.py                 # From-scratch Multinomial Naive Bayes
├── data_loader.py                 # Dataset loading and split management
├── evaluation.py                  # Evaluation metrics, plotting, reporting
├── experiments.py                 # Hyperparameter validation experiments
├── main.py                        # Complete pipeline execution script
├── tests/
│   └── test_naive_bayes.py        # 13 automated unit tests
└── outputs/
    ├── plots/
    │   ├── confusion_matrix.png       # Confusion matrix heatmap
    │   ├── class_metrics.png          # Precision, recall, F1 bar chart
    │   └── experiments_comparison.png # Hyperparameter comparison charts
    └── reports/
        ├── experiment_results.md      # Validation study table report
        └── final_evaluation_report.md # Official test set diagnostic report
```

