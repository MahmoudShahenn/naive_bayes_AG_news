# Hyperparameter Exploration and Model Selection Report

All experiments were conducted strictly on the **training and validation split** (108,000 train / 12,000 val), keeping the official 7,600 test split completely sequestered.

## 1. Laplace Smoothing (alpha) Evaluation
| Alpha | Val Accuracy (%) | Val Macro-F1 (%) | Train Time (s) | Pred Time (s) |
| :---: | :---: | :---: | :---: | :---: |
| 0.1 | 90.92 | 90.87 | 0.043 | 0.003 |
| 1.0 | 90.83 | 90.78 | 0.042 | 0.003 |
| 2.0 | 90.72 | 90.67 | 0.042 | 0.005 |

## 2. Feature Representation (Unigram vs Unigram+Bigram)
| Feature Representation | Vocab Size | Val Accuracy (%) | Val Macro-F1 (%) | Total Time (s) |
| :--- | :---: | :---: | :---: | :---: |
| Unigram (1,1) | 30,000 | 90.83 | 90.78 | 0.05 |
| Unigram+Bigram (1,2) | 30,000 | 90.77 | 90.71 | 12.74 |

## 3. Vocabulary Size Limit Evaluation
| Max Features Limit | Actual Vocab Size | Val Accuracy (%) | Val Macro-F1 (%) | Total Time (s) |
| :---: | :---: | :---: | :---: | :---: |
| 10,000 | 10,000 | 90.22 | 90.17 | 6.39 |
| 30,000 | 30,000 | 90.83 | 90.78 | 0.05 |
| 50,000 | 40,176 | 90.88 | 90.83 | 6.81 |

## 4. Verification Against Scikit-Learn MultinomialNB Baseline
| Model Implementation | Validation Accuracy (%) | Validation Macro-F1 (%) | Status |
| :--- | :---: | :---: | :---: |
| **From-Scratch MultinomialNB** | 90.8333% | 90.7813% | Target |
| **Scikit-Learn MultinomialNB** | 90.8333% | 90.7813% | Reference Baseline |
| **Difference (Scratch - Sklearn)** | 0.000000% | 0.000000% | Mathematically Equivalent |
