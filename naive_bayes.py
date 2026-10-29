"""Multinomial Naive Bayes classifier implemented from scratch using NumPy.

Supports Laplace smoothing, sparse matrix computations, log-posterior scoring,
and numerically stable softmax for class probability estimation.
"""

from typing import Optional, Union
import numpy as np
import scipy.sparse as sp


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes classifier for discrete text features.

    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
        alpha = 1.0 corresponds to standard Laplace smoothing.
        alpha = 0 corresponds to no smoothing (risking zero probability for unseen words).

    Attributes
    ----------
    classes_ : np.ndarray
        Unique class labels observed during fit.
    class_count_ : np.ndarray
        Number of training documents observed per class, shape (n_classes,).
    class_log_prior_ : np.ndarray
        Empirical log prior probability for each class, shape (n_classes,).
        Calculated as log(N_c / N).
    feature_count_ : np.ndarray
        Sum of token counts observed for each feature per class, shape (n_classes, n_features).
    feature_log_prob_ : np.ndarray
        Smoothed empirical log probability of features given a class, shape (n_classes, n_features).
        log P(w | c) = log(N_{w,c} + alpha) - log(sum_v N_{v,c} + alpha * |V|).
    n_features_in_ : int
        Number of features seen during fit.
    is_fitted_ : bool
        Whether the model has been fitted.

    Mathematical Foundations:
    ------------------------
    1. Class Priors:
       P(c) = N_c / N
       log P(c) = log(N_c) - log(N)

    2. Word Likelihood with Laplace Smoothing:
       P(w | c) = (N_{w,c} + alpha) / (sum_{v in V} N_{v,c} + alpha * |V|)
       log P(w | c) = log(N_{w,c} + alpha) - log(sum_{v in V} N_{v,c} + alpha * |V|)

    3. Log-Posterior Scoring:
       Under the Naive Bayes conditional independence assumption:
       P(d | c) = prod_{w in d} P(w | c)^{x_{d,w}}
       log P(c | d) prop log P(c) + sum_{w in d} x_{d,w} log P(w | c)

    4. Numerically Stable Softmax:
       P(c | d) = exp(log_score_{d,c} - max_k log_score_{d,k}) /
                  sum_j exp(log_score_{d,j} - max_k log_score_{d,k})

    Complexity Analysis:
    --------------------
    - Training Time Complexity: O(N_train * L_avg + |C| * |V|)
      where N_train is the number of documents, L_avg is average words per document,
      |C| is number of classes, and |V| is vocabulary size.
    - Inference Time Complexity: O(N_test * L_avg * |C|)
      Linear in the number of words in the query document and number of classes.
    - Memory Complexity: O(|C| * |V|) for model parameters.
    """

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"Smoothing parameter alpha must be non-negative, got {alpha}")
        self.alpha = float(alpha)

        # Attributes set during fit
        self.classes_: Optional[np.ndarray] = None
        self.class_count_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_count_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.is_fitted_: bool = False

    def fit(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        y: Union[np.ndarray, list],
    ) -> "MultinomialNaiveBayes":
        """Fits Multinomial Naive Bayes classifier on training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training feature count vectors.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : MultinomialNaiveBayes
            Fitted classifier.
        """
        y_arr = np.asarray(y)
        if X.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) does not match length of y ({y_arr.shape[0]})."
            )

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Identify unique classes
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)

        # Allocate arrays
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

        # Vectorized aggregation using one-hot indicator matrix for efficiency:
        # Y_indicator (n_samples, n_classes) where Y[i, c] = 1 if y[i] == class c
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([class_to_idx[val] for val in y_arr], dtype=np.int32)

        # Document count per class N_c
        for idx in range(n_classes):
            self.class_count_[idx] = np.sum(y_indices == idx)

        # Sum word counts per class N_{w,c}
        if sp.issparse(X):
            # Efficient sparse indicator multiplication: Y_indicator.T @ X
            row_idx = np.arange(n_samples)
            col_idx = y_indices
            data = np.ones(n_samples, dtype=np.float64)
            indicator = sp.csr_matrix((data, (col_idx, row_idx)), shape=(n_classes, n_samples))
            counts = indicator @ X
            if hasattr(counts, "toarray"):
                self.feature_count_ = np.asarray(counts.toarray(), dtype=np.float64)
            else:
                self.feature_count_ = np.asarray(counts, dtype=np.float64)
        else:
            for idx in range(n_classes):
                mask = y_indices == idx
                self.feature_count_[idx, :] = np.sum(X[mask, :], axis=0)

        # 1. Class Log Prior: log P(c) = log(N_c) - log(N)
        self.class_log_prior_ = np.log(self.class_count_) - np.log(n_samples)

        # 2. Word Likelihood with Laplace Smoothing:
        # P(w | c) = (N_{w,c} + alpha) / (sum_{v in V} N_{v,c} + alpha * |V|)
        smoothed_counts = self.feature_count_ + self.alpha
        smoothed_totals = np.sum(self.feature_count_, axis=1, keepdims=True) + (self.alpha * n_features)

        self.feature_log_prob_ = np.log(smoothed_counts) - np.log(smoothed_totals)
        self.is_fitted_ = True

        return self

    def _check_is_fitted(self) -> None:
        """Ensures that the model has been fitted before inference."""
        if not self.is_fitted_:
            raise RuntimeError("MultinomialNaiveBayes is not fitted yet. Call fit() first.")

    def predict_log_scores(
        self,
        X: Union[np.ndarray, sp.spmatrix],
    ) -> np.ndarray:
        """Calculates unnormalized log-posterior score for each class.

        log P(c | d) prop log P(c) + sum_{w in d} x_{d,w} log P(w | c)

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input count vectors.

        Returns
        -------
        log_scores : np.ndarray of shape (n_samples, n_classes)
            Log-posterior score for each document and class.
        """
        self._check_is_fitted()

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: X has {X.shape[1]} features, "
                f"model expected {self.n_features_in_}."
            )

        # X @ feature_log_prob_.T gives sum_{w} x_{d,w} * log P(w | c)
        prod = X @ self.feature_log_prob_.T
        if hasattr(prod, "toarray"):
            dot_product = np.asarray(prod.toarray(), dtype=np.float64)
        else:
            dot_product = np.asarray(prod, dtype=np.float64)

        # Add class log prior log P(c)
        log_scores = dot_product + self.class_log_prior_
        return log_scores

    def predict_proba(
        self,
        X: Union[np.ndarray, sp.spmatrix],
    ) -> np.ndarray:
        """Calculates normalized class posterior probabilities using stable softmax.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input count vectors.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Posterior probability distribution across classes for each sample.
        """
        log_scores = self.predict_log_scores(X)

        # Numerically stable softmax: subtract max per row to prevent exp overflow
        max_scores = np.max(log_scores, axis=1, keepdims=True)
        exp_scores = np.exp(log_scores - max_scores)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)

        probabilities = exp_scores / sum_exp
        return probabilities

    def predict(
        self,
        X: Union[np.ndarray, sp.spmatrix],
    ) -> np.ndarray:
        """Predicts class labels for query samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input count vectors.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        log_scores = self.predict_log_scores(X)
        best_class_indices = np.argmax(log_scores, axis=1)
        return self.classes_[best_class_indices]

    def score(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        y: Union[np.ndarray, list],
    ) -> float:
        """Calculates mean classification accuracy on given test data and labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test feature samples.
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        accuracy : float
            Classification accuracy in range [0.0, 1.0].
        """
        predictions = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(predictions == y_arr))
