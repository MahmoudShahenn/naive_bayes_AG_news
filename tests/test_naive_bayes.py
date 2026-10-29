"""Unit tests for from-scratch Multinomial Naive Bayes, preprocessing, and vectorizer.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from naive_bayes import MultinomialNaiveBayes
from preprocessing import TextPreprocessor
from vectorizer import BagOfWordsVectorizer


class TestPreprocessing:
    """Tests for text cleaning, tokenization, and stopword removal."""

    def test_lowercase_and_punctuation(self):
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True, remove_stopwords=False)
        tokens = preprocessor.tokenize("Hello, World! Fast-paced 2026.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "fast" in tokens
        assert "paced" in tokens
        assert "2026" in tokens

    def test_stopwords_removal(self):
        preprocessor = TextPreprocessor(remove_stopwords=True)
        tokens = preprocessor.tokenize("This is a sports match with the team")
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens
        assert "sports" in tokens
        assert "match" in tokens
        assert "team" in tokens

    def test_empty_and_none_text(self):
        preprocessor = TextPreprocessor()
        assert preprocessor.tokenize("") == []
        assert preprocessor.tokenize(None) == []
        assert preprocessor.tokenize("   \n\t  ") == []


class TestVectorizer:
    """Tests for Bag-of-Words feature extraction and sparse matrix conversion."""

    def test_vocabulary_construction_no_leakage(self):
        train_docs = ["apple orange banana", "apple apple orange", "banana grape"]
        test_docs = ["apple watermelon", "dragonfruit"]

        vec = BagOfWordsVectorizer(min_df=1)
        vec.fit(train_docs)

        # Ensure vocabulary is built exclusively from train_docs
        assert "apple" in vec.vocabulary_
        assert "banana" in vec.vocabulary_
        assert "watermelon" not in vec.vocabulary_
        assert "dragonfruit" not in vec.vocabulary_

        X_test = vec.transform(test_docs)
        assert sp.issparse(X_test)
        assert X_test.shape == (2, len(vec.vocabulary_))

    def test_max_features_and_min_df(self):
        docs = [
            "cat dog bird",
            "cat dog",
            "cat fish",
            "cat",
        ]
        # cat appears in 4 docs, dog in 2, bird in 1, fish in 1
        vec = BagOfWordsVectorizer(min_df=2, max_features=2)
        X = vec.fit_transform(docs)
        assert vec.vocab_size == 2
        assert set(vec.feature_names_) == {"cat", "dog"}
        assert X.shape == (4, 2)

    def test_bigram_extraction(self):
        docs = ["machine learning algorithms"]
        vec = BagOfWordsVectorizer(min_df=1, ngram_range=(1, 2))
        vec.fit(docs)
        assert "machine learning" in vec.vocabulary_
        assert "learning algorithms" in vec.vocabulary_


class TestMultinomialNaiveBayes:
    """Tests for from-scratch Multinomial Naive Bayes classifier."""

    @pytest.fixture
    def synthetic_data(self):
        """Creates a tiny synthetic corpus verifiable by hand.

        Class 0: 2 docs ("meow fish", "meow meow")
        Class 1: 2 docs ("bark bone", "bark bark")
        Vocab: ["bark", "bone", "fish", "meow"]
        """
        # Feature columns: [bark, bone, fish, meow]
        X = np.array([
            [0, 0, 1, 1],  # doc 0: fish, meow (class 0)
            [0, 0, 0, 2],  # doc 1: meow, meow (class 0)
            [1, 1, 0, 0],  # doc 2: bark, bone (class 1)
            [2, 0, 0, 0],  # doc 3: bark, bark (class 1)
        ], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        return X, y

    def test_class_priors(self, synthetic_data):
        X, y = synthetic_data
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(X, y)

        # 2 samples in class 0, 2 in class 1 -> P(c) = 0.5 each
        expected_log_priors = np.array([np.log(0.5), np.log(0.5)])
        np.testing.assert_allclose(model.class_log_prior_, expected_log_priors, rtol=1e-6)

    def test_word_counts_and_laplace_smoothing(self, synthetic_data):
        X, y = synthetic_data
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(X, y)

        # Word counts:
        # Class 0: bark=0, bone=0, fish=1, meow=3 (total = 4)
        # Class 1: bark=3, bone=1, fish=0, meow=0 (total = 4)
        expected_feature_counts = np.array([
            [0, 0, 1, 3],
            [3, 1, 0, 0],
        ], dtype=np.float64)
        np.testing.assert_array_equal(model.feature_count_, expected_feature_counts)

        # Laplace smoothed probabilities with alpha=1, |V|=4:
        # Denominator for both classes: 4 + 1 * 4 = 8
        # Class 0: [1/8, 1/8, 2/8, 4/8] = [0.125, 0.125, 0.25, 0.5]
        # Class 1: [4/8, 2/8, 1/8, 1/8] = [0.5, 0.25, 0.125, 0.125]
        expected_log_prob = np.log([
            [0.125, 0.125, 0.25, 0.5],
            [0.5, 0.25, 0.125, 0.125],
        ])
        np.testing.assert_allclose(model.feature_log_prob_, expected_log_prob, rtol=1e-6)

    def test_hand_calculated_prediction_and_softmax(self, synthetic_data):
        X, y = synthetic_data
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(X, y)

        # Query: "meow" -> vector [0, 0, 0, 1]
        # P(0 | meow) proportional to P(0) * P(meow|0) = 0.5 * 0.5 = 0.25
        # P(1 | meow) proportional to P(1) * P(meow|1) = 0.5 * 0.125 = 0.0625
        # Softmax:
        # P(0 | meow) = 0.25 / (0.25 + 0.0625) = 0.25 / 0.3125 = 0.80 (80%)
        # P(1 | meow) = 0.0625 / 0.3125 = 0.20 (20%)
        q = np.array([[0, 0, 0, 1]])
        pred = model.predict(q)
        proba = model.predict_proba(q)

        assert pred[0] == 0
        np.testing.assert_allclose(proba[0], [0.80, 0.20], rtol=1e-5)

    def test_unseen_words_and_empty_document(self, synthetic_data):
        X, y = synthetic_data
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(X, y)

        # Empty document: [0, 0, 0, 0]
        # Since class priors are equal (50% each), posterior must be equal (50% each)
        empty_doc = np.array([[0, 0, 0, 0]])
        proba_empty = model.predict_proba(empty_doc)
        np.testing.assert_allclose(proba_empty[0], [0.5, 0.5], rtol=1e-6)

    def test_probability_distribution_properties(self, synthetic_data):
        X, y = synthetic_data
        model = MultinomialNaiveBayes(alpha=1.0)
        model.fit(X, y)

        queries = np.array([
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [2, 3, 0, 1],
        ])
        probas = model.predict_proba(queries)

        # Check shapes and bounds: each row in [0, 1] and sum == 1
        assert probas.shape == (3, 2)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        np.testing.assert_allclose(np.sum(probas, axis=1), np.ones(3), rtol=1e-6)

    def test_sparse_and_dense_matrix_compatibility(self, synthetic_data):
        X_dense, y = synthetic_data
        X_sparse = sp.csr_matrix(X_dense)

        model_dense = MultinomialNaiveBayes(alpha=1.0).fit(X_dense, y)
        model_sparse = MultinomialNaiveBayes(alpha=1.0).fit(X_sparse, y)

        np.testing.assert_allclose(model_dense.feature_log_prob_, model_sparse.feature_log_prob_)

        pred_dense = model_dense.predict(X_dense)
        pred_sparse = model_sparse.predict(X_sparse)
        np.testing.assert_array_equal(pred_dense, pred_sparse)

    def test_equivalence_with_scikit_learn(self):
        """Verifies scratch MultinomialNB produces identical numerical results to scikit-learn."""
        rng = np.random.RandomState(42)
        n_samples = 100
        n_features = 50
        X = rng.poisson(lam=1.5, size=(n_samples, n_features))
        y = rng.choice([0, 1, 2, 3], size=n_samples)

        X_sparse = sp.csr_matrix(X)

        custom_nb = MultinomialNaiveBayes(alpha=1.0).fit(X_sparse, y)
        sklearn_nb = SklearnMultinomialNB(alpha=1.0).fit(X_sparse, y)

        # Priors match
        np.testing.assert_allclose(custom_nb.class_log_prior_, sklearn_nb.class_log_prior_, rtol=1e-6)

        # Feature log likelihoods match
        np.testing.assert_allclose(custom_nb.feature_log_prob_, sklearn_nb.feature_log_prob_, rtol=1e-6)

        # Predictions match
        custom_preds = custom_nb.predict(X_sparse)
        sklearn_preds = sklearn_nb.predict(X_sparse)
        np.testing.assert_array_equal(custom_preds, sklearn_preds)

        # Probabilities match
        custom_proba = custom_nb.predict_proba(X_sparse)
        sklearn_proba = sklearn_nb.predict_proba(X_sparse)
        np.testing.assert_allclose(custom_proba, sklearn_proba, rtol=1e-5)

