"""Bag-of-Words Vectorizer implemented from scratch.

Converts collections of raw text documents into sparse Bag-of-Words count matrices
(scipy.sparse.csr_matrix) with support for min_df, max_features, and n-grams.

Difference between Bag-of-Words (BoW) and TF-IDF:
-------------------------------------------------
1. Bag-of-Words (BoW):
   Represents each document as a vector of integer counts, where each element x_{d, w}
   is the frequency of token w in document d.
   Multinomial Naive Bayes models the generative probability of text using a
   Multinomial distribution over word counts:
       P(d | c) = (N_d! / prod_w x_{d,w}!) * prod_w P(w | c)^{x_{d,w}}
   Because Multinomial Naive Bayes assumes discrete integer event trials (word draws),
   Bag-of-Words count vectors are the mathematically appropriate and theoretically
   consistent input representation.

2. TF-IDF (Term Frequency-Inverse Document Frequency):
   Multiplies term frequencies by log(N / df_w) to penalize frequent words across the
   corpus. While TF-IDF normalizes document lengths and highlights discriminative terms
   (beneficial for SVMs and Logistic Regression), continuous weighting violates the pure
   multinomial count assumption. Thus, standard BoW is preferred for Multinomial Naive Bayes.
"""

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import scipy.sparse as sp

from preprocessing import TextPreprocessor


class BagOfWordsVectorizer:
    """Extracts Bag-of-Words count features into efficient sparse matrices.

    Parameters
    ----------
    min_df : int, default=2
        Minimum document frequency. Tokens appearing in fewer documents are ignored.
    max_features : Optional[int], default=None
        Maximum vocabulary size. If specified, only top `max_features` ordered
        by total corpus term frequency are retained.
    ngram_range : Tuple[int, int], default=(1, 1)
        Lower and upper boundary of range of n-values for different n-grams.
        e.g., (1, 1) means only unigrams; (1, 2) means unigrams and bigrams.
    preprocessor : Optional[TextPreprocessor], default=None
        Instance of TextPreprocessor. If None, default TextPreprocessor is used.
    """

    def __init__(
        self,
        min_df: int = 2,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.preprocessor = preprocessor if preprocessor is not None else TextPreprocessor()

        # Learned attributes
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _extract_ngrams(self, tokens: Sequence[str]) -> List[str]:
        """Extracts n-grams for a sequence of tokens according to ngram_range.

        Parameters
        ----------
        tokens : Sequence[str]
            List of unigram tokens.

        Returns
        -------
        List[str]
            List of extracted n-grams.
        """
        min_n, max_n = self.ngram_range
        if min_n == 1 and max_n == 1:
            return list(tokens)

        extracted: List[str] = []
        n_tokens = len(tokens)

        for n in range(min_n, max_n + 1):
            if n == 1:
                extracted.extend(tokens)
            else:
                for i in range(n_tokens - n + 1):
                    extracted.append(" ".join(tokens[i : i + n]))

        return extracted

    def fit(self, raw_documents: Iterable[Optional[str]]) -> "BagOfWordsVectorizer":
        """Builds vocabulary from training documents ONLY.

        Parameters
        ----------
        raw_documents : Iterable[Optional[str]]
            Iterable of raw text documents.

        Returns
        -------
        self : BagOfWordsVectorizer
        """
        doc_freq: Counter = Counter()
        term_freq: Counter = Counter()

        for doc in raw_documents:
            tokens = self.preprocessor.tokenize(doc)
            ngrams = self._extract_ngrams(tokens)
            term_freq.update(ngrams)
            doc_freq.update(set(ngrams))

        # Filter by min_df
        candidates = [term for term, df in doc_freq.items() if df >= self.min_df]

        # Rank candidates by total term frequency descending
        candidates.sort(key=lambda t: (term_freq[t], t), reverse=True)

        # Apply max_features cut-off if specified
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates = candidates[: self.max_features]

        # Build vocabulary dictionary and feature names list sorted alphabetically for consistency
        candidates.sort()
        self.vocabulary_ = {term: idx for idx, term in enumerate(candidates)}
        self.feature_names_ = candidates
        self.is_fitted_ = True

        return self

    def transform(self, raw_documents: Iterable[Optional[str]]) -> sp.csr_matrix:
        """Transforms documents into a sparse Bag-of-Words count matrix.

        Parameters
        ----------
        raw_documents : Iterable[Optional[str]]
            Iterable of raw text documents.

        Returns
        -------
        sp.csr_matrix
            Sparse matrix of token counts of shape (n_documents, vocab_size).
        """
        if not self.is_fitted_:
            raise RuntimeError("BagOfWordsVectorizer has not been fitted yet. Call fit() first.")

        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []

        vocab = self.vocabulary_
        vocab_size = len(vocab)

        n_docs = 0
        for doc_idx, doc in enumerate(raw_documents):
            n_docs += 1
            tokens = self.preprocessor.tokenize(doc)
            ngrams = self._extract_ngrams(tokens)
            counts = Counter(ngrams)

            for term, count in counts.items():
                col_idx = vocab.get(term)
                if col_idx is not None:
                    rows.append(doc_idx)
                    cols.append(col_idx)
                    data.append(count)

        if vocab_size == 0:
            return sp.csr_matrix((n_docs, 0), dtype=np.int32)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, vocab_size),
            dtype=np.int32,
        )

    def fit_transform(self, raw_documents: Iterable[Optional[str]]) -> sp.csr_matrix:
        """Fits vocabulary and returns transformed sparse count matrix.

        Parameters
        ----------
        raw_documents : Iterable[Optional[str]]
            Iterable of raw text documents.

        Returns
        -------
        sp.csr_matrix
            Sparse matrix of token counts.
        """
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names(self) -> List[str]:
        """Returns the list of feature names (vocabulary terms)."""
        return self.feature_names_

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocabulary_)

