"""Text preprocessing pipeline for AG News text classification.

Handles text normalization, tokenization, punctuation removal, stopword filtering,
and robust handling of empty/missing text inputs.
"""

import re
from typing import List, Optional, Set, Union


# Standard comprehensive English stopwords set (self-contained, no external download required)
ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves"
}


class TextPreprocessor:
    """Configurable text preprocessor for news article classification.

    Parameters
    ----------
    lowercase : bool, default=True
        Whether to convert text to lowercase.
    remove_punctuation : bool, default=True
        Whether to strip punctuation characters.
    remove_stopwords : bool, default=True
        Whether to filter out common English stopwords.
    custom_stopwords : Optional[Set[str]], default=None
        Custom set of stopwords. If None and remove_stopwords is True,
        standard English stopwords are used.
    min_token_length : int, default=1
        Minimum character length for a token to be retained.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[Set[str]] = None,
        min_token_length: int = 1,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        if custom_stopwords is not None:
            self.stopwords = set(custom_stopwords)
        elif self.remove_stopwords:
            self.stopwords = ENGLISH_STOPWORDS
        else:
            self.stopwords = set()

        # Regular expression for word tokenization (\b\w+\b)
        self._word_regex = re.compile(r"\b[a-zA-Z0-9_]+\b")
        # Regular expression to remove HTML tags if present (e.g. &lt; / &gt;)
        self._html_tag_regex = re.compile(r"<[^>]+>")

    def clean_text(self, text: Optional[str]) -> str:
        """Sanitizes raw text, handling empty, None, or malformed inputs.

        Parameters
        ----------
        text : Optional[str]
            Raw input text.

        Returns
        -------
        str
            Sanitized text string.
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        # Remove HTML tags if present
        text = self._html_tag_regex.sub(" ", text)

        # Convert to lowercase if configured
        if self.lowercase:
            text = text.lower()

        return text

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Tokenizes text into a list of word tokens with cleaning and filtering.

        Parameters
        ----------
        text : Optional[str]
            Raw input text.

        Returns
        -------
        List[str]
            List of processed tokens.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        # Find word tokens
        tokens = self._word_regex.findall(cleaned)

        # Filter stopwords and min length
        processed_tokens: List[str] = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            processed_tokens.append(token)

        return processed_tokens

    def preprocess_documents(self, documents: List[Optional[str]]) -> List[List[str]]:
        """Preprocesses a collection of raw text documents.

        Parameters
        ----------
        documents : List[Optional[str]]
            List of input document strings.

        Returns
        -------
        List[List[str]]
            List of tokenized documents.
        """
        return [self.tokenize(doc) for doc in documents]

