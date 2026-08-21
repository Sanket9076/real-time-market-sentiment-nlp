"""
Text Preprocessing Module

Performs domain-aware text normalization for NLP: lowercasing, tokenization,
stopword filtering (preserving critical financial negation and direction words),
punctuation removal, and lemmatization.
"""

import re
import string
from typing import List, Union
import pandas as pd

# Fallback basic stopword list in case NLTK data is downloading
FINANCIAL_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
    'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
    'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now'
}

# Preserve financial sentiment signals
PRESERVED_WORDS = {'not', 'no', 'up', 'down', 'gain', 'loss', 'bull', 'bear', 'rise', 'fall', 'above', 'below', 'high', 'low'}


class TextPreprocessor:
    """
    NLP Preprocessing Engine for Financial Text.
    """

    def __init__(self, use_lemmatizer: bool = True):
        self.use_lemmatizer = use_lemmatizer
        self.lemmatizer = None
        self.stopwords = FINANCIAL_STOPWORDS - PRESERVED_WORDS
        
        # Try initializing NLTK components gracefully
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            try:
                nltk_stops = set(stopwords.words('english'))
                self.stopwords = nltk_stops - PRESERVED_WORDS
            except Exception:
                pass

            if self.use_lemmatizer:
                try:
                    self.lemmatizer = WordNetLemmatizer()
                except Exception:
                    self.use_lemmatizer = False
        except Exception:
            self.use_lemmatizer = False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string and return normalized text string.
        
        Preprocessing steps:
        1. Lowercasing: Ensures case-insensitivity.
        2. Special character & digit handling: Retains alphabetical words and key tickers.
        3. Tokenization: Splitting into word tokens.
        4. Stopword Filtering: Removes low-information words while preserving financial direction & negation.
        5. Lemmatization: Reduces words to base morphological root.
        """
        if not text or pd.isna(text):
            return ""

        # Lowercase
        text_lower = str(text).lower()

        # Remove punctuation & special characters (keep spaces)
        text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text_lower)

        # Tokenize
        tokens = text_clean.split()

        # Filter stopwords
        filtered_tokens = [tok for tok in tokens if tok not in self.stopwords and len(tok) > 1]

        # Lemmatize
        if self.use_lemmatizer and self.lemmatizer:
            try:
                filtered_tokens = [self.lemmatizer.lemmatize(tok) for tok in filtered_tokens]
            except Exception:
                pass

        return " ".join(filtered_tokens)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        """
        Apply text preprocessing to DataFrame, adding 'processed_text' column
        while keeping original text untouched.
        """
        if df.empty or text_column not in df.columns:
            return df

        df = df.copy()
        df["processed_text"] = df[text_column].apply(self.preprocess_text)
        return df
