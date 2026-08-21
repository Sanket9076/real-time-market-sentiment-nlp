"""
Baseline Sentiment Analysis Module

Implements VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer
for financial headlines and text. Provides compound, positive, negative, and neutral scores
along with explicit threshold-based categorical labeling.
"""

from typing import Dict, Any, Tuple
import pandas as pd

# Financial Lexicon Boosters for Fallback Scorer
FALLBACK_POS_WORDS = {
    'record', 'surge', 'soar', 'beat', 'growth', 'gain', 'strong', 'profit', 'upgrade',
    'bullish', 'outperform', 'revenue', 'raise', 'expand', 'cheer', 'breakthrough',
    'contract', 'acquisition', 'high', 'dividend', 'buyback', 'climb', 'rally'
}

FALLBACK_NEG_WORDS = {
    'slide', 'miss', 'disappoint', 'decline', 'drop', 'fall', 'loss', 'warning', 'downgrade',
    'bearish', 'scrutiny', 'fine', 'delay', 'disrupt', 'lawsuit', 'cut', 'slump', 'plunge',
    'bottleneck', 'dispute', 'plummets', 'headwind', 'plunge', 'recession'
}


class SentimentAnalyzer:
    """
    Rule-based Sentiment Analyzer using VADER with custom fallback capabilities.
    """

    def __init__(self, pos_threshold: float = 0.05, neg_threshold: float = -0.05):
        """
        Initialize Sentiment Analyzer.
        
        Args:
            pos_threshold (float): Compound score threshold for Positive sentiment (Default >= 0.05).
            neg_threshold (float): Compound score threshold for Negative sentiment (Default <= -0.05).
        """
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.vader_analyzer = None

        # Attempt initializing VADER analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except ImportError:
            try:
                import nltk
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except Exception:
                print("[SentimentAnalyzer] VADER library not found. Operating with financial lexicon fallback engine.")
                self.vader_analyzer = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text string.
        
        Returns:
            Dict containing:
                - compound_score
                - positive_score
                - negative_score
                - neutral_score
                - sentiment_label
        """
        if not text or pd.isna(text):
            return {
                "compound_score": 0.0,
                "positive_score": 0.0,
                "negative_score": 0.0,
                "neutral_score": 1.0,
                "sentiment_label": "Neutral"
            }

        if self.vader_analyzer:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = round(float(scores.get("compound", 0.0)), 4)
            pos = round(float(scores.get("pos", 0.0)), 4)
            neg = round(float(scores.get("neg", 0.0)), 4)
            neu = round(float(scores.get("neu", 1.0)), 4)
        else:
            compound, pos, neg, neu = self._lexicon_fallback_scores(text)

        label = self.classify_compound(compound)

        return {
            "compound_score": compound,
            "positive_score": pos,
            "negative_score": neg,
            "neutral_score": neu,
            "sentiment_label": label
        }

    def classify_compound(self, compound: float) -> str:
        """Categorize compound score based on strict threshold configuration."""
        if compound >= self.pos_threshold:
            return "Positive"
        elif compound <= self.neg_threshold:
            return "Negative"
        else:
            return "Neutral"

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "title") -> pd.DataFrame:
        """
        Apply sentiment analysis across a DataFrame. Adds sentiment score & label columns.
        """
        if df.empty or text_column not in df.columns:
            return df

        df = df.copy()
        results = df[text_column].apply(self.analyze_text)
        
        results_df = pd.DataFrame(list(results))
        for col in ["compound_score", "positive_score", "negative_score", "neutral_score", "sentiment_label"]:
            df[col] = results_df[col]

        return df

    @staticmethod
    def _lexicon_fallback_scores(text: str) -> Tuple[float, float, float, float]:
        """Fallback financial lexicon scorer when VADER package is unavailable."""
        words = text.lower().split()
        if not words:
            return 0.0, 0.0, 0.0, 1.0

        pos_count = sum(1 for w in words if w in FALLBACK_POS_WORDS)
        neg_count = sum(1 for w in words if w in FALLBACK_NEG_WORDS)
        total = len(words)

        if pos_count > neg_count:
            compound = round(min(1.0, 0.2 + (pos_count - neg_count) * 0.25), 4)
        elif neg_count > pos_count:
            compound = round(max(-1.0, -0.2 - (neg_count - pos_count) * 0.25), 4)
        else:
            compound = 0.0

        pos_score = round(pos_count / total, 4)
        neg_score = round(neg_count / total, 4)
        neu_score = round(max(0.0, 1.0 - (pos_score + neg_score)), 4)

        return compound, pos_score, neg_score, neu_score
