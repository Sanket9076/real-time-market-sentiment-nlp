"""
Feature Engineering Module

Extracts analytical temporal, text, and numerical sentiment features
for business analytics, statistical modeling, and time-series aggregation.
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature engineering pipeline for financial news and sentiment data.
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered analytical features.
        
        Args:
            df (pd.DataFrame): DataFrame containing news text, timestamps, and sentiment scores.
            
        Returns:
            pd.DataFrame: Feature-enriched DataFrame.
        """
        if df.empty:
            return df

        df = df.copy()

        # 1. Text Metrics
        df["text_length"] = df["title"].astype(str).str.len()
        df["word_count"] = df["title"].astype(str).apply(lambda x: len(x.split()))
        df["avg_word_len"] = df["title"].astype(str).apply(
            lambda x: round(np.mean([len(w) for w in x.split()]), 2) if x.split() else 0.0
        )

        # 2. Temporal Features
        if "timestamp" in df.columns:
            ts_dt = pd.to_datetime(df["timestamp"], errors="coerce")
            df["article_date"] = ts_dt.dt.strftime("%Y-%m-%d")
            df["article_hour"] = ts_dt.dt.hour
            df["article_day"] = ts_dt.dt.day
            df["day_of_week"] = ts_dt.dt.day_name()
            df["is_weekend"] = ts_dt.dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)

        # 3. Sentiment Engineered Metrics
        if "positive_score" in df.columns and "negative_score" in df.columns:
            pos = df["positive_score"].fillna(0.0)
            neg = df["negative_score"].fillna(0.0)
            df["polarity_ratio"] = round((pos - neg) / (pos + neg + 1e-5), 4)

        # 4. Rolling Sentiment Velocity (Asset-level time-series velocity)
        if "compound_score" in df.columns and "asset" in df.columns and "timestamp" in df.columns:
            df = df.sort_values(by=["asset", "timestamp"]).reset_index(drop=True)
            df["sentiment_3period_sma"] = df.groupby("asset")["compound_score"].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            ).round(4)
            df["sentiment_velocity"] = df.groupby("asset")["sentiment_3period_sma"].diff().fillna(0.0).round(4)

        print(f"[FeatureEngineer] Successfully engineered {len(df.columns)} feature columns.")
        return df
