"""
Analytics Engine Module

High-level business intelligence, keyword extraction, correlation analytics,
and statistical alert engine.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class AnalyticsEngine:
    """
    Business Analytics and Statistical Insights Engine.
    """

    def compute_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute executive summary Key Performance Indicators (KPIs).
        """
        if df.empty:
            return {
                "total_articles": 0,
                "positive_pct": 0.0,
                "neutral_pct": 0.0,
                "negative_pct": 0.0,
                "avg_sentiment": 0.0,
                "most_mentioned_asset": "N/A",
                "most_positive_asset": "N/A",
                "most_negative_asset": "N/A"
            }

        total = len(df)
        counts = df["sentiment_label"].value_counts().to_dict()
        
        pos_pct = round((counts.get("Positive", 0) / total) * 100, 1)
        neu_pct = round((counts.get("Neutral", 0) / total) * 100, 1)
        neg_pct = round((counts.get("Negative", 0) / total) * 100, 1)
        
        avg_comp = round(float(df["compound_score"].mean()), 4)

        # Asset level aggregations
        asset_summary = df.groupby("asset").agg(
            article_count=("article_id", "count"),
            mean_sentiment=("compound_score", "mean")
        ).reset_index()

        most_mentioned = asset_summary.sort_values(by="article_count", ascending=False).iloc[0]["asset"] if not asset_summary.empty else "N/A"
        most_positive = asset_summary.sort_values(by="mean_sentiment", ascending=False).iloc[0]["asset"] if not asset_summary.empty else "N/A"
        most_negative = asset_summary.sort_values(by="mean_sentiment", ascending=True).iloc[0]["asset"] if not asset_summary.empty else "N/A"

        return {
            "total_articles": total,
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct,
            "avg_sentiment": avg_comp,
            "most_mentioned_asset": most_mentioned,
            "most_positive_asset": most_positive,
            "most_negative_asset": most_negative
        }

    def extract_keywords_by_sentiment(self, df: pd.DataFrame, text_col: str = "processed_text", top_k: int = 15) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract top keywords/n-grams separated by Positive vs Negative sentiment.
        """
        if df.empty or text_col not in df.columns:
            return {"positive": [], "negative": [], "all": []}

        pos_text = " ".join(df[df["sentiment_label"] == "Positive"][text_col].dropna())
        neg_text = " ".join(df[df["sentiment_label"] == "Negative"][text_col].dropna())
        all_text = " ".join(df[text_col].dropna())

        def _get_top_freq(text: str) -> List[Tuple[str, int]]:
            if not text.strip():
                return []
            vec = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_k)
            try:
                bag = vec.fit_transform([text])
                sum_words = bag.sum(axis=0)
                words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
                return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
            except Exception:
                return []

        return {
            "positive": _get_top_freq(pos_text),
            "negative": _get_top_freq(neg_text),
            "all": _get_top_freq(all_text)
        }

    def correlate_sentiment_market(self, news_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistical correlation between daily asset sentiment and daily stock price returns.
        Includes statistical significance (Pearson r) and disclaimer.
        """
        if news_df.empty or price_df.empty:
            return {"status": "Insufficient data"}

        # Daily sentiment aggregation per asset
        news_df["article_date"] = pd.to_datetime(news_df["timestamp"]).dt.strftime("%Y-%m-%d")
        daily_sent = news_df.groupby(["asset", "article_date"]).agg(
            daily_sentiment=("compound_score", "mean"),
            article_volume=("article_id", "count")
        ).reset_index()

        # Merge with daily market price returns
        price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.strftime("%Y-%m-%d")
        merged = pd.merge(
            daily_sent, price_df,
            left_on=["asset", "article_date"],
            right_on=["asset", "trade_date"],
            how="inner"
        )

        if len(merged) < 5:
            return {"status": "Insufficient overlap between news timestamps and trading days", "merged_count": len(merged)}

        pearson_r = round(float(merged["daily_sentiment"].corr(merged["daily_return"])), 4)
        spearman_r = round(float(merged["daily_sentiment"].corr(merged["daily_return"], method="spearman")), 4)

        return {
            "status": "Success",
            "sample_size": len(merged),
            "pearson_correlation": pearson_r,
            "spearman_correlation": spearman_r,
            "disclaimer": "IMPORTANT: Correlation does not imply causation. Sentiment analysis is an observational analytical tool and must NOT be used as investment advice or stock price prediction.",
            "data": merged[["asset", "trade_date", "daily_sentiment", "daily_return", "close_price", "article_volume"]].to_dict(orient="records")
        }

    def generate_statistical_alerts(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Generate automated statistical alerts based on anomaly thresholds.
        """
        alerts = []
        if df.empty or "compound_score" not in df.columns or "asset" not in df.columns:
            return alerts

        # Check asset-level sentiment shifts
        asset_stats = df.groupby("asset").agg(
            mean_sentiment=("compound_score", "mean"),
            article_count=("article_id", "count"),
            neg_ratio=("sentiment_label", lambda x: (x == "Negative").sum() / len(x))
        ).reset_index()

        overall_mean = df["compound_score"].mean()

        for _, row in asset_stats.iterrows():
            asset = row["asset"]
            sent = row["mean_sentiment"]
            neg_r = row["neg_ratio"]
            count = row["article_count"]

            if neg_r >= 0.45 and count >= 5:
                alerts.append({
                    "severity": "HIGH",
                    "title": f"High Negative Sentiment Alert: {asset}",
                    "message": f"Asset {asset} has an elevated negative sentiment ratio of {neg_r*100:.1f}% across {count} articles."
                })
            elif sent <= overall_mean - 0.25:
                alerts.append({
                    "severity": "MEDIUM",
                    "title": f"Sentiment Downtrend: {asset}",
                    "message": f"Asset {asset} average sentiment ({sent:.3f}) is significantly below portfolio benchmark ({overall_mean:.3f})."
                })
            elif sent >= overall_mean + 0.25:
                alerts.append({
                    "severity": "INFO",
                    "title": f"Strong Positive Sentiment: {asset}",
                    "message": f"Asset {asset} is experiencing strong positive news coverage with mean score {sent:.3f}."
                })

        return alerts
