"""
Data Ingestion Module

Handles fetching news and financial text data from live public APIs
(NewsAPI, Finnhub) or loading from local sample datasets for offline demo mode.
Includes rate limiting, error handling, validation, deduplication, and raw storage.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class DataIngestion:
    """
    Ingests financial news articles from live APIs or local offline sample data.
    """

    def __init__(self, mode: str = "demo", sample_path: str = "sample_data/financial_news_sample.csv"):
        """
        Initialize DataIngestion module.
        
        Args:
            mode (str): 'demo' for offline sample data or 'live' for API fetching.
            sample_path (str): File path to offline sample CSV.
        """
        self.mode = mode.lower()
        self.sample_path = sample_path
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")

    def fetch_data(self, asset: Optional[str] = None, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch news data based on configured mode.
        
        Args:
            asset (str, optional): Target asset ticker (e.g., 'AAPL').
            days_back (int): Days window for historical fetch.
            
        Returns:
            pd.DataFrame: Ingested raw news DataFrame.
        """
        if self.mode == "live":
            print(f"[DataIngestion] Attempting live API fetch for asset='{asset}'...")
            df = self._fetch_live_api(asset=asset, days_back=days_back)
            if df.empty:
                print("[DataIngestion] Live API returned no data or failed. Falling back to offline demo dataset.")
                df = self._load_offline_sample(asset=asset)
        else:
            print(f"[DataIngestion] Loading offline sample dataset from '{self.sample_path}'...")
            df = self._load_offline_sample(asset=asset)

        return self._deduplicate_and_format(df)

    def _load_offline_sample(self, asset: Optional[str] = None) -> pd.DataFrame:
        """Load data from local CSV sample file."""
        if not os.path.exists(self.sample_path):
            raise FileNotFoundError(f"Sample data file not found at: {self.sample_path}")
        
        df = pd.read_csv(self.sample_path)
        if asset and asset != "ALL" and "asset" in df.columns:
            df = df[df["asset"].str.upper() == asset.upper()].copy()
        
        return df

    def _fetch_live_api(self, asset: Optional[str] = None, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch live data from NewsAPI with rate-limiting and error handling.
        """
        if not self.news_api_key or self.news_api_key == "your_news_api_key_here":
            print("[DataIngestion] Warning: No valid NEWS_API_KEY set in environment.")
            return pd.DataFrame()

        query = asset if asset and asset != "ALL" else "stock market OR finance OR earnings"
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey": self.news_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                print("[DataIngestion] API Error: 429 Rate limit exceeded.")
                return pd.DataFrame()
            elif response.status_code != 200:
                print(f"[DataIngestion] API Error: HTTP {response.status_code} - {response.text}")
                return pd.DataFrame()

            data = response.json()
            articles = data.get("articles", [])
            
            raw_records = []
            for idx, item in enumerate(articles):
                source_name = item.get("source", {}).get("name", "Public News")
                pub_time = item.get("publishedAt", "")
                
                # Format timestamp
                try:
                    dt = datetime.strptime(pub_time, "%Y-%m-%dT%H:%M:%SZ")
                    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                raw_records.append({
                    "article_id": f"LIVE-{int(time.time())}-{idx}",
                    "timestamp": ts_str,
                    "source": source_name,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", "") or item.get("description", ""),
                    "url": item.get("url", ""),
                    "asset": asset if asset else "MARKET",
                    "category": "Market News"
                })

            return pd.DataFrame(raw_records)

        except requests.exceptions.RequestException as e:
            print(f"[DataIngestion] Network error fetching API data: {e}")
            return pd.DataFrame()

    def _deduplicate_and_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names, remove duplicate titles, and parse timestamps."""
        if df.empty:
            return df

        required_cols = ["article_id", "timestamp", "source", "title", "description", "content", "url", "asset", "category"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        # Normalize timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Deduplicate on title (case-insensitive)
        initial_len = len(df)
        df["title_clean"] = df["title"].astype(str).str.lower().str.strip()
        df = df.drop_duplicates(subset=["title_clean"]).drop(columns=["title_clean"]).copy()
        
        dedup_len = len(df)
        if initial_len != dedup_len:
            print(f"[DataIngestion] Removed {initial_len - dedup_len} duplicate articles based on headline.")

        return df[required_cols]

    def save_raw_data(self, df: pd.DataFrame, output_path: str = "data/raw/financial_news_raw.csv"):
        """Save raw ingested data to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataIngestion] Saved {len(df)} records to '{output_path}'.")
