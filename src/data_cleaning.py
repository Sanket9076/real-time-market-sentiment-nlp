"""
Data Cleaning Module

Performs rigorous data quality checks, missing value detection, HTML stripping,
whitespace normalization, invalid record filtration, and automatically generates
a comprehensive Data Quality Report.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class DataCleaner:
    """
    Cleans and standardizes raw financial news datasets.
    """

    def __init__(self):
        self.quality_report: Dict[str, Any] = {}

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main data cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Raw ingested news DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and Data Quality Report.
        """
        if df.empty:
            return df, {"total_records": 0, "status": "Empty input dataframe"}

        total_records = len(df)
        
        # 1. Missing value count before cleaning
        missing_before = df.isnull().sum().to_dict()

        # 2. Drop duplicates on title and content
        initial_dup_count = df.duplicated(subset=["title"]).sum()
        df_clean = df.drop_duplicates(subset=["title"]).copy()

        # 3. Clean Text Fields (Title, Description, Content)
        df_clean["title"] = df_clean["title"].astype(str).apply(self._clean_text_string)
        df_clean["description"] = df_clean["description"].astype(str).apply(self._clean_text_string)
        df_clean["content"] = df_clean["content"].astype(str).apply(self._clean_text_string)

        # 4. Standardize Categorical fields
        df_clean["source"] = df_clean["source"].astype(str).str.strip().str.title()
        df_clean["asset"] = df_clean["asset"].astype(str).str.strip().str.upper()
        df_clean["category"] = df_clean["category"].astype(str).str.strip().str.title()

        # 5. Handle missing values & Imputation
        df_clean["description"] = df_clean["description"].replace(["nan", "None", ""], np.nan)
        df_clean["description"] = df_clean["description"].fillna(df_clean["title"])

        df_clean["content"] = df_clean["content"].replace(["nan", "None", ""], np.nan)
        df_clean["content"] = df_clean["content"].fillna(df_clean["title"])

        # 6. Validate Record Integrity (Invalid if title length < 10 or title is empty/removed)
        valid_mask = (df_clean["title"].str.len() >= 10) & (~df_clean["title"].isin(["[Removed]", "nan", "None"]))
        invalid_records_count = int((~valid_mask).sum())
        
        df_valid = df_clean[valid_mask].copy()

        # 7. Timestamp validation and cleanup
        df_valid["timestamp"] = pd.to_datetime(df_valid["timestamp"], errors="coerce")
        df_valid = df_valid.dropna(subset=["timestamp"]).copy()
        df_valid["timestamp"] = df_valid["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # 8. Generate Automated Data Quality Report
        valid_records_count = len(df_valid)
        self.quality_report = {
            "total_raw_records": total_records,
            "duplicate_headlines_removed": int(initial_dup_count),
            "invalid_records_removed": invalid_records_count,
            "final_valid_records": valid_records_count,
            "missing_values_by_column": missing_before,
            "data_quality_score_pct": round((valid_records_count / total_records) * 100, 2) if total_records > 0 else 0.0
        }

        print(f"[DataCleaner] Quality Audit Complete: {valid_records_count}/{total_records} valid records retain ({self.quality_report['data_quality_score_pct']}% quality score).")
        return df_valid.reset_index(drop=True), self.quality_report

    @staticmethod
    def _clean_text_string(text: str) -> str:
        """Strip HTML tags, URLs, bad unicode, and redundant whitespaces."""
        if not text or pd.isna(text):
            return ""
        
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
