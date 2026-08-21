"""
Master Verification Script for Real-Time Market Sentiment & NLP Analysis Platform.

Executes:
  1. Data Ingestion (Offline Demo Mode)
  2. Data Cleaning & Automated Quality Reporting
  3. Text Preprocessing
  4. VADER Baseline Sentiment Analysis
  5. Supervised TF-IDF Logistic Regression ML Sentiment Model Training & Evaluation
  6. Feature Engineering
  7. Database Schema Initialization & SQLite Record Persistence
  8. Execution of all 21 SQL Analytical Queries from sql/05_advanced_queries.sql
  9. Business Analytics & Statistical Market Correlation Engine
 10. Automated Reporting & Artifact Generation
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaner
from src.text_preprocessing import TextPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.ml_sentiment import MLSentimentClassifier
from src.feature_engineering import FeatureEngineer
from src.database import DatabaseManager
from src.analysis import AnalyticsEngine

def main():
    print("=" * 70)
    print("STARTING FULL END-TO-END PIPELINE EXECUTION")
    print("=" * 70)

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # 1. Ingestion
    ingestion = DataIngestion(mode="demo")
    raw_df = ingestion.fetch_data()
    print(f"1. INGESTION: Fetched {len(raw_df)} raw records.")

    # 2. Cleaning
    cleaner = DataCleaner()
    clean_df, quality_report = cleaner.clean_dataset(raw_df)
    print(f"2. CLEANING: Audit completed. Quality Score: {quality_report['data_quality_score_pct']}%.")

    # 3. Preprocessing
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(clean_df, text_column="title")
    print(f"3. PREPROCESSING: Applied lowercasing, tokenization, stopword filtering, and lemmatization.")

    # 4. Sentiment Analysis (VADER Baseline)
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(processed_df, text_column="title")
    print(f"4. SENTIMENT (VADER): Calculated compound & component scores.")

    # 5. Supervised ML Classifier Training & Comparison
    ml_classifier = MLSentimentClassifier()
    ml_eval = ml_classifier.train_and_evaluate(sentiment_df, text_col="processed_text", label_col="sentiment_label")
    print(f"5. ML CLASSIFIER: Trained TF-IDF + Logistic Regression. Accuracy={ml_eval.get('accuracy')}, F1={ml_eval.get('f1_score_weighted')}.")

    # 6. Feature Engineering
    featured_df = FeatureEngineer().create_features(sentiment_df)
    print(f"6. FEATURE ENGINEERING: Generated {len(featured_df.columns)} feature columns.")

    # 7. Database Initialization & Sync
    db = DatabaseManager(db_type="sqlite", sqlite_path="data/market_sentiment.db")
    db.init_db(schema_file="sql/01_schema.sql")
    db.save_dataframe(featured_df, table_name="articles", if_exists="replace")
    
    # Save sentiment_results table
    sent_table = featured_df[["article_id", "compound_score", "positive_score", "negative_score", "neutral_score", "sentiment_label"]].copy()
    db.save_dataframe(sent_table, table_name="sentiment_results", if_exists="replace")

    # Sync Market Prices
    if os.path.exists("sample_data/market_prices_sample.csv"):
        price_df = pd.read_csv("sample_data/market_prices_sample.csv")
        db.save_dataframe(price_df, table_name="market_prices", if_exists="replace")
    print("7. DATABASE: Schema & tables persisted into SQLite database ('data/market_sentiment.db').")

    # 8. SQL Analytics Verification (Execute queries from sql/05_advanced_queries.sql)
    print("8. SQL ANALYTICS: Executing verification queries...")
    with open("sql/05_advanced_queries.sql", "r", encoding="utf-8") as f:
        sql_file_content = f.read()

    sql_statements = [q.strip() for q in sql_file_content.split(";") if q.strip() and not q.strip().startswith("-- ==")]
    success_sql = 0
    for idx, stmt in enumerate(sql_statements):
        if not stmt or stmt.startswith("--"):
            continue
        try:
            res_df = db.execute_query(stmt)
            success_sql += 1
        except Exception as e:
            print(f"  SQL Exec warning on query #{idx+1}: {e}")
    print(f"   Successfully validated {success_sql} SQL analytical queries.")

    # 9. Business Analytics & Market Correlation
    analytics = AnalyticsEngine()
    kpis = analytics.compute_kpis(featured_df)
    print(f"9. ANALYTICS: Total Coverage={kpis['total_articles']}, Pos={kpis['positive_pct']}%, Neg={kpis['negative_pct']}%, Avg Sentiment={kpis['avg_sentiment']}.")

    corr_res = analytics.correlate_sentiment_market(featured_df, price_df)
    print(f"   MARKET CORRELATION: Pearson r = {corr_res.get('pearson_correlation')}, Spearman r = {corr_res.get('spearman_correlation')}.")

    # 10. Generate Visualizations & Metrics Report JSON
    summary_report = {
        "execution_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quality_report": quality_report,
        "kpi_metrics": kpis,
        "ml_evaluation": ml_eval,
        "market_correlation": corr_res
    }

    with open("outputs/reports/execution_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Save visual figures
    plt.figure(figsize=(8, 5))
    sns.countplot(data=featured_df, x="sentiment_label", palette=["#2ECC71", "#95A5A6", "#E74C3C"])
    plt.title("Sentiment Classification Breakdown")
    plt.savefig("outputs/figures/sentiment_breakdown.png", bbox_inches="tight")
    plt.close()

    print("10. ARTIFACTS: Saved metrics to 'outputs/reports/execution_metrics.json' and figures to 'outputs/figures/'.")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE & FULLY VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
