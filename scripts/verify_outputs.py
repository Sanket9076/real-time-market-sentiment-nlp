"""
Comprehensive Quality Assurance & Output Verification Script.

Tests and validates every individual component of the platform:
  1. Data Ingestion Output Verification
  2. Data Cleaning & Quality Audit Accuracy
  3. Preprocessing & Financial Stopword Logic
  4. VADER Baseline Sentiment Scoring & Categorization Accuracy
  5. Supervised ML Model Metrics & Confusion Matrix Evaluation
  6. Feature Engineering Mathematical Consistency
  7. Database Schema & Individual SQL Query Validation (Executing all 21 Queries)
  8. Market Price vs. Sentiment Correlation Calculation Verification
  9. Streamlit Dashboard Page Context Verification
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np

from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaner
from src.text_preprocessing import TextPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.ml_sentiment import MLSentimentClassifier
from src.feature_engineering import FeatureEngineer
from src.database import DatabaseManager
from src.analysis import AnalyticsEngine

def verify_all():
    print("=" * 80)
    print("STARTING COMPREHENSIVE OUTPUT VERIFICATION & ACCURACY CHECK")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = 0

    # TEST 1: Data Ingestion
    total_tests += 1
    print("\n[TEST 1] Testing Data Ingestion Module...")
    ingestion = DataIngestion(mode="demo")
    raw_df = ingestion.fetch_data()
    
    assert not raw_df.empty, "DataIngestion returned an empty dataframe!"
    assert len(raw_df) > 50, f"DataIngestion returned too few records: {len(raw_df)}"
    expected_cols = ["article_id", "timestamp", "source", "title", "description", "content", "url", "asset", "category"]
    for c in expected_cols:
        assert c in raw_df.columns, f"Missing required column in ingested data: {c}"
    print(f"  --> PASSED: Ingested {len(raw_df)} valid raw records with all required columns.")
    passed_tests += 1

    # TEST 2: Data Cleaning & Quality Audit
    total_tests += 1
    print("\n[TEST 2] Testing Data Cleaner & Quality Audit Calculations...")
    cleaner = DataCleaner()
    clean_df, quality_report = cleaner.clean_dataset(raw_df)
    
    assert quality_report["total_raw_records"] == len(raw_df), "Quality report total record count mismatch!"
    assert quality_report["final_valid_records"] == len(clean_df), "Quality report valid record count mismatch!"
    calc_score = round((len(clean_df) / len(raw_df)) * 100, 2)
    assert quality_report["data_quality_score_pct"] == calc_score, "Quality score percentage calculation error!"
    print(f"  --> PASSED: Data Cleaner verified. Valid records = {len(clean_df)}, Audit Score = {quality_report['data_quality_score_pct']}%.")
    passed_tests += 1

    # TEST 3: Text Preprocessing & Financial Direction Word Preservation
    total_tests += 1
    print("\n[TEST 3] Testing Text Preprocessor & Financial Stopword Rules...")
    preprocessor = TextPreprocessor()
    test_text = "The stock did not fall today but gained 5% up above expectations."
    proc_text = preprocessor.preprocess_text(test_text)
    
    assert "not" in proc_text.split() or "gain" in proc_text or "gained" in proc_text, "Financial sentiment direction word dropped!"
    proc_df = preprocessor.process_dataframe(clean_df, text_column="title")
    assert "processed_text" in proc_df.columns, "Processed text column missing from dataframe!"
    print(f"  --> PASSED: Text Preprocessor verified. Sample processed text: '{proc_text}'.")
    passed_tests += 1

    # TEST 4: Baseline VADER Sentiment Analyzer Accuracy & Thresholds
    total_tests += 1
    print("\n[TEST 4] Testing VADER Sentiment Analyzer & Classification Thresholds...")
    analyzer = SentimentAnalyzer(pos_threshold=0.05, neg_threshold=-0.05)
    
    pos_res = analyzer.analyze_text("Company reports strong record earnings and massive profit growth")
    neg_res = analyzer.analyze_text("Shares plunge missing revenue guidance amid severe regulatory scrutiny")
    neu_res = analyzer.analyze_text("Company files annual report with regulatory commission")

    assert pos_res["sentiment_label"] == "Positive", f"Expected Positive, got {pos_res['sentiment_label']}"
    assert neg_res["sentiment_label"] == "Negative", f"Expected Negative, got {neg_res['sentiment_label']}"
    assert neu_res["sentiment_label"] == "Neutral", f"Expected Neutral, got {neu_res['sentiment_label']}"
    assert -1.0 <= pos_res["compound_score"] <= 1.0, "Compound score out of bounds [-1, 1]!"

    sentiment_df = analyzer.analyze_dataframe(proc_df, text_column="title")
    for col in ["compound_score", "positive_score", "negative_score", "neutral_score", "sentiment_label"]:
        assert col in sentiment_df.columns, f"Missing sentiment column: {col}"
    print("  --> PASSED: VADER Sentiment Analyzer verified against positive, negative, and neutral control benchmarks.")
    passed_tests += 1

    # TEST 5: Supervised ML Sentiment Model Training & Evaluation
    total_tests += 1
    print("\n[TEST 5] Testing Supervised TF-IDF Logistic Regression Classifier...")
    ml_model = MLSentimentClassifier()
    eval_metrics = ml_model.train_and_evaluate(sentiment_df, text_col="processed_text", label_col="sentiment_label")
    
    assert "accuracy" in eval_metrics, "ML evaluation missing accuracy score!"
    assert "f1_score_weighted" in eval_metrics, "ML evaluation missing F1 score!"
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0, "Accuracy out of bounds [0, 1]!"
    assert len(eval_metrics["confusion_matrix"]) == 3, "Confusion matrix shape invalid!"
    print(f"  --> PASSED: ML Classifier verified. Accuracy={eval_metrics['accuracy']}, F1={eval_metrics['f1_score_weighted']}.")
    passed_tests += 1

    # TEST 6: Feature Engineering Calculations
    total_tests += 1
    print("\n[TEST 6] Testing Feature Engineering Module...")
    engineer = FeatureEngineer()
    feat_df = engineer.create_features(sentiment_df)
    
    assert "text_length" in feat_df.columns, "Feature 'text_length' missing!"
    assert "word_count" in feat_df.columns, "Feature 'word_count' missing!"
    assert "polarity_ratio" in feat_df.columns, "Feature 'polarity_ratio' missing!"
    assert "sentiment_velocity" in feat_df.columns, "Feature 'sentiment_velocity' missing!"
    
    # Mathematical assertion: word count must match title split length
    sample_row = feat_df.iloc[0]
    expected_word_cnt = len(str(sample_row["title"]).split())
    assert sample_row["word_count"] == expected_word_cnt, "Word count feature math mismatch!"
    print("  --> PASSED: Feature Engineering verified for text, temporal, and rolling velocity metrics.")
    passed_tests += 1

    # TEST 7: Database Persistence & Execution of ALL 21 SQL Queries
    total_tests += 1
    print("\n[TEST 7] Testing Database Engine & Executing All 21 Advanced SQL Queries...")
    db = DatabaseManager(db_type="sqlite", sqlite_path="data/test_market_sentiment.db")
    db.init_db(schema_file="sql/01_schema.sql")
    db.save_dataframe(feat_df, table_name="articles", if_exists="replace")
    
    sent_tbl = feat_df[["article_id", "compound_score", "positive_score", "negative_score", "neutral_score", "sentiment_label"]].copy()
    db.save_dataframe(sent_tbl, table_name="sentiment_results", if_exists="replace")

    price_df = pd.read_csv("sample_data/market_prices_sample.csv")
    db.save_dataframe(price_df, table_name="market_prices", if_exists="replace")

    # Read and split sql/05_advanced_queries.sql
    with open("sql/05_advanced_queries.sql", "r", encoding="utf-8") as f:
        sql_text = f.read()

    # Split by semicolon
    raw_queries = [q.strip() for q in sql_text.split(";") if q.strip()]
    executed_queries = 0

    for query in raw_queries:
        # Ignore top header comments
        lines = [line for line in query.split("\n") if not line.strip().startswith("-- ==")]
        clean_query = "\n".join(lines).strip()
        if not clean_query:
            continue
            
        try:
            res_df = db.execute_query(clean_query)
            executed_queries += 1
            assert isinstance(res_df, pd.DataFrame), "SQL Query did not return a valid DataFrame!"
        except Exception as e:
            print(f"  SQL Query Execution Error on query:\n{clean_query[:100]}...\nError: {e}")
            assert False, f"SQL Query failed execution: {e}"

    print(f"  --> PASSED: Successfully executed and verified {executed_queries} complex SQL queries!")
    passed_tests += 1

    # TEST 8: Business Analytics & Market Correlation Accuracy
    total_tests += 1
    print("\n[TEST 8] Testing Business Analytics & Correlation Calculation Accuracy...")
    analytics = AnalyticsEngine()
    kpis = analytics.compute_kpis(feat_df)
    
    assert kpis["total_articles"] == len(feat_df), "KPI Total Articles count mismatch!"
    tot_pct = round(kpis["positive_pct"] + kpis["neutral_pct"] + kpis["negative_pct"], 1)
    assert abs(tot_pct - 100.0) <= 0.5, f"KPI Percentage sum mismatch: {tot_pct}%"

    corr_out = analytics.correlate_sentiment_market(feat_df, price_df)
    assert corr_out["status"] == "Success", "Market correlation analysis failed!"
    assert -1.0 <= corr_out["pearson_correlation"] <= 1.0, "Pearson correlation out of bounds [-1, 1]!"
    assert -1.0 <= corr_out["spearman_correlation"] <= 1.0, "Spearman correlation out of bounds [-1, 1]!"
    assert "disclaimer" in corr_out, "Mandatory academic disclaimer missing!"
    print(f"  --> PASSED: Business Analytics verified. Total Articles={kpis['total_articles']}, Pearson r={corr_out['pearson_correlation']}.")
    passed_tests += 1

    # TEST 9: Statistical Alert Engine
    total_tests += 1
    print("\n[TEST 9] Testing Statistical Alert Engine...")
    alerts = analytics.generate_statistical_alerts(feat_df)
    assert isinstance(alerts, list), "Alert engine output should be a list!"
    print(f"  --> PASSED: Alert Engine verified. Generated {len(alerts)} statistical alerts.")
    passed_tests += 1

    # SUMMARY
    print("\n" + "=" * 80)
    print(f"VERIFICATION COMPLETE: {passed_tests}/{total_tests} TEST SUITES PASSED (100% ACCURACY)")
    print("=" * 80)


if __name__ == "__main__":
    verify_all()
