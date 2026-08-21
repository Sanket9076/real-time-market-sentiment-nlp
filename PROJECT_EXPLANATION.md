# Project Explanation & Interview Defense Guide
## Real-Time Market Sentiment & NLP Analysis Platform

This document provides multi-duration pitch scripts and technical defense strategies for explaining this project during data analyst and data science interviews.

---

## 1. Project Pitches by Duration

### ⚡ 30-Second Pitch
> "I built an end-to-end Real-Time Market Sentiment & NLP Analysis Platform that ingests financial news data, cleans and preprocesses the text, runs sentiment analysis using VADER and supervised machine learning models, stores structured records in a PostgreSQL database with an SQLite fallback, and presents real-time business metrics through an interactive 5-page Streamlit dashboard."

---

### ⏱️ 60-Second Pitch
> "Financial markets produce massive amounts of unstructured news text every day. I built a modular Python and SQL platform to quantify market sentiment and evaluate its alignment with stock price movements. 
> 
> The pipeline ingests data via REST APIs or offline datasets, applies NLTK text preprocessing, scores compound sentiment using VADER, and trains a supervised TF-IDF Logistic Regression model for comparison. The processed records are stored in a relational database with 5 normalized tables. I wrote over 20 advanced SQL queries utilizing window functions, CTEs, and CASE statements to analyze sentiment velocity and asset rankings. Finally, I deployed an interactive Streamlit dashboard featuring statistical anomaly alerts, NLP keyword analysis, and dual-axis stock price vs. sentiment charts."

---

### ⏳ 2-Minute Pitch
> "As a Data Analyst, I wanted to create a platform that turns raw, unstructured text into decision-ready financial intelligence.
> 
> Here is how the platform is architected:
> First, **Ingestion & Data Quality**: The `DataIngestion` module fetches live news via REST APIs or loads offline sample data. The `DataCleaner` removes HTML, normalizes unicode, deduplicates headlines, and automatically outputs a Data Quality Report audit score.
> 
> Second, **NLP & Sentiment Modeling**: Text is preprocessed by lowercasing, tokenizing, filtering financial stopwords, and lemmatizing. I implemented a baseline VADER sentiment model to calculate compound scores and categorical labels (Positive, Neutral, Negative). For rigorous evaluation, I built a supervised TF-IDF Logistic Regression classifier to benchmark model performance.
> 
> Third, **Database & SQL Analytics**: Cleaned records and features are persisted into PostgreSQL (with a zero-config local SQLite fallback). I wrote 21 SQL queries featuring 7-day rolling window averages (`AVG() OVER`), day-over-day sentiment velocity (`LAG()`), and dense ranking (`DENSE_RANK()`).
> 
> Fourth, **Streamlit Dashboard & Insights**: The 5-page dashboard presents executive KPIs, sentiment deep-dives, positive/negative TF-IDF keyword charts, market price vs sentiment correlation scatter plots, and statistical anomaly alerts. Importantly, the platform emphasizes that correlation does not imply causation and avoids false claims of stock price prediction."

---

### 🔬 5-Minute Detailed Deep-Dive Pitch
*(Use when the interviewer asks: "Walk me through your project from scratch to finish.")*

1. **Business Problem & Objectives**: Financial analysts face information overload. The objective was to build an automated pipeline that ingests financial text, measures sentiment dynamics across companies over time, identifies key sentiment drivers, and correlates sentiment with price trends without making illegal financial advice claims.
2. **Data Pipeline Architecture**: Ingestion (`src/data_ingestion.py`) handles HTTP rate-limiting, error codes (e.g. 429), and offline fallback. Cleaning (`src/data_cleaning.py`) strips HTML, imputes missing values, filters invalid records (<10 chars), and outputs an automated audit report.
3. **NLP Processing**: Text is tokenized and filtered using domain-aware stopwords that preserve direction words ('up', 'down', 'gain', 'loss', 'not'). Lemmatization reduces words to root form.
4. **Sentiment & Supervised ML**: VADER evaluates sentiment intensity. In parallel, a TF-IDF Logistic Regression model is trained on an 80/20 split, yielding accuracy, precision, recall, and confusion matrix metrics.
5. **Database & SQL Suite**: Database manager dynamically selects PostgreSQL or SQLite. Schema contains 5 tables (`articles`, `sentiment_results`, `market_prices`, `assets`, `sources`). A 21-query SQL suite executes CTEs, window functions (`LAG`, `AVG OVER`), and joins.
6. **Streamlit UI**: Built with custom CSS glassmorphism styling, `@st.cache_data` optimization, manual refresh triggers, statistical alert banners, and dual-axis Plotly charts.

---

## 2. Interview Defense & Technical QA

### Q1: "Why did you choose VADER for sentiment analysis?"
**Defense:** VADER is specifically tuned for microblog and headline text. It handles capitalization, punctuation emphasis, and degree modifiers without requiring massive GPU infrastructure. Additionally, I implemented a supervised TF-IDF Logistic Regression model alongside VADER to compare rule-based vs machine learning performance.

### Q2: "Why did you support both PostgreSQL and SQLite?"
**Defense:** PostgreSQL represents industry-standard enterprise database architecture for production applications. However, requiring an interviewer or recruiter to configure a local Postgres database creates setup friction. Implementing an auto-detecting SQLite fallback guarantees zero-setup execution while preserving standard ANSI SQL compatibility.

### Q3: "Why did you choose Streamlit over Dash or Tableau?"
**Defense:** Streamlit allows full-stack Python data science applications to be built rapidly without context switching between design tools and codebases. It seamlessly integrates with Pandas, Plotly, and scikit-learn.

### Q4: "How did you handle API failures and rate limits?"
**Defense:** The ingestion pipeline checks HTTP status codes explicitly (e.g., status 429). If rate limits are reached or API keys are missing, the pipeline logs a warning and automatically falls back to the curated offline sample dataset, preventing application crashes.

### Q5: "How did you handle duplicate articles?"
**Defense:** Case-insensitive headline normalization (`df['title'].str.lower().str.strip()`) is applied during ingestion and cleaning. `.drop_duplicates(subset=['title'])` removes duplicate wire stories before NLP processing.

### Q6: "How did you calculate sentiment scores?"
**Defense:** VADER calculates a normalized compound score between -1.0 and +1.0 based on lexical valence scores. Scores $\ge +0.05$ are labeled Positive, $\le -0.05$ Negative, and remaining values Neutral.

### Q7: "How do you know your sentiment model is reliable?"
**Defense:** Reliability was evaluated by comparing VADER labels against a supervised TF-IDF Logistic Regression model trained on labeled financial data, achieving verified accuracy, precision, recall, and F1 metrics logged in `outputs/reports/execution_metrics.json`.

### Q8: "Does sentiment predict stock prices?"
**Defense:** No, and the project explicitly disclaims stock price prediction or financial advice. Sentiment is an observational analytical metric that measures market narrative, not a guaranteed leading price indicator.

### Q9: "Why doesn't correlation prove causation?"
**Defense:** Correlation simply measures co-movement between daily sentiment scores and daily stock returns. Both variables can be driven by a third confounding variable, such as interest rate decisions, macroeconomic inflation data, or broader market rallies.

### Q10: "What would you improve in Version 2?"
**Defense:** 
1. Incorporate Named Entity Recognition (NER) with spaCy to handle articles mentioning multiple companies.
2. Train a fine-tuned FinBERT transformer model on financial domain corpora.
3. Deploy streaming ingestion using Apache Kafka and Airflow DAG orchestration.
