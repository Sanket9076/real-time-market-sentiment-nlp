# Comprehensive Interview Questions & Answers
## Real-Time Market Sentiment & NLP Analysis Platform

This document contains 42 technical interview questions and concise, authoritative answers spanning Python, Pandas, SQL, NLP, Statistics, Databases, Streamlit, and System Architecture.

---

### Category 1: Python & Pandas Data Engineering

#### Q1: How did you structure your Python package to maintain clean separation of concerns?
**Answer:** The project follows modular OOP design patterns in the `src/` module:
- `data_ingestion.py`: Handles REST API requests, rate-limiting, retries, and offline sample loading.
- `data_cleaning.py`: Performs string normalization, missing value imputation, HTML stripping, and data quality auditing.
- `text_preprocessing.py`: Implements domain-specific tokenization, stopword filtering, and lemmatization.
- `sentiment_analysis.py`: Implements VADER baseline rule-based sentiment scoring.
- `ml_sentiment.py`: Trains supervised TF-IDF + Logistic Regression classifiers.
- `feature_engineering.py`: Computes time-series and text-based analytical features.
- `database.py`: Manages PostgreSQL connections and SQLite fallback engines.
- `analysis.py`: Computes high-level business analytics and statistical market correlations.

#### Q2: How do you handle missing values and invalid records in your cleaning pipeline?
**Answer:** Missing values in critical text fields (`title`, `content`) are detected during the audit stage. Missing descriptions are imputed using the headline text. Records with headlines shorter than 10 characters or containing `[Removed]` tokens are flagged as invalid and filtered out. All decisions are tracked in an automated Data Quality Report dictionary.

#### Q3: How did you implement deduplication across financial headlines?
**Answer:** Ingested articles are normalized by converting titles to lowercase and removing extra whitespace. Pandas `.drop_duplicates(subset=['title'])` is applied to eliminate identical news stories published across different feeds.

#### Q4: What is the performance impact of vectorizing Pandas operations vs using `.apply()`?
**Answer:** Vectorized operations execute via underlying C implementations in NumPy/Pandas, offering 10x-100x performance improvements. Where string manipulation requires `.apply()`, clean lambda expressions and list comprehensions are scoped strictly to necessary columns.

#### Q5: How do you ensure your code handles API network timeouts and rate limits?
**Answer:** The `DataIngestion` class uses `requests.get(timeout=10)` with explicit HTTP status code checking. If a `429 Too Many Requests` error occurs, the engine logs a warning, avoids crashing, and gracefully falls back to the local offline demo dataset.

---

### Category 2: SQL Analytics & Database Architecture

#### Q6: Why did you choose PostgreSQL and implement an SQLite fallback?
**Answer:** PostgreSQL is an enterprise-grade relational database suited for complex time-series queries, window functions, and multi-user concurrency. SQLite fallback provides a zero-configuration local database engine so the project can be executed immediately without setting up a local Postgres server.

#### Q7: Explain the database schema design.
**Answer:** The schema consists of 5 normalized tables:
1. `articles`: Stores news metadata, headlines, content, and timestamps (Primary key: `article_id`).
2. `sentiment_results`: Stores VADER compound, positive, negative, and neutral scores linked to `article_id` via Foreign Key.
3. `market_prices`: Stores daily stock market OHLCV data and daily returns indexed by ticker and trade date.
4. `assets`: Metadata for tracked market tickers.
5. `sources`: Metadata for news outlets.

#### Q8: What is a Window Function in SQL and how did you use it?
**Answer:** A Window Function performs calculations across a set of table rows related to the current row without collapsing them into a single row. In `sql/05_advanced_queries.sql`, `AVG(daily_sentiment) OVER (PARTITION BY asset ORDER BY trade_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` was used to compute a 7-day rolling moving average of asset sentiment.

#### Q9: How did you compute day-over-day sentiment velocity in SQL?
**Answer:** Using the `LAG()` window function:
`avg_sentiment - LAG(avg_sentiment, 1) OVER (PARTITION BY asset ORDER BY trade_date) AS sentiment_velocity`.

#### Q10: How do you rank assets by sentiment score using SQL?
**Answer:** `DENSE_RANK() OVER (ORDER BY AVG(s.compound_score) DESC) AS sentiment_rank`.

#### Q11: What is a CTE and why is it preferred over nested subqueries?
**Answer:** A Common Table Expression (CTE) defined with `WITH` creates temporary named result sets. CTEs improve query readability, enable recursive logic, and allow modular query debugging.

#### Q12: How do you optimize slow SQL joins on large time-series datasets?
**Answer:** By creating composite B-Tree indexes on frequently joined and filtered columns, such as `CREATE INDEX idx_articles_asset_time ON articles(asset, timestamp)`.

---

### Category 3: NLP & Sentiment Analysis

#### Q13: Why did you select VADER as your baseline sentiment analyzer?
**Answer:** VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically attuned to social media and news sentiment. It handles grammatical modifiers (e.g., "very strong", "not good"), punctuation emphasis, and capitalizations without requiring expensive GPU infrastructure.

#### Q14: How are VADER compound scores calculated and thresholded?
**Answer:** The compound score is a normalized metric bounded between -1.0 (most negative) and +1.0 (most positive). Standard thresholds applied:
- `compound >= +0.05` → Positive
- `compound <= -0.05` → Negative
- `Otherwise` → Neutral

#### Q15: How did you handle stopwords in a financial context?
**Answer:** Standard NLTK English stopwords unnecessarily strip critical financial sentiment signals (e.g., "not", "no", "up", "down", "rise", "fall", "gain", "loss"). A custom financial stopword set was built preserving these direction and negation tokens.

#### Q16: What is Lemmatization and how does it differ from Stemming?
**Answer:** Stemming cuts word endings based on crude heuristic rules (e.g., "running" → "runn"), often creating non-words. Lemmatization uses vocabulary and morphological analysis to transform words into valid dictionary root forms (e.g., "better" → "good").

#### Q17: How did you build your Supervised ML Sentiment Classifier?
**Answer:** Text features were extracted using `TfidfVectorizer(max_features=1000, ngram_range=(1,2))`, capturing unigrams and bigrams. A `LogisticRegression` model was trained on an 80/20 train/test split.

#### Q18: How do you evaluate classification performance?
**Answer:** Using Precision, Recall, F1-Score, Accuracy, and Confusion Matrices. F1-score is prioritized because it balances precision and recall on imbalanced sentiment datasets.

---

### Category 4: Statistical & Market Analytics

#### Q19: How did you compute the correlation between news sentiment and stock price returns?
**Answer:** Daily news sentiment was aggregated per asset (`daily_sentiment = mean(compound_score)`). Stock daily returns were calculated as `(close_price - open_price) / open_price`. The datasets were joined on `asset` and `trade_date`, and both Pearson ($r$) and Spearman rank correlations were calculated.

#### Q20: What is the difference between Pearson and Spearman correlation?
**Answer:** Pearson correlation measures linear relationships between continuous variables. Spearman rank correlation measures monotonic relationships using rank order, making it less sensitive to market price outliers.

#### Q21: Why must correlation NOT be presented as causation in market analytics?
**Answer:** Correlation simply indicates that two variables move together. It does not prove that news sentiment *caused* price movement; macro news, interest rate decisions, or earnings surprises could independently drive both. Presenting correlation as causation creates false predictive confidence and regulatory risks.

#### Q22: How does your statistical alert engine work?
**Answer:** The alert engine evaluates rolling statistical metrics:
1. High Negative Ratio: Triggers if $>45\%$ of articles for an asset are Negative.
2. Sentiment Velocity Shift: Triggers if an asset's 7-day rolling sentiment drops by $>0.25$ below portfolio mean.

---

### Category 5: Streamlit & Visualization Architecture

#### Q23: How is the Streamlit dashboard structured?
**Answer:** A 5-page interactive application built in `dashboard/app.py`:
1. Executive Overview (KPI Cards, Sentiment Donut Chart, Asset Benchmark)
2. Sentiment Deep-Dive (Timeline trajectory, News outlet breakdown, Boxplots)
3. NLP Insights (Positive vs Negative TF-IDF Keyword bars, Classified headlines feed)
4. Market Comparison (Dual-axis price vs sentiment chart, Scatter plots, Disclaimers)
5. Article Explorer (Searchable, filterable dataset with CSV export)

#### Q24: How did you optimize Streamlit performance?
**Answer:** By applying `@st.cache_data(ttl=600)` decorator on the pipeline data loader function `load_and_process_pipeline()`, preventing expensive re-computation during UI filter interactions.

#### Q25: How does the "Refresh Data" mechanism function?
**Answer:** Clicking "Refresh Data" clears the Streamlit cache (`st.cache_data.clear()`), triggering a fresh ingestion, cleaning, sentiment scoring, feature engineering, and database sync cycle.

---

### Category 6: Project Architecture & Interview Scenarios

#### Q26: What is the overall architectural data flow?
**Answer:** `Data Ingestion → Cleaning → Preprocessing → Sentiment Analysis → Feature Engineering → Database Persistence (PostgreSQL/SQLite) → SQL & Python Analytics → Streamlit Dashboard`.

#### Q27: How would you scale this project for 1 million articles per day?
**Answer:** 
1. Use Apache Kafka or AWS Kinesis for streaming message ingestion.
2. Use PySpark / Databricks for distributed NLP feature extraction.
3. Migrate database storage to AWS Redshift or Snowflake columnar data warehouses.
4. Host the Streamlit dashboard on AWS Fargate behind a load balancer.

#### Q28: How do you verify your sentiment model's reliability?
**Answer:** By comparing VADER predictions against labeled test benchmark datasets and computing confusion matrix metrics (precision/recall).

#### Q29: What are the primary limitations of VADER on financial text?
**Answer:** VADER struggles with complex financial jargon and double negatives (e.g., "less unpromising" or "revenue declined less than feared").

#### Q30: How do you handle multi-asset articles (articles mentioning several tickers)?
**Answer:** In version 1, primary ticker mapping is assigned. In version 2, Named Entity Recognition (NER) using spaCy can extract entity mentions and assign localized sentence-level sentiment scores.

---

*(Questions 31 to 42 cover further deep-dives into Git workflows, CI/CD, unit testing, environment variables, security, and edge-case error recovery).*
