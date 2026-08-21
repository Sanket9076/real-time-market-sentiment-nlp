# Real-Time Market Sentiment & NLP Analysis Platform

An end-to-end institutional-grade financial analytics platform that ingests financial news and public market text, performs natural language processing (NLP) and sentiment scoring, persists structured data into a relational database (PostgreSQL with SQLite local engine fallback), executes 21 complex SQL analytical queries, and presents real-time business insights via an interactive 5-page Streamlit dashboard.

---

## 🎯 Executive Summary & Real Metrics

This project demonstrates production-grade Data Analyst, Data Engineer, and Python/SQL analytics skills. Every metric listed below is produced directly by actual execution of the underlying data pipeline (`outputs/reports/execution_metrics.json`):

| Analytical Metric | Empirical Value |
| :--- | :--- |
| **Total Ingested Articles** | `149 Valid Financial Articles` |
| **Data Quality Audit Score** | `100.0%` (0 invalid records, 71 duplicates dropped) |
| **Positive Sentiment Share** | `32.9%` |
| **Neutral Sentiment Share** | `40.9%` |
| **Negative Sentiment Share** | `26.2%` |
| **Portfolio Mean Sentiment** | `+0.0416` (Mildly Bullish) |
| **Most Mentioned Asset** | `AMZN` |
| **Most Positive Asset** | `MSFT` (Mean Compound: `+0.2185`) |
| **Most Negative Asset** | `NVDA` (Mean Compound: `-0.0543`) |
| **Supervised ML Classifier F1-Score** | `1.0000` (TF-IDF + Logistic Regression) |
| **Price vs. Sentiment Alignment** | `99 Sample Trading Days` |
| **Daily Return vs. Sentiment Pearson $r$** | `-0.0403` (Spearman $r = -0.0747$) |

---

## 📐 Architecture Diagram

```
+-------------------------------------------------------------------------------+
|                               DATA INGESTION                                  |
|         Live News APIs (NewsAPI / Finnhub) OR Offline Sample Dataset          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                            DATA CLEANING & AUDIT                              |
|    HTML Strip | Unicode Fix | Headline Deduplication | Quality Audit Score    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                             NLP PREPROCESSING                                 |
|  Lowercasing | Tokenization | Financial Stopword Filter | Lemmatization      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                     SENTIMENT ENGINE & MODEL COMPARISON                       |
|   VADER Rule-Based Baseline  <--- VS --->  Supervised TF-IDF Logistic Regression|
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                            FEATURE ENGINEERING                                |
|   Text Metrics | Temporal Features | Polarity Ratios | Sentiment Velocity     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        RELATIONAL DATABASE STORAGE                            |
|       PostgreSQL (Primary) <--- Auto-Fallback Engine ---> SQLite Local DB      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      ADVANCED SQL ANALYTICS SUITE                             |
|    21 Queries | CTEs | Window Functions (AVG OVER, LAG, RANK) | Subqueries    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                        INTERACTIVE STREAMLIT DASHBOARD                        |
|   Executive KPIs | Sentiment Deep-Dive | NLP Keywords | Market Correlation     |
+-------------------------------------------------------------------------------+
```

---

## 🛠️ Technology Stack

- **Core Programming**: Python 3.10+
- **Data Wrangling & Manipulation**: Pandas, NumPy
- **Natural Language Processing (NLP)**: NLTK, VADER Sentiment, Scikit-learn (`TfidfVectorizer`)
- **Machine Learning**: Logistic Regression
- **Relational Databases & SQL**: PostgreSQL, SQLite, SQLAlchemy, Psycopg2, ANSI SQL
- **Interactive Dashboard**: Streamlit, Plotly Express, Plotly Subplots
- **Statistical Visualization**: Matplotlib, Seaborn

---

## 📁 Project Directory Structure

```
real-time-market-sentiment-nlp/
│
├── data/
│   ├── raw/                           # Raw ingested CSV files
│   ├── processed/                     # Cleaned & featured CSV files
│   └── market_sentiment.db            # SQLite local database instance
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis notebook
│   ├── 02_nlp_analysis.ipynb          # NLP & ML model evaluation notebook
│   └── 03_market_sentiment_analysis.ipynb # Market correlation notebook
│
├── sql/
│   ├── 01_schema.sql                  # PostgreSQL & SQLite DDL table schemas
│   ├── 02_data_quality.sql            # Data quality validation & audit queries
│   ├── 03_sentiment_analysis.sql      # Sentiment aggregation & source queries
│   ├── 04_market_analysis.sql         # Market return alignment queries
│   └── 05_advanced_queries.sql        # 21 Advanced SQL queries (CTEs, Window Functions)
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_ingestion.py              # Live API fetch & sample loader
│   ├── data_cleaning.py               # Data quality audit & string sanitizer
│   ├── text_preprocessing.py          # Tokenizer & financial stopword filter
│   ├── sentiment_analysis.py          # VADER baseline sentiment analyzer
│   ├── ml_sentiment.py                # Supervised TF-IDF Logistic Regression classifier
│   ├── feature_engineering.py         # Time-series & text feature extractor
│   ├── database.py                    # PostgreSQL / SQLite dual DB engine manager
│   └── analysis.py                    # Business intelligence & correlation engine
│
├── dashboard/
│   └── app.py                         # 5-Page Streamlit interactive application
│
├── outputs/
│   ├── figures/                       # Generated PNG charts & visualizations
│   └── reports/                       # Execution metrics JSON reports
│
├── sample_data/
│   ├── financial_news_sample.csv      # Curated offline news dataset (220 raw records)
│   └── market_prices_sample.csv       # Daily OHLCV stock price history (440 records)
│
├── scripts/
│   ├── generate_sample_data.py        # Sample dataset generator script
│   ├── build_notebooks.py             # Jupyter notebook generator script
│   └── run_full_pipeline.py           # Master end-to-end pipeline execution runner
│
├── .env.example                       # Environment template
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Dependency manifest
├── INTERVIEW_QUESTIONS.md             # 42 Interview Technical Questions & Answers
├── PROJECT_EXPLANATION.md             # Pitch scripts & interviewer defense guide
├── app.py                             # Root Streamlit app entry point
└── README.md                          # Project documentation
```

---

## 🗄️ Database Schema & Advanced SQL Analytics

The database architecture features 5 normalized tables:
- `articles`: Primary metadata (`article_id`, `timestamp`, `source_name`, `title`, `description`, `content`, `url`, `asset`, `category`)
- `sentiment_results`: NLP metrics (`sentiment_id`, `article_id`, `compound_score`, `positive_score`, `negative_score`, `neutral_score`, `sentiment_label`)
- `market_prices`: Daily stock metrics (`price_id`, `asset`, `trade_date`, `open_price`, `high_price`, `low_price`, `close_price`, `volume`, `daily_return`)
- `assets` & `sources`: Dimension tables

### Key SQL Highlights (`sql/05_advanced_queries.sql`):
1. **7-Day Rolling Sentiment Moving Average** using `AVG() OVER`:
   ```sql
   AVG(daily_sentiment) OVER (
       PARTITION BY asset 
       ORDER BY trade_date 
       ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
   ) AS rolling_7day_sentiment
   ```
2. **Day-over-Day Sentiment Velocity** using `LAG()`:
   ```sql
   avg_sentiment - LAG(avg_sentiment, 1) OVER (PARTITION BY asset ORDER BY trade_date) AS sentiment_velocity
   ```
3. **Dynamic Asset Sentiment Ranking** using `DENSE_RANK()`:
   ```sql
   DENSE_RANK() OVER (ORDER BY AVG(s.compound_score) DESC) AS sentiment_rank
   ```

---

## 🖥️ Interactive Streamlit Dashboard

The dashboard provides a 5-page analytical experience:
1. **Executive Overview**: Portfolio KPIs, Sentiment Donut Chart, Asset Sentiment Benchmarks, Statistical Anomaly Alerts.
2. **Sentiment Deep-Dive**: Multi-asset time-series line charts, news outlet sentiment share, score dispersion boxplots.
3. **NLP Insights**: Positive vs Negative TF-IDF Keyword frequency bar charts, classified headline feed.
4. **Market Comparison**: Dual-axis stock price vs daily sentiment trend, return scatter plot, Pearson/Spearman metrics.
5. **Article Explorer**: Filterable table with keyword search and CSV export.

---

## ⚙️ How to Install and Run

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/real-time-market-sentiment-nlp.git
cd real-time-market-sentiment-nlp

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optional: Add NewsAPI key to `.env` for live mode. Offline demo mode works out of the box!)*

### 3. Run Full Verification Pipeline
Execute the end-to-end data pipeline to clean data, run NLP models, seed SQLite database, and compute execution metrics:
```bash
python scripts/run_full_pipeline.py
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically at `http://localhost:8501`.

---

## 📝 ATS-Friendly Resume Bullets (Empirical Metrics)

- **Engineered an end-to-end Real-Time Market Sentiment & NLP Platform** in Python and SQL, ingesting and cleaning 149 financial news articles across 8 market tickers with a verified 100.0% data quality score.
- **Implemented dual VADER and supervised TF-IDF Logistic Regression sentiment engines**, achieving 1.0000 F1-score classification performance and benchmarking positive (32.9%) vs negative (26.2%) news coverage.
- **Designed a 5-table relational database (PostgreSQL/SQLite) and executed 21 advanced SQL queries** utilizing CTEs, `AVG() OVER` rolling window functions, `LAG()` velocity metrics, and `DENSE_RANK()` to analyze sentiment dynamics across 99 market alignment days.

---

## ⚠️ Academic & Analytical Disclaimer

**IMPORTANT**: Correlation does not imply causation. Sentiment analysis is an observational analytical tool designed for news coverage monitoring. The platform does **NOT** predict stock prices and does **NOT** provide financial advice or investment recommendations.

---

## 📄 Dataset & API Attribution

- News data: [NewsAPI](https://newsapi.org/) and curated offline financial datasets.
- Market Price Data: Simulated OHLCV financial market daily series based on historical volatility.
