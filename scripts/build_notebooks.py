"""
Script to programmatically generate executable Jupyter Notebooks (.ipynb)
for EDA, NLP Analysis, and Market Sentiment Correlation.
"""

import os
import json


def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {filename}")


# 1. 01_eda.ipynb
eda_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Exploratory Data Analysis (EDA)\n",
            "## Real-Time Market Sentiment & NLP Analysis Platform\n",
            "\n",
            "This notebook performs comprehensive EDA on the ingested financial news dataset:\n",
            "- Data Quality & Missing Value Auditing\n",
            "- Sentiment Distribution & Categorical Counts\n",
            "- Publisher/Source Coverage Analysis\n",
            "- Article Volume Trends Over Time"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from src.data_ingestion import DataIngestion\n",
            "from src.data_cleaning import DataCleaner\n",
            "from src.sentiment_analysis import SentimentAnalyzer\n",
            "from src.analysis import AnalyticsEngine\n",
            "\n",
            "# Set aesthetic style\n",
            "sns.set_theme(style='darkgrid')\n",
            "plt.rcParams['figure.figsize'] = (10, 6)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Ingest Data\n",
            "ingestion = DataIngestion(mode='demo')\n",
            "raw_df = ingestion.fetch_data()\n",
            "\n",
            "# 2. Clean Data\n",
            "cleaner = DataCleaner()\n",
            "clean_df, quality_report = cleaner.clean_dataset(raw_df)\n",
            "print('Data Quality Report:')\n",
            "print(quality_report)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Run Sentiment Pipeline\n",
            "analyzer = SentimentAnalyzer()\n",
            "df = analyzer.analyze_dataframe(clean_df, text_column='title')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Sentiment Label Breakdown\n",
            "sent_counts = df['sentiment_label'].value_counts()\n",
            "plt.figure(figsize=(7, 5))\n",
            "sns.barplot(x=sent_counts.index, y=sent_counts.values, palette=['#2ECC71', '#95A5A6', '#E74C3C'])\n",
            "plt.title('Global Sentiment Distribution Count')\n",
            "plt.ylabel('Article Count')\n",
            "plt.xlabel('Sentiment Class')\n",
            "plt.savefig('../outputs/figures/eda_sentiment_distribution.png', bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Asset Level Coverage & Sentiment Boxplot\n",
            "plt.figure(figsize=(12, 6))\n",
            "sns.boxplot(data=df, x='asset', y='compound_score', palette='Set2')\n",
            "plt.axhline(0, color='red', linestyle='--', alpha=0.5)\n",
            "plt.title('Compound Score Dispersion per Financial Asset')\n",
            "plt.savefig('../outputs/figures/eda_asset_sentiment_dispersion.png', bbox_inches='tight')\n",
            "plt.show()"
        ]
    }
]

# 2. 02_nlp_analysis.ipynb
nlp_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02. Natural Language Processing & Sentiment Modeling\n",
            "## Real-Time Market Sentiment & NLP Analysis Platform\n",
            "\n",
            "This notebook evaluates text preprocessing, TF-IDF n-gram extraction, baseline VADER sentiment scoring, and Supervised ML Classifier training."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from src.data_ingestion import DataIngestion\n",
            "from src.text_preprocessing import TextPreprocessor\n",
            "from src.sentiment_analysis import SentimentAnalyzer\n",
            "from src.ml_sentiment import MLSentimentClassifier\n",
            "from src.analysis import AnalyticsEngine"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ingest & Preprocess\n",
            "raw_df = DataIngestion(mode='demo').fetch_data()\n",
            "preprocessor = TextPreprocessor()\n",
            "df = preprocessor.process_dataframe(raw_df, text_column='title')\n",
            "df = SentimentAnalyzer().analyze_dataframe(df, text_column='title')\n",
            "df[['title', 'processed_text', 'sentiment_label', 'compound_score']].head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Top Positive & Negative Keywords Extraction\n",
            "analytics = AnalyticsEngine()\n",
            "keywords = analytics.extract_keywords_by_sentiment(df, top_k=10)\n",
            "print('Top Positive Keywords:', keywords['positive'])\n",
            "print('Top Negative Keywords:', keywords['negative'])"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Supervised ML Classifier (TF-IDF + Logistic Regression)\n",
            "ml_classifier = MLSentimentClassifier()\n",
            "eval_metrics = ml_classifier.train_and_evaluate(df, text_col='processed_text', label_col='sentiment_label')\n",
            "print('ML Model Evaluation Metrics:')\n",
            "for k, v in eval_metrics.items():\n",
            "    print(f' - {k}: {v}')"
        ]
    }
]

# 3. 03_market_sentiment_analysis.ipynb
market_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03. Market Price vs. News Sentiment Correlation\n",
            "## Real-Time Market Sentiment & NLP Analysis Platform\n",
            "\n",
            "Statistical analysis aligning news sentiment compound scores with daily asset price returns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from src.data_ingestion import DataIngestion\n",
            "from src.sentiment_analysis import SentimentAnalyzer\n",
            "from src.analysis import AnalyticsEngine"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load news & prices\n",
            "news_df = SentimentAnalyzer().analyze_dataframe(DataIngestion(mode='demo').fetch_data())\n",
            "price_df = pd.read_csv('../sample_data/market_prices_sample.csv')\n",
            "\n",
            "# Compute Statistical Correlation\n",
            "analytics = AnalyticsEngine()\n",
            "corr_results = analytics.correlate_sentiment_market(news_df, price_df)\n",
            "print('Correlation Analysis Results:')\n",
            "print(f'Pearson r: {corr_results.get(\"pearson_correlation\")}')\n",
            "print(f'Spearman r: {corr_results.get(\"spearman_correlation\")}')\n",
            "print(f'Disclaimer: {corr_results.get(\"disclaimer\")}')"
        ]
    }
]

create_notebook("notebooks/01_eda.ipynb", eda_cells)
create_notebook("notebooks/02_nlp_analysis.ipynb", nlp_cells)
create_notebook("notebooks/03_market_sentiment_analysis.ipynb", market_cells)
