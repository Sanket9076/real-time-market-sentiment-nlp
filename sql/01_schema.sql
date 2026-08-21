-- ============================================================================
-- SQL SCHEMA: Real-Time Market Sentiment & NLP Analysis Platform
-- Target Database: PostgreSQL (with ANSI SQL compatibility for SQLite)
-- ============================================================================

-- Table 1: Assets (Financial Tickers / Cryptos / Equities)
CREATE TABLE IF NOT EXISTS assets (
    asset_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    asset_name VARCHAR(100) NOT NULL,
    sector VARCHAR(50) DEFAULT 'General',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Sources (News Publishers / Outlets)
CREATE TABLE IF NOT EXISTS sources (
    source_id SERIAL PRIMARY KEY,
    source_name VARCHAR(100) UNIQUE NOT NULL,
    credibility_score DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 3: Articles (Raw and Cleaned Ingested Text Metadata)
CREATE TABLE IF NOT EXISTS articles (
    article_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    source_name VARCHAR(100) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    content TEXT,
    url VARCHAR(1000),
    asset VARCHAR(10) NOT NULL,
    category VARCHAR(50) DEFAULT 'Market News',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 4: Sentiment Results (NLP Baseline & ML Outputs)
CREATE TABLE IF NOT EXISTS sentiment_results (
    sentiment_id SERIAL PRIMARY KEY,
    article_id VARCHAR(50) UNIQUE REFERENCES articles(article_id) ON DELETE CASCADE,
    compound_score DOUBLE PRECISION NOT NULL,
    positive_score DOUBLE PRECISION NOT NULL,
    negative_score DOUBLE PRECISION NOT NULL,
    neutral_score DOUBLE PRECISION NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    model_version VARCHAR(50) DEFAULT 'VADER_Baseline_v1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 5: Market Prices (Daily Stock Market OHLCV and Returns)
CREATE TABLE IF NOT EXISTS market_prices (
    price_id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    open_price DOUBLE PRECISION NOT NULL,
    high_price DOUBLE PRECISION NOT NULL,
    low_price DOUBLE PRECISION NOT NULL,
    close_price DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL,
    daily_return DOUBLE PRECISION NOT NULL,
    UNIQUE(asset, trade_date)
);

-- Performance Indexes for Time-Series and Analytics Queries
CREATE INDEX IF NOT EXISTS idx_articles_timestamp ON articles(timestamp);
CREATE INDEX IF NOT EXISTS idx_articles_asset ON articles(asset);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_results(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_market_asset_date ON market_prices(asset, trade_date);
