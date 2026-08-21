-- ============================================================================
-- SQL ADVANCED ANALYTICS (20+ PRODUCTION-GRADE SQL QUERIES)
-- Features CTEs, Window Functions (AVG OVER, LAG, RANK), Subqueries, CASE statements
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: Total Volume of Articles Ingested
-- Explanation: Baseline metric counting all unique news articles in database.
-- ----------------------------------------------------------------------------
SELECT COUNT(*) AS total_articles FROM articles;

-- ----------------------------------------------------------------------------
-- QUERY 2: Total Articles Breakdown by News Source
-- Explanation: Audits publisher distribution and coverage density.
-- ----------------------------------------------------------------------------
SELECT source_name, COUNT(*) AS article_count
FROM articles
GROUP BY source_name
ORDER BY article_count DESC;

-- ----------------------------------------------------------------------------
-- QUERY 3: Total Articles Ingested per Financial Asset
-- Explanation: Identifies which assets receive the highest news coverage.
-- ----------------------------------------------------------------------------
SELECT asset, COUNT(*) AS asset_article_count
FROM articles
GROUP BY asset
ORDER BY asset_article_count DESC;

-- ----------------------------------------------------------------------------
-- QUERY 4: Sentiment Label Count Breakdown (Positive / Neutral / Negative)
-- Explanation: Aggregates categorical sentiment labels across the dataset.
-- ----------------------------------------------------------------------------
SELECT sentiment_label, COUNT(*) AS count
FROM sentiment_results
GROUP BY sentiment_label
ORDER BY count DESC;

-- ----------------------------------------------------------------------------
-- QUERY 5: Average Sentiment Score by Asset
-- Explanation: Ranks assets by overall net sentiment compound score.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    ROUND(AVG(s.compound_score), 4) AS avg_compound_score
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
ORDER BY avg_compound_score DESC;

-- ----------------------------------------------------------------------------
-- QUERY 6: Average Sentiment Score by News Source
-- Explanation: Measures publisher sentiment bias (bullish vs bearish lean).
-- ----------------------------------------------------------------------------
SELECT 
    a.source_name,
    ROUND(AVG(s.compound_score), 4) AS source_avg_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.source_name
ORDER BY source_avg_sentiment DESC;

-- ----------------------------------------------------------------------------
-- QUERY 7: Daily Aggregated Sentiment Time-Series
-- Explanation: Tracks day-by-day sentiment score fluctuation.
-- ----------------------------------------------------------------------------
SELECT 
    DATE(a.timestamp) AS trade_date,
    COUNT(a.article_id) AS total_news_volume,
    ROUND(AVG(s.compound_score), 4) AS daily_avg_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY DATE(a.timestamp)
ORDER BY trade_date ASC;

-- ----------------------------------------------------------------------------
-- QUERY 8: Weekly Sentiment Aggregation (Strata Analysis)
-- Explanation: Groups news sentiment by calendar week for macro trends.
-- ----------------------------------------------------------------------------
SELECT 
    DATE(a.timestamp, 'weekday 0', '-6 days') AS week_start_date,
    COUNT(a.article_id) AS weekly_articles,
    ROUND(AVG(s.compound_score), 4) AS weekly_avg_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY week_start_date
ORDER BY week_start_date ASC;

-- ----------------------------------------------------------------------------
-- QUERY 9: Monthly Sentiment Rollup
-- Explanation: Monthly summary of sentiment compound scores.
-- ----------------------------------------------------------------------------
SELECT 
    STRFTIME('%Y-%m', a.timestamp) AS year_month,
    COUNT(a.article_id) AS monthly_articles,
    ROUND(AVG(s.compound_score), 4) AS monthly_avg_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY year_month
ORDER BY year_month ASC;

-- ----------------------------------------------------------------------------
-- QUERY 10: Ranks Assets with Most Negative Sentiment Impact
-- Explanation: Filters assets with lowest compound score & high negative ratio.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    COUNT(a.article_id) AS article_count,
    ROUND(AVG(s.negative_score), 4) AS avg_neg_score,
    ROUND(AVG(s.compound_score), 4) AS net_compound_score
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
HAVING COUNT(a.article_id) >= 3
ORDER BY net_compound_score ASC;

-- ----------------------------------------------------------------------------
-- QUERY 11: Ranks Assets with Most Positive Sentiment Impact
-- Explanation: Identifies top bullish assets by net positive sentiment.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    COUNT(a.article_id) AS article_count,
    ROUND(AVG(s.positive_score), 4) AS avg_pos_score,
    ROUND(AVG(s.compound_score), 4) AS net_compound_score
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
HAVING COUNT(a.article_id) >= 3
ORDER BY net_compound_score DESC;

-- ----------------------------------------------------------------------------
-- QUERY 12: Top Mentioned Assets by Article Coverage Volume
-- Explanation: Counts total articles per ticker to assess market mindshare.
-- ----------------------------------------------------------------------------
SELECT 
    asset,
    COUNT(article_id) AS total_coverage
FROM articles
GROUP BY asset
ORDER BY total_coverage DESC
LIMIT 5;

-- ----------------------------------------------------------------------------
-- QUERY 13: Sentiment Share & Percentage Distribution Breakdown
-- Explanation: Calculates relative proportion of Positive, Neutral, Negative coverage.
-- ----------------------------------------------------------------------------
SELECT 
    sentiment_label,
    COUNT(*) AS absolute_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sentiment_results), 2) AS percentage_share
FROM sentiment_results
GROUP BY sentiment_label;

-- ----------------------------------------------------------------------------
-- QUERY 14: WINDOW FUNCTION - 7-Day Rolling Moving Average Sentiment per Asset
-- Explanation: Uses AVG() OVER window function to smooth out daily sentiment noise.
-- ----------------------------------------------------------------------------
WITH DailyAssetSentiment AS (
    SELECT 
        a.asset,
        DATE(a.timestamp) AS trade_date,
        AVG(s.compound_score) AS daily_sentiment
    FROM articles a
    JOIN sentiment_results s ON a.article_id = s.article_id
    GROUP BY a.asset, DATE(a.timestamp)
)
SELECT 
    asset,
    trade_date,
    ROUND(daily_sentiment, 4) AS raw_daily_sentiment,
    ROUND(AVG(daily_sentiment) OVER (
        PARTITION BY asset 
        ORDER BY trade_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 4) AS rolling_7day_sentiment
FROM DailyAssetSentiment
ORDER BY asset, trade_date ASC;

-- ----------------------------------------------------------------------------
-- QUERY 15: WINDOW FUNCTION - Day-over-Day Sentiment Velocity (LAG)
-- Explanation: Calculates sentiment delta from previous trading day using LAG().
-- ----------------------------------------------------------------------------
WITH DailyScores AS (
    SELECT 
        a.asset,
        DATE(a.timestamp) AS trade_date,
        AVG(s.compound_score) AS avg_sentiment
    FROM articles a
    JOIN sentiment_results s ON a.article_id = s.article_id
    GROUP BY a.asset, DATE(a.timestamp)
)
SELECT 
    asset,
    trade_date,
    ROUND(avg_sentiment, 4) AS current_day_sentiment,
    ROUND(LAG(avg_sentiment, 1) OVER (PARTITION BY asset ORDER BY trade_date), 4) AS previous_day_sentiment,
    ROUND(avg_sentiment - LAG(avg_sentiment, 1) OVER (PARTITION BY asset ORDER BY trade_date), 4) AS sentiment_velocity
FROM DailyScores
ORDER BY asset, trade_date ASC;

-- ----------------------------------------------------------------------------
-- QUERY 16: WINDOW FUNCTION - Dense Rank of Assets by Net Sentiment Score
-- Explanation: Ranks assets dynamically using DENSE_RANK() OVER.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    ROUND(AVG(s.compound_score), 4) AS avg_compound,
    DENSE_RANK() OVER (ORDER BY AVG(s.compound_score) DESC) AS sentiment_rank
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset;

-- ----------------------------------------------------------------------------
-- QUERY 17: CTE + SUBQUERY - Anomaly Market Return Days vs Article Sentiment
-- Explanation: Isolates high-return trading days (>2% move) and joins sentiment.
-- ----------------------------------------------------------------------------
WITH HighVolatilityDays AS (
    SELECT asset, trade_date, daily_return, close_price
    FROM market_prices
    WHERE ABS(daily_return) >= 0.02
)
SELECT 
    h.asset,
    h.trade_date,
    h.daily_return,
    h.close_price,
    ROUND(AVG(s.compound_score), 4) AS sentiment_on_anomaly_day,
    COUNT(a.article_id) AS article_volume
FROM HighVolatilityDays h
LEFT JOIN articles a ON h.asset = a.asset AND DATE(a.timestamp) = h.trade_date
LEFT JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY h.asset, h.trade_date, h.daily_return, h.close_price
ORDER BY ABS(h.daily_return) DESC;

-- ----------------------------------------------------------------------------
-- QUERY 18: CASE STATEMENT - Asset Polarity Categorization Classifier
-- Explanation: Uses CASE WHEN to classify asset baseline stance into Bearish/Neutral/Bullish.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    ROUND(AVG(s.compound_score), 4) AS mean_compound,
    CASE 
        WHEN AVG(s.compound_score) >= 0.15 THEN 'Strongly Bullish'
        WHEN AVG(s.compound_score) >= 0.05 THEN 'Mildly Bullish'
        WHEN AVG(s.compound_score) <= -0.15 THEN 'Strongly Bearish'
        WHEN AVG(s.compound_score) <= -0.05 THEN 'Mildly Bearish'
        ELSE 'Neutral / Mixed'
    END AS asset_sentiment_tier
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
ORDER BY mean_compound DESC;

-- ----------------------------------------------------------------------------
-- QUERY 19: MULTI-TABLE JOIN - Full Market Price and Sentiment Alignment
-- Explanation: Comprehensive join across articles, sentiment, and stock market tables.
-- ----------------------------------------------------------------------------
SELECT 
    a.article_id,
    a.asset,
    a.timestamp,
    a.source_name,
    a.title,
    s.compound_score,
    s.sentiment_label,
    m.close_price,
    m.daily_return
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
LEFT JOIN market_prices m ON a.asset = m.asset AND DATE(a.timestamp) = m.trade_date
ORDER BY a.timestamp DESC
LIMIT 20;

-- ----------------------------------------------------------------------------
-- QUERY 20: SUBQUERY - Assets Outperforming Portfolio Average Sentiment
-- Explanation: Subquery compares each asset's mean sentiment against portfolio baseline.
-- ----------------------------------------------------------------------------
SELECT 
    a.asset,
    ROUND(AVG(s.compound_score), 4) AS asset_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
HAVING AVG(s.compound_score) > (
    SELECT AVG(compound_score) FROM sentiment_results
)
ORDER BY asset_sentiment DESC;

-- ----------------------------------------------------------------------------
-- QUERY 21: CTE + WINDOW FUNCTION - Cumulative Running Article Count and Sentiment
-- Explanation: Computes running cumulative total of news articles over time.
-- ----------------------------------------------------------------------------
WITH DailyVolume AS (
    SELECT 
        DATE(timestamp) AS pub_date,
        COUNT(*) AS daily_cnt
    FROM articles
    GROUP BY DATE(timestamp)
)
SELECT 
    pub_date,
    daily_cnt,
    SUM(daily_cnt) OVER (ORDER BY pub_date ASC) AS running_cumulative_articles
FROM DailyVolume
ORDER BY pub_date ASC;
