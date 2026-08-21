-- ============================================================================
-- SQL MARKET & SENTIMENT CORRELATION ANALYSIS
-- Joins sentiment aggregates with daily market price movements
-- ============================================================================

-- Query 1: Daily Asset Sentiment paired with Daily Return and Volume
SELECT 
    m.asset,
    m.trade_date,
    m.close_price,
    m.daily_return,
    m.volume,
    COALESCE(ROUND(AVG(s.compound_score), 4), 0.0) AS avg_daily_sentiment,
    COUNT(a.article_id) AS article_count
FROM market_prices m
LEFT JOIN articles a ON m.asset = a.asset AND DATE(a.timestamp) = m.trade_date
LEFT JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY m.asset, m.trade_date, m.close_price, m.daily_return, m.volume
ORDER BY m.asset, m.trade_date ASC;

-- Query 2: High Volatility Days paired with Sentiment Extreme Scores
WITH DailyStats AS (
    SELECT 
        m.asset,
        m.trade_date,
        m.daily_return,
        ABS(m.daily_return) AS abs_return,
        ROUND(AVG(s.compound_score), 4) AS avg_sentiment,
        COUNT(a.article_id) AS volume
    FROM market_prices m
    LEFT JOIN articles a ON m.asset = a.asset AND DATE(a.timestamp) = m.trade_date
    LEFT JOIN sentiment_results s ON a.article_id = s.article_id
    GROUP BY m.asset, m.trade_date, m.daily_return
)
SELECT 
    asset,
    trade_date,
    daily_return,
    avg_sentiment,
    volume
FROM DailyStats
WHERE abs_return >= 0.02 -- Returns > +2% or < -2%
ORDER BY abs_return DESC;
