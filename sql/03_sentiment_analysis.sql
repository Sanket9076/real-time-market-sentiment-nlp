-- ============================================================================
-- SQL SENTIMENT ANALYTICS SCRIPTS
-- Categorical distributions, asset sentiment rankings, and source breakdown
-- ============================================================================

-- Query 1: Global sentiment distribution count and percentage
SELECT 
    s.sentiment_label,
    COUNT(*) AS total_articles,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS sentiment_percentage
FROM sentiment_results s
GROUP BY s.sentiment_label
ORDER BY total_articles DESC;

-- Query 2: Asset-level average sentiment ranking
SELECT 
    a.asset,
    COUNT(a.article_id) AS total_articles,
    ROUND(AVG(s.compound_score), 4) AS avg_compound_score,
    ROUND(AVG(s.positive_score), 4) AS avg_positive_score,
    ROUND(AVG(s.negative_score), 4) AS avg_negative_score
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.asset
ORDER BY avg_compound_score DESC;

-- Query 3: News source sentiment credibility breakdown
SELECT 
    a.source_name,
    COUNT(a.article_id) AS total_coverage,
    ROUND(AVG(s.compound_score), 4) AS avg_sentiment,
    SUM(CASE WHEN s.sentiment_label = 'Positive' THEN 1 ELSE 0 END) AS positive_count,
    SUM(CASE WHEN s.sentiment_label = 'Negative' THEN 1 ELSE 0 END) AS negative_count,
    SUM(CASE WHEN s.sentiment_label = 'Neutral' THEN 1 ELSE 0 END) AS neutral_count
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY a.source_name
HAVING COUNT(a.article_id) >= 5
ORDER BY total_coverage DESC;

-- Query 4: Daily sentiment trend time-series
SELECT 
    DATE(a.timestamp) AS article_date,
    COUNT(a.article_id) AS daily_article_volume,
    ROUND(AVG(s.compound_score), 4) AS daily_avg_sentiment
FROM articles a
JOIN sentiment_results s ON a.article_id = s.article_id
GROUP BY DATE(a.timestamp)
ORDER BY article_date ASC;
