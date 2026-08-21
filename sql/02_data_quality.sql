-- ============================================================================
-- SQL DATA QUALITY AUDIT SCRIPTS
-- Validates record integrity, null values, orphans, and date boundaries
-- ============================================================================

-- Audit Query 1: Missing values and null count verification
SELECT 
    COUNT(*) AS total_articles,
    SUM(CASE WHEN title IS NULL OR title = '' THEN 1 ELSE 0 END) AS missing_titles,
    SUM(CASE WHEN description IS NULL OR description = '' THEN 1 ELSE 0 END) AS missing_descriptions,
    SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) AS missing_content,
    SUM(CASE WHEN url IS NULL OR url = '' THEN 1 ELSE 0 END) AS missing_urls
FROM articles;

-- Audit Query 2: Duplicate headline identification
SELECT 
    LOWER(TRIM(title)) AS clean_title, 
    COUNT(*) AS occurrences
FROM articles
GROUP BY LOWER(TRIM(title))
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- Audit Query 3: Orphan sentiment records (Sentiment without matching article)
SELECT 
    s.sentiment_id, 
    s.article_id
FROM sentiment_results s
LEFT JOIN articles a ON s.article_id = a.article_id
WHERE a.article_id IS NULL;

-- Audit Query 4: Unscored articles (Articles without sentiment scores)
SELECT 
    a.article_id, 
    a.title, 
    a.timestamp
FROM articles a
LEFT JOIN sentiment_results s ON a.article_id = s.article_id
WHERE s.sentiment_id IS NULL;

-- Audit Query 5: Date range boundary validation
SELECT 
    MIN(timestamp) AS earliest_article,
    MAX(timestamp) AS latest_article,
    COUNT(DISTINCT DATE(timestamp)) AS active_days_count
FROM articles;
