"""
Script to generate realistic, reproducible offline demo dataset for the
Real-Time Market Sentiment & NLP Analysis Platform.
Generates:
  1. sample_data/financial_news_sample.csv (220 articles)
  2. sample_data/market_prices_sample.csv (Market daily price history for 8 assets over 60 trading days)
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fix random seed for reproducibility
np.random.seed(42)
random.seed(42)

os.makedirs('sample_data', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

assets = [
    ('AAPL', 'Apple Inc.', 'Technology'),
    ('NVDA', 'NVIDIA Corp.', 'Technology'),
    ('TSLA', 'Tesla Inc.', 'Automotive/Tech'),
    ('MSFT', 'Microsoft Corp.', 'Technology'),
    ('GOOGL', 'Alphabet Inc.', 'Technology'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical'),
    ('JPM', 'JPMorgan Chase & Co.', 'Financial Services'),
    ('BTC', 'Bitcoin USD', 'Cryptocurrency')
]

sources = [
    'Bloomberg Financial', 'Reuters Business', 'Wall Street Journal',
    'Financial Times', 'CNBC Markets', 'MarketWatch', 'Seeking Alpha',
    'TechCrunch Tech'
]

categories = ['Earnings', 'Product Launch', 'Regulatory', 'Macroeconomics', 'Executive', 'Analyst Rating']

# Sample headlines templates per sentiment tilt
positive_templates = [
    "{company} Reports Record Quarter Driven by Strong Cloud & AI Demand",
    "{company} Launches Next-Gen AI Chip Architecture, Beating Wall Street Estimates",
    "Analyst Upgrade: {company} Target Price Raised to All-Time High Following Revenue Surge",
    "{company} Secures \$5 Billion Enterprise Contract, Expanding Market Leadership",
    "Bullish Sentiment Grows as {company} Announces Strategic Acquisition in Deep Tech",
    "{company} Operating Margins Expand 350 bps on Cost Optimization and High Demand",
    "Investors Cheer {company}'s Dividend Increase and \$10B Share Buyback Plan",
    "{company} Breakthrough Innovation Receives Glowing Reviews from Industry Experts",
    "Global Expansion Drives {company}'s Double-Digit Revenue Growth in Q2",
    "{company} Outperforms Competitors in Latest Industry Benchmark Ratings"
]

negative_templates = [
    "{company} Shares Slide Following Disappointing Q2 Margin Guidance and Revenue Miss",
    "Regulatory Scrutiny Intensifies for {company} Over Antitrust and Data Privacy Concerns",
    "{company} Recalls Major Product Line Over Hardware Flaws, Impacting Profit Outlook",
    "Wall Street Downgrades {company} Citing Rising Capital Expenditures and Supply Bottlenecks",
    "{company} Faces Substantial Fine in European Court Over Patent Infringement",
    "Supply Chain Disruptions Delay {company}'s Flagship Product Rollout by 6 Months",
    "{company} CEO Warns of Softening Macro Demand in H2, Triggering Sell-Off",
    "Labor Disputes and Factory Delays Disrupt {company}'s Global Operations",
    "Analyst Warning: {company}'s Valuation Overstretched Amid Rising Interest Rates",
    "{company} Cuts Full-Year Forecast as Competition Erodes Core Market Share"
]

neutral_templates = [
    "{company} Announces Schedule for Upcoming Annual Shareholder Meeting and Keynote",
    "{company} Files Form 10-Q with SEC Detailing Q2 Operational Financials",
    "Industry Summary: How {company} and Peers Are Adapting to Evolving Regulations",
    "{company} Appoints New Chief Financial Officer Effective Next Quarter",
    "Summary of {company}'s Latest Sustainability and Environmental Impact Report",
    "{company} to Present at Upcoming Global Technology & Markets Conference",
    "Market Snapshot: Trading Volume Steady for {company} Ahead of Earnings Release",
    "{company} Expands Corporate Headquarters and R&D Facilities in North America",
    "Overview of Institutional Holdings and Ownership Changes for {company}",
    "{company} Finalizes Terms of Routine Corporate Debt Refinancing Structure"
]

start_date = datetime(2026, 6, 1)
end_date = datetime(2026, 8, 15)
date_range_days = (end_date - start_date).days

news_rows = []
article_counter = 1001

for _ in range(220):
    symbol, company_name, sector = random.choice(assets)
    sentiment_tilt = random.choices(['positive', 'negative', 'neutral'], weights=[0.42, 0.33, 0.25])[0]
    
    if sentiment_tilt == 'positive':
        template = random.choice(positive_templates)
        desc_text = f"{company_name} demonstrated exceptional financial performance and operational strength, surpassing market analyst expectations across key business metrics."
    elif sentiment_tilt == 'negative':
        template = random.choice(negative_templates)
        desc_text = f"{company_name} faces operational headwinds and market uncertainties as unexpected costs and regulatory obstacles pressure near-term earnings outlook."
    else:
        template = random.choice(neutral_templates)
        desc_text = f"{company_name} released routine operational documentation and corporate updates regarding scheduled governance events and executive updates."
        
    headline = template.format(company=symbol)
    source = random.choice(sources)
    category = random.choice(categories)
    
    # Random timestamp within date range
    random_days = random.randint(0, date_range_days)
    random_hours = random.randint(6, 22)
    random_mins = random.randint(0, 59)
    random_secs = random.randint(0, 59)
    pub_dt = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_mins, seconds=random_secs)
    timestamp_str = pub_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    url = f"https://www.{source.lower().replace(' ', '')}.com/articles/{symbol.lower()}-{article_counter}"
    content_text = f"{headline}. {desc_text} Analysts and investors are closely monitoring {symbol} as market conditions fluctuate. Industry insiders note that macro trends, regulatory developments, and corporate execution will dictate the stock's trajectory over coming quarters."
    
    news_rows.append({
        'article_id': f"ART-{article_counter}",
        'timestamp': timestamp_str,
        'source': source,
        'title': headline,
        'description': desc_text,
        'content': content_text,
        'url': url,
        'asset': symbol,
        'category': category
    })
    article_counter += 1

news_df = pd.DataFrame(news_rows)
news_df.sort_values(by='timestamp', inplace=True)
news_df.to_csv('sample_data/financial_news_sample.csv', index=False)
news_df.to_csv('data/raw/financial_news_raw.csv', index=False)
print(f"Generated {len(news_df)} news articles in sample_data/financial_news_sample.csv")

# Generate matching daily market price dataset
price_base = {
    'AAPL': 210.0,
    'NVDA': 125.0,
    'TSLA': 240.0,
    'MSFT': 440.0,
    'GOOGL': 175.0,
    'AMZN': 185.0,
    'JPM': 205.0,
    'BTC': 64000.0
}

volatility = {
    'AAPL': 0.015,
    'NVDA': 0.028,
    'TSLA': 0.035,
    'MSFT': 0.014,
    'GOOGL': 0.018,
    'AMZN': 0.020,
    'JPM': 0.012,
    'BTC': 0.040
}

price_rows = []
trading_days = pd.date_range(start='2026-06-01', end='2026-08-15', freq='B')

for symbol, company_name, sector in assets:
    curr_price = price_base[symbol]
    vol = volatility[symbol]
    
    for dt in trading_days:
        date_str = dt.strftime('%Y-%m-%d')
        # Simulate daily price change with mild random walk
        pct_change = np.random.normal(0.0005, vol)
        open_p = round(curr_price, 2)
        close_p = round(open_p * (1 + pct_change), 2)
        high_p = round(max(open_p, close_p) * (1 + abs(np.random.normal(0, vol * 0.5))), 2)
        low_p = round(min(open_p, close_p) * (1 - abs(np.random.normal(0, vol * 0.5))), 2)
        volume = int(np.random.normal(40000000, 10000000) if symbol != 'BTC' else np.random.normal(25000000000, 5000000000))
        daily_return = round((close_p - open_p) / open_p, 4)
        
        price_rows.append({
            'asset': symbol,
            'trade_date': date_str,
            'open_price': open_p,
            'high_price': high_p,
            'low_price': low_p,
            'close_price': close_p,
            'volume': max(100000, volume),
            'daily_return': daily_return
        })
        curr_price = close_p

price_df = pd.DataFrame(price_rows)
price_df.to_csv('sample_data/market_prices_sample.csv', index=False)
price_df.to_csv('data/raw/market_prices_raw.csv', index=False)
print(f"Generated {len(price_df)} price records in sample_data/market_prices_sample.csv")
