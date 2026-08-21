"""
Real-Time Market Sentiment & NLP Analysis Platform
Streamlit Interactive Analytics Dashboard
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import streamlit as st

from src.data_ingestion import DataIngestion
from src.data_cleaning import DataCleaner
from src.text_preprocessing import TextPreprocessor
from src.sentiment_analysis import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.database import DatabaseManager
from src.analysis import AnalyticsEngine

# Page Config
st.set_page_config(
    page_title="Market Sentiment & NLP Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern High-End UI Aesthetics
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
        color: #00E6FF;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #A0AAB0;
    }

    /* Sentiment Tags */
    .tag-pos { background-color: rgba(46, 204, 113, 0.2); color: #2ECC71; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .tag-neg { background-color: rgba(231, 76, 60, 0.2); color: #E74C3C; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .tag-neu { background-color: rgba(149, 165, 166, 0.2); color: #95A5A6; padding: 4px 10px; border-radius: 6px; font-weight: 600; }

    /* Alert Banner */
    .alert-high {
        background-color: rgba(231, 76, 60, 0.15);
        border-left: 4px solid #E74C3C;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .alert-medium {
        background-color: rgba(241, 196, 15, 0.15);
        border-left: 4px solid #F1C40F;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .alert-info {
        background-color: rgba(52, 152, 219, 0.15);
        border-left: 4px solid #3498DB;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
        color: #FFA726;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_and_process_pipeline(mode: str, asset: str = "ALL"):
    """
    Data Pipeline Loader (Ingestion -> Cleaning -> Preprocessing -> Sentiment -> Feature Engineering -> DB)
    """
    # 1. Ingestion
    ingestion = DataIngestion(mode=mode)
    raw_news = ingestion.fetch_data(asset=asset)
    
    # Load Price Data
    price_df = pd.DataFrame()
    if os.path.exists("sample_data/market_prices_sample.csv"):
        price_df = pd.read_csv("sample_data/market_prices_sample.csv")

    if raw_news.empty:
        return pd.DataFrame(), price_df, {}

    # 2. Cleaning
    cleaner = DataCleaner()
    cleaned_news, quality_report = cleaner.clean_dataset(raw_news)

    # 3. Preprocessing
    preprocessor = TextPreprocessor()
    processed_news = preprocessor.process_dataframe(cleaned_news, text_column="title")

    # 4. Sentiment Analysis
    analyzer = SentimentAnalyzer()
    sentiment_news = analyzer.analyze_dataframe(processed_news, text_column="title")

    # 5. Feature Engineering
    engineer = FeatureEngineer()
    final_df = engineer.create_features(sentiment_news)

    # 6. Database Storage (SQLite / Postgres)
    try:
        db = DatabaseManager()
        db.init_db()
        db.save_dataframe(final_df, table_name="articles", if_exists="replace")
    except Exception as e:
        print(f"Database sync notice: {e}")

    return final_df, price_df, quality_report


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    st.title("📈 Real-Time Market Sentiment & NLP Analysis Platform")
    st.caption("Institutional Financial Sentiment Intelligence Engine & Multi-Asset Analytics")

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.header("⚙️ Platform Controls")

    data_mode = st.sidebar.radio(
        "Data Source Mode",
        options=["Demo (Offline Dataset)", "Live API Ingestion"],
        index=0
    )
    mode_str = "demo" if "Demo" in data_mode else "live"

    # Refresh Button
    if st.sidebar.button("🔄 Refresh Data Pipeline"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Re-running pipeline...")

    # Load Data Pipeline
    with st.spinner("Executing Data & NLP Pipeline..."):
        df_news, df_prices, quality_report = load_and_process_pipeline(mode=mode_str)

    if df_news.empty:
        st.error("No dataset available. Please check sample files or API configuration.")
        return

    # Filters
    st.sidebar.subheader("🎯 Filters & Dimensions")

    # Asset Filter
    available_assets = ["ALL"] + sorted(list(df_news["asset"].unique()))
    selected_asset = st.sidebar.selectbox("Select Asset / Ticker", options=available_assets, index=0)

    # Date Range Filter
    df_news["timestamp_dt"] = pd.to_datetime(df_news["timestamp"])
    min_date = df_news["timestamp_dt"].min().date()
    max_date = df_news["timestamp_dt"].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Source Filter
    available_sources = ["ALL"] + sorted(list(df_news["source"].unique()))
    selected_source = st.sidebar.selectbox("Select News Outlet", options=available_sources, index=0)

    # Sentiment Label Filter
    available_sentiments = ["ALL", "Positive", "Neutral", "Negative"]
    selected_sentiment = st.sidebar.selectbox("Sentiment Filter", options=available_sentiments, index=0)

    # APPLY FILTERS TO DATAFRAME
    filtered_df = df_news.copy()

    if selected_asset != "ALL":
        filtered_df = filtered_df[filtered_df["asset"] == selected_asset]

    if len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["timestamp_dt"].dt.date >= start_d) &
            (filtered_df["timestamp_dt"].dt.date <= end_d)
        ]

    if selected_source != "ALL":
        filtered_df = filtered_df[filtered_df["source"] == selected_source]

    if selected_sentiment != "ALL":
        filtered_df = filtered_df[filtered_df["sentiment_label"] == selected_sentiment]

    # Analytics Engine Instance
    analytics = AnalyticsEngine()
    kpi_metrics = analytics.compute_kpis(filtered_df)

    # PAGE NAVIGATION
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation Pages",
        options=[
            "1. Executive Overview",
            "2. Sentiment Deep-Dive",
            "3. NLP & Keyword Insights",
            "4. Market Price vs. Sentiment",
            "5. Interactive Article Explorer"
        ]
    )

    # --------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW
    # --------------------------------------------------------------------------
    if page.startswith("1"):
        st.header("📊 Executive Overview & Portfolio KPIs")

        # Top KPI Cards Row
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Coverage</div>
                <div class="metric-value">{kpi_metrics['total_articles']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positive Coverage</div>
                <div class="metric-value" style="color:#2ECC71">{kpi_metrics['positive_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Negative Coverage</div>
                <div class="metric-value" style="color:#E74C3C">{kpi_metrics['negative_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Sentiment</div>
                <div class="metric-value" style="color:#F1C40F">{kpi_metrics['avg_sentiment']}</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Most Mentioned</div>
                <div class="metric-value" style="color:#00E6FF">{kpi_metrics['most_mentioned_asset']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Statistical Alerts Section
        alerts = analytics.generate_statistical_alerts(filtered_df)
        if alerts:
            st.subheader("⚠️ Automated Statistical Alerts")
            for alert in alerts[:3]:
                alert_class = "alert-high" if alert["severity"] == "HIGH" else ("alert-medium" if alert["severity"] == "MEDIUM" else "alert-info")
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{alert['title']}</strong> — {alert['message']}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Main Overview Charts Row
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Sentiment Label Breakdown")
            if not filtered_df.empty:
                sent_counts = filtered_df["sentiment_label"].value_counts().reset_index()
                sent_counts.columns = ["Label", "Count"]
                
                fig_pie = px.pie(
                    sent_counts,
                    names="Label",
                    values="Count",
                    color="Label",
                    color_discrete_map={"Positive": "#2ECC71", "Neutral": "#95A5A6", "Negative": "#E74C3C"},
                    hole=0.45
                )
                fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("Asset Level Net Sentiment Benchmark")
            if not filtered_df.empty:
                asset_agg = filtered_df.groupby("asset")["compound_score"].mean().reset_index()
                asset_agg = asset_agg.sort_values(by="compound_score", ascending=True)
                
                fig_asset = px.bar(
                    asset_agg,
                    x="compound_score",
                    y="asset",
                    orientation="h",
                    color="compound_score",
                    color_continuous_scale="RdYlGn",
                    labels={"compound_score": "Mean Compound Score", "asset": "Asset"}
                )
                fig_asset.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
                st.plotly_chart(fig_asset, use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGE 2: SENTIMENT DEEP-DIVE
    # --------------------------------------------------------------------------
    elif page.startswith("2"):
        st.header("🔍 Sentiment Time-Series & Cross-Asset Deep-Dive")

        if filtered_df.empty:
            st.warning("No data for current filters.")
            return

        # Daily Sentiment Trend Line
        st.subheader("Daily Sentiment Score Trajectory")
        daily_sent = filtered_df.groupby(["article_date", "asset"])["compound_score"].mean().reset_index()

        fig_trend = px.line(
            daily_sent,
            x="article_date",
            y="compound_score",
            color="asset" if selected_asset == "ALL" else None,
            markers=True,
            labels={"article_date": "Date", "compound_score": "Mean Sentiment Compound Score"},
            title="Daily Sentiment Compound Score by Asset"
        )
        fig_trend.add_hline(y=0.05, line_dash="dash", line_color="#2ECC71", annotation_text="Positive Threshold")
        fig_trend.add_hline(y=-0.05, line_dash="dash", line_color="#E74C3C", annotation_text="Negative Threshold")
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("News Outlet Sentiment Leans")
            source_agg = filtered_df.groupby(["source", "sentiment_label"]).size().unstack(fill_value=0).reset_index()
            fig_src = px.bar(
                source_agg,
                x="source",
                y=["Positive", "Neutral", "Negative"] if set(["Positive", "Neutral", "Negative"]).issubset(source_agg.columns) else source_agg.columns[1:],
                color_discrete_map={"Positive": "#2ECC71", "Neutral": "#95A5A6", "Negative": "#E74C3C"},
                title="Coverage Share by News Outlet"
            )
            fig_src.update_layout(height=380, barmode="stack")
            st.plotly_chart(fig_src, use_container_width=True)

        with col_b:
            st.subheader("Compound Score Dispersion (Box Plot)")
            fig_box = px.box(
                filtered_df,
                x="asset",
                y="compound_score",
                color="asset",
                points="all",
                title="Sentiment Score Variance per Asset"
            )
            fig_box.update_layout(height=380)
            st.plotly_chart(fig_box, use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGE 3: NLP & KEYWORD INSIGHTS
    # --------------------------------------------------------------------------
    elif page.startswith("3"):
        st.header("🔤 Natural Language Processing & Keyword Insights")

        if filtered_df.empty:
            st.warning("No data for current filters.")
            return

        kw_data = analytics.extract_keywords_by_sentiment(filtered_df, top_k=12)

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.subheader("💚 Top Keywords in Positive Coverage")
            if kw_data["positive"]:
                kw_df = pd.DataFrame(kw_data["positive"], columns=["Keyword", "Frequency"])
                fig_kw_pos = px.bar(
                    kw_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color_discrete_sequence=["#2ECC71"]
                )
                fig_kw_pos.update_layout(height=380, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_kw_pos, use_container_width=True)
            else:
                st.info("No positive coverage keywords found.")

        with col_neg:
            st.subheader("💔 Top Keywords in Negative Coverage")
            if kw_data["negative"]:
                kw_neg_df = pd.DataFrame(kw_data["negative"], columns=["Keyword", "Frequency"])
                fig_kw_neg = px.bar(
                    kw_neg_df,
                    x="Frequency",
                    y="Keyword",
                    orientation="h",
                    color_discrete_sequence=["#E74C3C"]
                )
                fig_kw_neg.update_layout(height=380, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_kw_neg, use_container_width=True)
            else:
                st.info("No negative coverage keywords found.")

        st.markdown("---")
        st.subheader("📰 Recent Headlines with NLP Sentiment Classification")

        recent_df = filtered_df[["timestamp", "asset", "source", "title", "compound_score", "sentiment_label"]].head(10)
        for _, row in recent_df.iterrows():
            lbl = row["sentiment_label"]
            tag_class = "tag-pos" if lbl == "Positive" else ("tag-neg" if lbl == "Negative" else "tag-neu")
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; margin-bottom: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <span class="{tag_class}">{lbl} ({row['compound_score']:+.2f})</span> &nbsp;
                <strong>[{row['asset']}]</strong> {row['title']} &nbsp;
                <span style="color: #A0AAB0; font-size: 0.8rem;">— {row['source']} ({row['timestamp']})</span>
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # PAGE 4: MARKET PRICE VS. SENTIMENT
    # --------------------------------------------------------------------------
    elif page.startswith("4"):
        st.header("📉 Market Price vs. News Sentiment Analysis")

        if df_prices.empty or filtered_df.empty:
            st.warning("Price data unavailable for current selection.")
            return

        corr_res = analytics.correlate_sentiment_market(filtered_df, df_prices)

        if corr_res.get("status") == "Success":
            c_stat1, c_stat2, c_stat3 = st.columns(3)

            with c_stat1:
                st.metric("Sample Alignment Days", corr_res["sample_size"])
            with c_stat2:
                st.metric("Pearson Correlation (r)", corr_res["pearson_correlation"])
            with c_stat3:
                st.metric("Spearman Rank Correlation", corr_res["spearman_correlation"])

            corr_df = pd.DataFrame(corr_res["data"])

            # Dual Axis Chart
            st.subheader("Price Movement vs. Daily Sentiment")

            target_asset = selected_asset if selected_asset != "ALL" else corr_df["asset"].iloc[0]
            asset_sub = corr_df[corr_df["asset"] == target_asset].sort_values(by="trade_date")

            if not asset_sub.empty:
                fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

                fig_dual.add_trace(
                    gg.Scatter(x=asset_sub["trade_date"], y=asset_sub["close_price"], name="Closing Price ($)", line=dict(color="#00E6FF", width=2)),
                    secondary_y=False
                )

                fig_dual.add_trace(
                    gg.Bar(x=asset_sub["trade_date"], y=asset_sub["daily_sentiment"], name="Daily Avg Sentiment", marker_color=np.where(asset_sub["daily_sentiment"] >= 0, "#2ECC71", "#E74C3C"), opacity=0.5),
                    secondary_y=True
                )

                fig_dual.update_layout(title_text=f"Asset: {target_asset} — Stock Price vs Daily Sentiment", height=450)
                fig_dual.update_xaxes(title_text="Trading Date")
                fig_dual.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
                fig_dual.update_yaxes(title_text="Sentiment Score", secondary_y=True)

                st.plotly_chart(fig_dual, use_container_width=True)

            # Scatter Plot: Sentiment vs Returns
            st.subheader("Scatter Plot: Daily Return (%) vs. Sentiment Compound Score")
            fig_scat = px.scatter(
                corr_df,
                x="daily_sentiment",
                y="daily_return",
                color="asset",
                size="article_volume",
                hover_data=["trade_date", "close_price"],
                trendline="ols",
                labels={"daily_sentiment": "Daily Mean Sentiment Score", "daily_return": "Daily Stock Return"}
            )
            fig_scat.update_layout(height=400)
            st.plotly_chart(fig_scat, use_container_width=True)

            # MANDATORY DISCLAIMER
            st.markdown(f"""
            <div class="disclaimer-box">
                <strong>⚠️ ACADEMIC & ANALYTICAL DISCLAIMER:</strong><br>
                {corr_res['disclaimer']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Market price data overlap is limited for current asset filter. Select 'AAPL' or 'NVDA' to inspect price correlation.")

    # --------------------------------------------------------------------------
    # PAGE 5: ARTICLE EXPLORER
    # --------------------------------------------------------------------------
    elif page.startswith("5"):
        st.header("📑 Interactive Article & Query Explorer")

        st.caption(f"Showing {len(filtered_df)} records matching selected filters.")

        cols_to_show = ["timestamp", "asset", "source", "category", "title", "sentiment_label", "compound_score", "url"]
        display_df = filtered_df[[c for c in cols_to_show if c in filtered_df.columns]].copy()

        # Search Bar
        search_query = st.text_input("🔍 Search Headlines / Keywords", "")
        if search_query:
            display_df = display_df[display_df["title"].str.contains(search_query, case=False, na=False)]

        st.dataframe(display_df, use_container_width=True, height=450)

        # Download CSV
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Articles CSV",
            data=csv_data,
            file_name="filtered_market_sentiment_articles.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
