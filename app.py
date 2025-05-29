import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import custom modules
from modules.market_analyzer import MarketAnalyzer
from modules.sales_analyzer import SalesAnalyzer
from modules.data_processor import DataProcessor
from modules.report_generator import ReportGenerator
from utils.helpers import format_currency, get_date_range_options

# Page configuration
st.set_page_config(
    page_title="Unified Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

def main():
    # Main title
    st.title("📊 Unified Data Analysis Platform")
    st.markdown("### Market Trends & Sales Insights Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Market Trend Analysis", "Sales Data Analysis", "Combined Reports"]
    )
    
    # Initialize analyzers
    market_analyzer = MarketAnalyzer()
    sales_analyzer = SalesAnalyzer()
    data_processor = DataProcessor()
    report_generator = ReportGenerator()
    
    if page == "Overview":
        show_overview()
    elif page == "Market Trend Analysis":
        show_market_analysis(market_analyzer, data_processor)
    elif page == "Sales Data Analysis":
        show_sales_analysis(sales_analyzer, data_processor)
    elif page == "Combined Reports":
        show_combined_reports(report_generator)

def show_overview():
    """Display the overview page with summary statistics and quick insights"""
    st.header("📈 Platform Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏪 Market Analysis Features")
        st.markdown("""
        - **Stock Data Analysis**: Real-time stock prices and historical trends
        - **Sentiment Analysis**: News sentiment impact on market movements
        - **Technical Indicators**: Moving averages, volatility analysis
        - **Market Insights**: Automated trend detection and predictions
        """)
        
        if st.button("Start Market Analysis"):
            st.switch_page = "Market Trend Analysis"
    
    with col2:
        st.subheader("🛍️ Sales Analysis Features")
        st.markdown("""
        - **Sales Data Upload**: CSV file processing and validation
        - **Anomaly Detection**: Identify unusual sales patterns
        - **Performance Metrics**: KPIs and trend analysis
        - **Interactive Visualizations**: Charts and dashboards
        """)
        
        if st.button("Start Sales Analysis"):
            st.switch_page = "Sales Data Analysis"
    
    # Quick stats if data is available
    if st.session_state.market_data is not None or st.session_state.sales_data is not None:
        st.header("📊 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.market_data is not None:
            with col1:
                st.metric("Market Data Points", len(st.session_state.market_data))
            with col2:
                if 'Close' in st.session_state.market_data.columns:
                    latest_price = st.session_state.market_data['Close'].iloc[-1]
                    st.metric("Latest Price", format_currency(latest_price))
        
        if st.session_state.sales_data is not None:
            with col3:
                st.metric("Sales Records", len(st.session_state.sales_data))
            with col4:
                if 'revenue' in st.session_state.sales_data.columns:
                    total_revenue = st.session_state.sales_data['revenue'].sum()
                    st.metric("Total Revenue", format_currency(total_revenue))

def show_market_analysis(market_analyzer, data_processor):
    """Display market trend analysis interface"""
    st.header("📈 Market Trend Analysis")
    
    # Input section
    with st.expander("Market Data Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL)")
        
        with col2:
            period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        with col3:
            news_sources = st.multiselect(
                "News Sources",
                ["Financial News", "Social Media", "Press Releases"],
                default=["Financial News"]
            )
    
    # Fetch and analyze data
    if st.button("Analyze Market Trends", type="primary"):
        with st.spinner("Fetching market data and analyzing trends..."):
            try:
                # Fetch stock data
                stock_data = market_analyzer.fetch_stock_data(symbol, period)
                
                if stock_data is not None and not stock_data.empty:
                    st.session_state.market_data = stock_data
                    
                    # Perform technical analysis
                    technical_analysis = market_analyzer.perform_technical_analysis(stock_data)
                    
                    # Fetch and analyze news sentiment
                    news_sentiment = market_analyzer.analyze_news_sentiment(symbol)
                    
                    # Store results
                    st.session_state.analysis_results['market'] = {
                        'stock_data': stock_data,
                        'technical_analysis': technical_analysis,
                        'news_sentiment': news_sentiment
                    }
                    
                    st.success("Market analysis completed successfully!")
                else:
                    st.error("Failed to fetch stock data. Please check the symbol and try again.")
                    
            except Exception as e:
                st.error(f"Error during market analysis: {str(e)}")
    
    # Display results
    if 'market' in st.session_state.analysis_results:
        display_market_results()

def display_market_results():
    """Display market analysis results"""
    results = st.session_state.analysis_results['market']
    stock_data = results['stock_data']
    technical_analysis = results['technical_analysis']
    news_sentiment = results['news_sentiment']
    
    # Stock price chart
    st.subheader("📊 Stock Price Trends")
    fig = px.line(stock_data.reset_index(), x='Date', y='Close', 
                  title="Stock Price Over Time")
    fig.add_scatter(x=stock_data.index, y=stock_data['MA_20'], 
                   mode='lines', name='20-day MA', line=dict(dash='dash'))
    fig.add_scatter(x=stock_data.index, y=stock_data['MA_50'], 
                   mode='lines', name='50-day MA', line=dict(dash='dot'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", format_currency(stock_data['Close'].iloc[-1]))
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        st.metric("Daily Change", format_currency(price_change), 
                 delta=f"{(price_change/stock_data['Close'].iloc[-2]*100):.2f}%")
    
    with col2:
        st.metric("Volatility", f"{technical_analysis['volatility']:.2f}%")
        st.metric("RSI", f"{technical_analysis['rsi']:.2f}")
    
    with col3:
        if news_sentiment:
            avg_sentiment = sum([s['score'] for s in news_sentiment]) / len(news_sentiment)
            sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            st.metric("News Sentiment", sentiment_label, delta=f"{avg_sentiment:.2f}")
    
    # Volume analysis
    st.subheader("📈 Volume Analysis")
    fig_volume = px.bar(stock_data.reset_index(), x='Date', y='Volume',
                       title="Trading Volume")
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # News sentiment details
    if news_sentiment:
        st.subheader("📰 News Sentiment Analysis")
        sentiment_df = pd.DataFrame(news_sentiment)
        
        fig_sentiment = px.scatter(sentiment_df, x='date', y='score',
                                 size='relevance', hover_data=['headline'],
                                 title="News Sentiment Over Time")
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Display recent headlines
        st.subheader("Recent Headlines")
        for news in news_sentiment[:5]:
            sentiment_emoji = "📈" if news['score'] > 0.1 else "📉" if news['score'] < -0.1 else "➡️"
            st.write(f"{sentiment_emoji} **{news['headline']}** (Score: {news['score']:.2f})")

def show_sales_analysis(sales_analyzer, data_processor):
    """Display sales data analysis interface"""
    st.header("🛍️ Sales Data Analysis")
    
    # File upload section
    with st.expander("Data Upload", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload Sales Data (CSV)",
            type=['csv'],
            help="Upload a CSV file with sales data including columns like date, product, revenue, quantity"
        )
        
        if uploaded_file is not None:
            try:
                sales_data = pd.read_csv(uploaded_file)
                st.session_state.sales_data = sales_data
                
                st.success(f"Data uploaded successfully! {len(sales_data)} records loaded.")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(sales_data.head())
                
                # Data validation
                validation_results = data_processor.validate_sales_data(sales_data)
                if validation_results['is_valid']:
                    st.success("✅ Data validation passed!")
                else:
                    st.warning(f"⚠️ Data validation issues: {', '.join(validation_results['issues'])}")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Analysis configuration
    if st.session_state.sales_data is not None:
        with st.expander("Analysis Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                date_column = st.selectbox("Date Column", st.session_state.sales_data.columns)
                revenue_column = st.selectbox("Revenue Column", st.session_state.sales_data.columns)
            
            with col2:
                anomaly_threshold = st.slider("Anomaly Detection Sensitivity", 0.1, 0.5, 0.2)
                analysis_period = st.selectbox("Analysis Period", ["Daily", "Weekly", "Monthly"], index=1)
        
        # Perform analysis
        if st.button("Analyze Sales Data", type="primary"):
            with st.spinner("Analyzing sales data and detecting anomalies..."):
                try:
                    # Prepare data
                    processed_data = data_processor.process_sales_data(
                        st.session_state.sales_data, date_column, revenue_column
                    )
                    
                    # Perform sales analysis
                    sales_analysis = sales_analyzer.analyze_sales_trends(processed_data, analysis_period)
                    
                    # Detect anomalies
                    anomalies = sales_analyzer.detect_anomalies(processed_data, anomaly_threshold)
                    
                    # Store results
                    st.session_state.analysis_results['sales'] = {
                        'processed_data': processed_data,
                        'sales_analysis': sales_analysis,
                        'anomalies': anomalies
                    }
                    
                    st.success("Sales analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during sales analysis: {str(e)}")
        
        # Display results
        if 'sales' in st.session_state.analysis_results:
            display_sales_results()

def display_sales_results():
    """Display sales analysis results"""
    results = st.session_state.analysis_results['sales']
    processed_data = results['processed_data']
    sales_analysis = results['sales_analysis']
    anomalies = results['anomalies']
    
    # Key metrics
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = processed_data['revenue'].sum()
        st.metric("Total Revenue", format_currency(total_revenue))
    
    with col2:
        avg_daily_revenue = processed_data['revenue'].mean()
        st.metric("Avg Daily Revenue", format_currency(avg_daily_revenue))
    
    with col3:
        revenue_growth = sales_analysis.get('growth_rate', 0)
        st.metric("Growth Rate", f"{revenue_growth:.2f}%")
    
    with col4:
        anomaly_count = len(anomalies)
        st.metric("Anomalies Detected", anomaly_count)
    
    # Revenue trends
    st.subheader("📈 Revenue Trends")
    fig_revenue = px.line(processed_data, x='date', y='revenue',
                         title="Revenue Over Time")
    
    # Highlight anomalies
    if not anomalies.empty:
        fig_revenue.add_scatter(x=anomalies['date'], y=anomalies['revenue'],
                               mode='markers', name='Anomalies',
                               marker=dict(color='red', size=10, symbol='x'))
    
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Sales distribution
    st.subheader("📊 Sales Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily distribution
        fig_dist = px.histogram(processed_data, x='revenue', nbins=30,
                               title="Revenue Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Weekly patterns
        processed_data['day_of_week'] = pd.to_datetime(processed_data['date']).dt.day_name()
        weekly_revenue = processed_data.groupby('day_of_week')['revenue'].mean().reset_index()
        fig_weekly = px.bar(weekly_revenue, x='day_of_week', y='revenue',
                           title="Average Revenue by Day of Week")
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Anomaly details
    if not anomalies.empty:
        st.subheader("🚨 Detected Anomalies")
        st.write(f"Found {len(anomalies)} anomalous data points:")
        
        anomaly_display = anomalies.copy()
        anomaly_display['revenue'] = anomaly_display['revenue'].apply(format_currency)
        st.dataframe(anomaly_display, use_container_width=True)

def show_combined_reports(report_generator):
    """Display combined reports interface"""
    st.header("📋 Combined Reports")
    
    # Check if both analyses are available
    has_market_data = 'market' in st.session_state.analysis_results
    has_sales_data = 'sales' in st.session_state.analysis_results
    
    if not has_market_data and not has_sales_data:
        st.warning("No analysis data available. Please run Market Analysis or Sales Analysis first.")
        return
    
    # Report configuration
    with st.expander("Report Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Detailed Analysis", "Custom Report"]
            )
            
            include_charts = st.checkbox("Include Charts", value=True)
        
        with col2:
            report_format = st.selectbox("Export Format", ["HTML", "PDF", "CSV"])
            
            date_range = st.date_input(
                "Report Period",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                max_value=datetime.now()
            )
    
    # Generate report
    if st.button("Generate Combined Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            try:
                report_data = {
                    'market_results': st.session_state.analysis_results.get('market'),
                    'sales_results': st.session_state.analysis_results.get('sales'),
                    'config': {
                        'report_type': report_type,
                        'include_charts': include_charts,
                        'date_range': date_range
                    }
                }
                
                # Generate report
                report_content = report_generator.generate_combined_report(report_data)
                
                st.success("Report generated successfully!")
                
                # Display report preview
                st.subheader("📖 Report Preview")
                st.markdown(report_content['summary'])
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    if report_format == "HTML":
                        st.download_button(
                            "Download HTML Report",
                            data=report_content['html'],
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html"
                        )
                
                with col2:
                    if report_content.get('csv_data'):
                        st.download_button(
                            "Download CSV Data",
                            data=report_content['csv_data'],
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
    
    # Display individual summaries
    if has_market_data or has_sales_data:
        st.subheader("📊 Analysis Summaries")
        
        col1, col2 = st.columns(2)
        
        if has_market_data:
            with col1:
                st.markdown("#### 📈 Market Analysis Summary")
                market_data = st.session_state.analysis_results['market']
                
                # Quick stats
                stock_data = market_data['stock_data']
                current_price = stock_data['Close'].iloc[-1]
                price_change = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
                
                st.write(f"**Current Price:** {format_currency(current_price)}")
                st.write(f"**Period Change:** {price_change:.2f}%")
                
                if market_data['news_sentiment']:
                    avg_sentiment = sum([s['score'] for s in market_data['news_sentiment']]) / len(market_data['news_sentiment'])
                    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                    st.write(f"**News Sentiment:** {sentiment_label} ({avg_sentiment:.2f})")
        
        if has_sales_data:
            with col2:
                st.markdown("#### 🛍️ Sales Analysis Summary")
                sales_data = st.session_state.analysis_results['sales']
                
                # Quick stats
                processed_data = sales_data['processed_data']
                total_revenue = processed_data['revenue'].sum()
                anomaly_count = len(sales_data['anomalies'])
                
                st.write(f"**Total Revenue:** {format_currency(total_revenue)}")
                st.write(f"**Data Points:** {len(processed_data)}")
                st.write(f"**Anomalies:** {anomaly_count}")

if __name__ == "__main__":
    main()
