# Unified Data Analysis Platform

This project is a unified data analysis and insights platform built with Streamlit. It provides interactive dashboards and tools for market trend analysis, sales data analysis, and general data exploration.

## Features

- **Market Trend Analysis**: Analyze stock data, technical indicators, and news sentiment for market insights.
- **Sales Data Analysis**: Upload sales CSV files, validate data, detect anomalies, and visualize sales trends.
- **General Data Analysis**: Upload any CSV file for automated data profiling, summary statistics, and visualizations.
- **Combined Reports**: Generate and download comprehensive reports combining market and sales analyses.

## How to Run

1. **Install dependencies** (if not already installed):
   ```powershell
   pip install -r requirements.txt
   # or, if using pyproject.toml:
   pip install .
   ```

2. **Start the Streamlit app**:
   ```powershell
   & "$env:USERPROFILE\AppData\Roaming\Python\Python311\Scripts\streamlit.exe" run app.py --server.port 5000
   ```

3. **Open your browser** and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure

- `app.py` — Main Streamlit application
- `modules/` — Custom analysis modules (market, sales, general, etc.)
- `utils/` — Helper functions
- `sample_sales_data.csv` — Example data for testing
- `pyproject.toml` — Project dependencies

## What You Can Do

- **Market Analysis**: Enter a stock symbol, select a period, and analyze price trends, moving averages, volatility, and news sentiment.
- **Sales Analysis**: Upload your sales data, validate it, analyze trends, detect anomalies, and visualize key metrics.
- **General Data Analysis**: Upload any structured CSV file to get automatic profiling, column type detection, and a suite of visualizations.
- **Download Reports**: Export analysis results and processed data as reports or CSV files.

## Requirements
- Python 3.11+
- All dependencies listed in `pyproject.toml`

## Deployment
- You can deploy this app on Streamlit Community Cloud, Heroku, or any server that supports Python and Streamlit.

