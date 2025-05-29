import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import os

class MarketAnalyzer:
    """Market trend analysis and sentiment analysis for financial data"""
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        
    def fetch_stock_data(self, symbol, period="6mo"):
        """
        Fetch stock data using yfinance
        
        Args:
            symbol (str): Stock ticker symbol
            period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV and technical indicators
        """
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Add technical indicators
            hist_data = self._add_technical_indicators(hist_data)
            
            return hist_data
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data):
        """Add technical indicators to stock data"""
        # Moving averages
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        
        return data
    
    def perform_technical_analysis(self, stock_data):
        """
        Perform comprehensive technical analysis
        
        Args:
            stock_data (pandas.DataFrame): Stock data with technical indicators
        
        Returns:
            dict: Technical analysis results
        """
        latest_data = stock_data.iloc[-1]
        
        analysis = {
            'current_price': latest_data['Close'],
            'ma_20': latest_data['MA_20'],
            'ma_50': latest_data['MA_50'],
            'rsi': latest_data['RSI'],
            'volatility': latest_data['Volatility'],
            'bb_position': self._get_bollinger_position(latest_data),
            'trend_direction': self._determine_trend(stock_data),
            'support_resistance': self._find_support_resistance(stock_data),
            'volume_analysis': self._analyze_volume(stock_data)
        }
        
        # Add trading signals
        analysis['signals'] = self._generate_trading_signals(stock_data)
        
        return analysis
    
    def _get_bollinger_position(self, latest_data):
        """Determine position relative to Bollinger Bands"""
        price = latest_data['Close']
        upper = latest_data['BB_upper']
        lower = latest_data['BB_lower']
        
        if price > upper:
            return "Above Upper Band (Overbought)"
        elif price < lower:
            return "Below Lower Band (Oversold)"
        else:
            return "Within Bands (Normal)"
    
    def _determine_trend(self, stock_data):
        """Determine overall trend direction"""
        recent_data = stock_data.tail(20)
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_data))
        y = recent_data['Close'].values
        
        if len(y) < 2:
            return "Insufficient Data"
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.5:
            return "Strong Uptrend"
        elif slope > 0.1:
            return "Mild Uptrend"
        elif slope < -0.5:
            return "Strong Downtrend"
        elif slope < -0.1:
            return "Mild Downtrend"
        else:
            return "Sideways"
    
    def _find_support_resistance(self, stock_data):
        """Find support and resistance levels"""
        # Use recent highs and lows
        recent_data = stock_data.tail(50)
        
        # Find local maxima and minima
        highs = recent_data['High'].rolling(window=5, center=True).max()
        lows = recent_data['Low'].rolling(window=5, center=True).min()
        
        resistance_levels = highs[highs == recent_data['High']].drop_duplicates().tail(3)
        support_levels = lows[lows == recent_data['Low']].drop_duplicates().tail(3)
        
        return {
            'resistance': resistance_levels.tolist() if not resistance_levels.empty else [],
            'support': support_levels.tolist() if not support_levels.empty else []
        }
    
    def _analyze_volume(self, stock_data):
        """Analyze volume patterns"""
        recent_volume = stock_data['Volume'].tail(20)
        avg_volume = recent_volume.mean()
        latest_volume = stock_data['Volume'].iloc[-1]
        
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 2:
            return "High Volume (Above Average)"
        elif volume_ratio > 1.5:
            return "Elevated Volume"
        elif volume_ratio < 0.5:
            return "Low Volume"
        else:
            return "Normal Volume"
    
    def _generate_trading_signals(self, stock_data):
        """Generate basic trading signals"""
        latest = stock_data.iloc[-1]
        signals = []
        
        # RSI signals
        if latest['RSI'] < 30:
            signals.append("Buy Signal: RSI Oversold")
        elif latest['RSI'] > 70:
            signals.append("Sell Signal: RSI Overbought")
        
        # Moving average signals
        if latest['Close'] > latest['MA_20'] > latest['MA_50']:
            signals.append("Buy Signal: Price Above Moving Averages")
        elif latest['Close'] < latest['MA_20'] < latest['MA_50']:
            signals.append("Sell Signal: Price Below Moving Averages")
        
        # Bollinger Band signals
        if latest['Close'] < latest['BB_lower']:
            signals.append("Buy Signal: Price Below Lower Bollinger Band")
        elif latest['Close'] > latest['BB_upper']:
            signals.append("Sell Signal: Price Above Upper Bollinger Band")
        
        return signals if signals else ["No Clear Signals"]
    
    def analyze_news_sentiment(self, symbol, days_back=7):
        """
        Analyze news sentiment for a given stock symbol
        
        Args:
            symbol (str): Stock ticker symbol
            days_back (int): Number of days to look back for news
        
        Returns:
            list: List of news items with sentiment scores
        """
        try:
            # Get company info for better search
            ticker = yf.Ticker(symbol)
            
            # Try to get company name
            try:
                info = ticker.info
                company_name = info.get('longName', symbol)
            except:
                company_name = symbol
            
            # Fetch news using yfinance (free alternative)
            news_data = self._fetch_yahoo_news(symbol, company_name, days_back)
            
            # If no news from yfinance, try News API if key is available
            if not news_data and self.news_api_key:
                news_data = self._fetch_news_api(symbol, company_name, days_back)
            
            # Analyze sentiment for each news item
            sentiment_results = []
            for news_item in news_data:
                sentiment_score = self._analyze_text_sentiment(news_item['title'])
                sentiment_results.append({
                    'headline': news_item['title'],
                    'date': news_item['date'],
                    'score': sentiment_score,
                    'url': news_item.get('url', ''),
                    'relevance': news_item.get('relevance', 1.0)
                })
            
            return sentiment_results
            
        except Exception as e:
            print(f"Error in news sentiment analysis: {str(e)}")
            return []
    
    def _fetch_yahoo_news(self, symbol, company_name, days_back):
        """Fetch news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_data = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for item in news[:10]:  # Limit to recent 10 items
                try:
                    # Convert timestamp to datetime
                    news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                    
                    if news_date >= cutoff_date:
                        news_data.append({
                            'title': item.get('title', ''),
                            'date': news_date.strftime('%Y-%m-%d'),
                            'url': item.get('link', ''),
                            'relevance': 1.0
                        })
                except:
                    continue
            
            return news_data
            
        except Exception as e:
            print(f"Error fetching Yahoo news: {str(e)}")
            return []
    
    def _fetch_news_api(self, symbol, company_name, days_back):
        """Fetch news from News API (if API key is available)"""
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            
            params = {
                'q': f"{company_name} OR {symbol}",
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                news_data = []
                
                for article in data.get('articles', [])[:20]:  # Limit to 20 articles
                    news_data.append({
                        'title': article.get('title', ''),
                        'date': article.get('publishedAt', '')[:10],  # Extract date part
                        'url': article.get('url', ''),
                        'relevance': 1.0
                    })
                
                return news_data
            else:
                print(f"News API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news from API: {str(e)}")
            return []
    
    def _analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text (str): Text to analyze
        
        Returns:
            float: Sentiment score (-1 to 1, negative to positive)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0
    
    def get_market_summary(self, symbols, period="1mo"):
        """
        Get market summary for multiple symbols
        
        Args:
            symbols (list): List of stock symbols
            period (str): Time period
        
        Returns:
            dict: Market summary data
        """
        summary = {}
        
        for symbol in symbols:
            try:
                stock_data = self.fetch_stock_data(symbol, period)
                if stock_data is not None:
                    latest = stock_data.iloc[-1]
                    first = stock_data.iloc[0]
                    
                    change_pct = ((latest['Close'] - first['Close']) / first['Close']) * 100
                    
                    summary[symbol] = {
                        'current_price': latest['Close'],
                        'change_percent': change_pct,
                        'volume': latest['Volume'],
                        'rsi': latest.get('RSI', 0),
                        'volatility': latest.get('Volatility', 0)
                    }
            except Exception as e:
                print(f"Error getting summary for {symbol}: {str(e)}")
                continue
        
        return summary
