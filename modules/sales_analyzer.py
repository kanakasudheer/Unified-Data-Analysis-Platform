import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesAnalyzer:
    """Sales data analysis and anomaly detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    def analyze_sales_trends(self, sales_data, period="weekly"):
        """
        Analyze sales trends and patterns
        
        Args:
            sales_data (pandas.DataFrame): Processed sales data
            period (str): Analysis period (daily, weekly, monthly)
        
        Returns:
            dict: Sales trend analysis results
        """
        try:
            # Ensure date column is datetime
            sales_data['date'] = pd.to_datetime(sales_data['date'])
            
            # Basic statistics
            analysis = {
                'total_revenue': sales_data['revenue'].sum(),
                'avg_revenue': sales_data['revenue'].mean(),
                'median_revenue': sales_data['revenue'].median(),
                'std_revenue': sales_data['revenue'].std(),
                'min_revenue': sales_data['revenue'].min(),
                'max_revenue': sales_data['revenue'].max(),
                'data_points': len(sales_data)
            }
            
            # Time-based analysis
            analysis.update(self._analyze_time_patterns(sales_data, period))
            
            # Growth analysis
            analysis.update(self._calculate_growth_metrics(sales_data, period))
            
            # Seasonal patterns
            analysis.update(self._analyze_seasonal_patterns(sales_data))
            
            # Performance metrics
            analysis.update(self._calculate_performance_metrics(sales_data))
            
            return analysis
            
        except Exception as e:
            print(f"Error in sales trend analysis: {str(e)}")
            return {}
    
    def _analyze_time_patterns(self, sales_data, period):
        """Analyze time-based patterns in sales data"""
        patterns = {}
        
        try:
            # Set date as index for easier time-based operations
            df = sales_data.set_index('date').sort_index()
            
            if period.lower() == "daily":
                # Daily patterns
                daily_avg = df['revenue'].resample('D').sum().mean()
                patterns['daily_average'] = daily_avg
                patterns['best_day'] = df['revenue'].resample('D').sum().idxmax()
                patterns['worst_day'] = df['revenue'].resample('D').sum().idxmin()
                
            elif period.lower() == "weekly":
                # Weekly patterns
                weekly_data = df['revenue'].resample('W').sum()
                patterns['weekly_average'] = weekly_data.mean()
                patterns['best_week'] = weekly_data.idxmax()
                patterns['worst_week'] = weekly_data.idxmin()
                patterns['weekly_growth'] = self._calculate_period_growth(weekly_data)
                
            elif period.lower() == "monthly":
                # Monthly patterns
                monthly_data = df['revenue'].resample('M').sum()
                patterns['monthly_average'] = monthly_data.mean()
                patterns['best_month'] = monthly_data.idxmax()
                patterns['worst_month'] = monthly_data.idxmin()
                patterns['monthly_growth'] = self._calculate_period_growth(monthly_data)
            
            # Day of week analysis
            df['day_of_week'] = df.index.day_name()
            dow_analysis = df.groupby('day_of_week')['revenue'].agg(['mean', 'sum', 'count'])
            patterns['day_of_week_performance'] = dow_analysis.to_dict()
            
        except Exception as e:
            print(f"Error in time pattern analysis: {str(e)}")
        
        return patterns
    
    def _calculate_growth_metrics(self, sales_data, period):
        """Calculate various growth metrics"""
        growth_metrics = {}
        
        try:
            df = sales_data.set_index('date').sort_index()
            
            # Overall growth rate
            if len(df) > 1:
                first_value = df['revenue'].iloc[0]
                last_value = df['revenue'].iloc[-1]
                total_days = (df.index[-1] - df.index[0]).days
                
                if first_value > 0 and total_days > 0:
                    total_growth = ((last_value - first_value) / first_value) * 100
                    annualized_growth = ((last_value / first_value) ** (365 / total_days) - 1) * 100
                    
                    growth_metrics['total_growth_rate'] = total_growth
                    growth_metrics['annualized_growth_rate'] = annualized_growth
            
            # Period-over-period growth
            if period.lower() == "weekly":
                weekly_data = df['revenue'].resample('W').sum()
                growth_metrics['avg_weekly_growth'] = self._calculate_avg_growth(weekly_data)
            elif period.lower() == "monthly":
                monthly_data = df['revenue'].resample('M').sum()
                growth_metrics['avg_monthly_growth'] = self._calculate_avg_growth(monthly_data)
            
            # Moving averages
            df['ma_7'] = df['revenue'].rolling(window=7).mean()
            df['ma_30'] = df['revenue'].rolling(window=30).mean()
            
            growth_metrics['current_vs_ma7'] = ((df['revenue'].iloc[-1] - df['ma_7'].iloc[-1]) / df['ma_7'].iloc[-1]) * 100 if not pd.isna(df['ma_7'].iloc[-1]) else 0
            growth_metrics['current_vs_ma30'] = ((df['revenue'].iloc[-1] - df['ma_30'].iloc[-1]) / df['ma_30'].iloc[-1]) * 100 if not pd.isna(df['ma_30'].iloc[-1]) else 0
            
        except Exception as e:
            print(f"Error calculating growth metrics: {str(e)}")
        
        return growth_metrics
    
    def _calculate_period_growth(self, series):
        """Calculate period-over-period growth rates"""
        if len(series) < 2:
            return 0
        
        growth_rates = series.pct_change().dropna() * 100
        return growth_rates.mean()
    
    def _calculate_avg_growth(self, series):
        """Calculate average growth rate for a series"""
        if len(series) < 2:
            return 0
        
        growth_rates = []
        for i in range(1, len(series)):
            if series.iloc[i-1] > 0:
                growth = ((series.iloc[i] - series.iloc[i-1]) / series.iloc[i-1]) * 100
                growth_rates.append(growth)
        
        return np.mean(growth_rates) if growth_rates else 0
    
    def _analyze_seasonal_patterns(self, sales_data):
        """Analyze seasonal patterns in sales data"""
        seasonal = {}
        
        try:
            df = sales_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.day_name()
            
            # Monthly patterns
            monthly_avg = df.groupby('month')['revenue'].mean()
            seasonal['monthly_patterns'] = {
                'peak_month': monthly_avg.idxmax(),
                'lowest_month': monthly_avg.idxmin(),
                'monthly_averages': monthly_avg.to_dict()
            }
            
            # Quarterly patterns
            quarterly_avg = df.groupby('quarter')['revenue'].mean()
            seasonal['quarterly_patterns'] = {
                'peak_quarter': quarterly_avg.idxmax(),
                'lowest_quarter': quarterly_avg.idxmin(),
                'quarterly_averages': quarterly_avg.to_dict()
            }
            
            # Day of week patterns
            dow_avg = df.groupby('day_of_week')['revenue'].mean()
            seasonal['day_of_week_patterns'] = {
                'peak_day': dow_avg.idxmax(),
                'lowest_day': dow_avg.idxmin(),
                'daily_averages': dow_avg.to_dict()
            }
            
        except Exception as e:
            print(f"Error in seasonal analysis: {str(e)}")
        
        return seasonal
    
    def _calculate_performance_metrics(self, sales_data):
        """Calculate key performance metrics"""
        metrics = {}
        
        try:
            # Revenue metrics
            metrics['coefficient_of_variation'] = (sales_data['revenue'].std() / sales_data['revenue'].mean()) * 100
            
            # Percentile analysis
            metrics['percentiles'] = {
                '25th': sales_data['revenue'].quantile(0.25),
                '50th': sales_data['revenue'].quantile(0.50),
                '75th': sales_data['revenue'].quantile(0.75),
                '90th': sales_data['revenue'].quantile(0.90),
                '95th': sales_data['revenue'].quantile(0.95)
            }
            
            # Consistency metrics
            recent_data = sales_data.tail(30)  # Last 30 records
            if len(recent_data) > 1:
                metrics['recent_consistency'] = 1 - (recent_data['revenue'].std() / recent_data['revenue'].mean())
            
            # Trend direction
            if len(sales_data) > 10:
                recent_trend = np.polyfit(range(len(sales_data.tail(10))), sales_data.tail(10)['revenue'], 1)[0]
                metrics['trend_direction'] = 'Increasing' if recent_trend > 0 else 'Decreasing'
                metrics['trend_strength'] = abs(recent_trend)
            
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
        
        return metrics
    
    def detect_anomalies(self, sales_data, contamination=0.1):
        """
        Detect anomalies in sales data using Isolation Forest
        
        Args:
            sales_data (pandas.DataFrame): Sales data with date and revenue columns
            contamination (float): Expected proportion of anomalies (0.1 = 10%)
        
        Returns:
            pandas.DataFrame: Detected anomalies
        """
        try:
            if len(sales_data) < 10:
                print("Insufficient data for anomaly detection")
                return pd.DataFrame()
            
            # Prepare features for anomaly detection
            df = sales_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Feature engineering
            features = self._engineer_anomaly_features(df)
            
            if features.empty:
                return pd.DataFrame()
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Configure and fit anomaly detector
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict anomalies
            anomaly_labels = detector.fit_predict(features_scaled)
            anomaly_scores = detector.score_samples(features_scaled)
            
            # Add results to dataframe
            df['anomaly'] = anomaly_labels
            df['anomaly_score'] = anomaly_scores
            
            # Filter anomalies (label = -1)
            anomalies = df[df['anomaly'] == -1].copy()
            
            if not anomalies.empty:
                # Add anomaly severity
                anomalies['severity'] = self._classify_anomaly_severity(anomalies['anomaly_score'])
                
                # Add contextual information
                anomalies = self._add_anomaly_context(anomalies, df)
            
            return anomalies[['date', 'revenue', 'anomaly_score', 'severity']].reset_index(drop=True)
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            return pd.DataFrame()
    
    def _engineer_anomaly_features(self, df):
        """Engineer features for anomaly detection"""
        try:
            features_df = pd.DataFrame()
            
            # Basic revenue feature
            features_df['revenue'] = df['revenue']
            
            # Time-based features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day_of_month'] = df['date'].dt.day
            
            features_df['day_of_week'] = df['day_of_week']
            features_df['month'] = df['month']
            features_df['day_of_month'] = df['day_of_month']
            
            # Rolling statistics
            if len(df) >= 7:
                features_df['rolling_mean_7'] = df['revenue'].rolling(window=7, min_periods=1).mean()
                features_df['rolling_std_7'] = df['revenue'].rolling(window=7, min_periods=1).std().fillna(0)
                features_df['revenue_vs_mean_7'] = df['revenue'] / features_df['rolling_mean_7']
            
            if len(df) >= 30:
                features_df['rolling_mean_30'] = df['revenue'].rolling(window=30, min_periods=1).mean()
                features_df['revenue_vs_mean_30'] = df['revenue'] / features_df['rolling_mean_30']
            
            # Lag features
            if len(df) >= 2:
                features_df['revenue_lag_1'] = df['revenue'].shift(1).fillna(df['revenue'].iloc[0])
                features_df['revenue_change'] = df['revenue'] - features_df['revenue_lag_1']
                features_df['revenue_pct_change'] = df['revenue'].pct_change().fillna(0)
            
            # Remove any infinite or NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            return pd.DataFrame()
    
    def _classify_anomaly_severity(self, scores):
        """Classify anomaly severity based on scores"""
        # Isolation Forest scores are typically between -1 and 1
        # More negative scores indicate stronger anomalies
        severity = []
        
        for score in scores:
            if score < -0.5:
                severity.append('High')
            elif score < -0.3:
                severity.append('Medium')
            else:
                severity.append('Low')
        
        return severity
    
    def _add_anomaly_context(self, anomalies, full_data):
        """Add contextual information to anomalies"""
        try:
            # Calculate how much each anomaly deviates from normal
            mean_revenue = full_data['revenue'].mean()
            std_revenue = full_data['revenue'].std()
            
            anomalies = anomalies.copy()
            anomalies['deviation_from_mean'] = ((anomalies['revenue'] - mean_revenue) / std_revenue).round(2)
            anomalies['is_high_anomaly'] = anomalies['revenue'] > (mean_revenue + 2 * std_revenue)
            anomalies['is_low_anomaly'] = anomalies['revenue'] < (mean_revenue - 2 * std_revenue)
            
            return anomalies
            
        except Exception as e:
            print(f"Error adding anomaly context: {str(e)}")
            return anomalies
    
    def generate_sales_insights(self, analysis_results, anomalies):
        """
        Generate actionable insights from sales analysis
        
        Args:
            analysis_results (dict): Results from sales trend analysis
            anomalies (pandas.DataFrame): Detected anomalies
        
        Returns:
            dict: Actionable insights and recommendations
        """
        insights = {
            'key_findings': [],
            'recommendations': [],
            'alerts': [],
            'opportunities': []
        }
        
        try:
            # Revenue insights
            if 'total_revenue' in analysis_results:
                total_revenue = analysis_results['total_revenue']
                avg_revenue = analysis_results['avg_revenue']
                
                insights['key_findings'].append(f"Total revenue: ${total_revenue:,.2f}")
                insights['key_findings'].append(f"Average daily revenue: ${avg_revenue:,.2f}")
            
            # Growth insights
            if 'total_growth_rate' in analysis_results:
                growth_rate = analysis_results['total_growth_rate']
                if growth_rate > 10:
                    insights['opportunities'].append(f"Strong growth of {growth_rate:.1f}% - consider scaling operations")
                elif growth_rate < -10:
                    insights['alerts'].append(f"Declining revenue by {abs(growth_rate):.1f}% - investigate causes")
            
            # Seasonal insights
            if 'seasonal' in analysis_results and 'monthly_patterns' in analysis_results['seasonal']:
                peak_month = analysis_results['seasonal']['monthly_patterns'].get('peak_month')
                if peak_month:
                    insights['opportunities'].append(f"Peak performance in month {peak_month} - prepare for seasonal demand")
            
            # Anomaly insights
            if not anomalies.empty:
                high_severity_count = len(anomalies[anomalies['severity'] == 'High'])
                if high_severity_count > 0:
                    insights['alerts'].append(f"{high_severity_count} high-severity anomalies detected - require immediate attention")
                
                # Recent anomalies
                recent_anomalies = anomalies[pd.to_datetime(anomalies['date']) >= datetime.now() - timedelta(days=7)]
                if not recent_anomalies.empty:
                    insights['alerts'].append(f"{len(recent_anomalies)} anomalies in the last 7 days")
            
            # Performance insights
            if 'coefficient_of_variation' in analysis_results:
                cv = analysis_results['coefficient_of_variation']
                if cv > 50:
                    insights['recommendations'].append("High revenue variability - consider stabilizing factors")
                elif cv < 20:
                    insights['key_findings'].append("Consistent revenue performance")
            
            # Trend insights
            if 'trend_direction' in analysis_results:
                direction = analysis_results['trend_direction']
                if direction == 'Increasing':
                    insights['opportunities'].append("Positive trend detected - good time for investment")
                elif direction == 'Decreasing':
                    insights['recommendations'].append("Negative trend - implement improvement strategies")
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
        
        return insights
    
    def forecast_sales(self, sales_data, periods=30):
        """
        Simple sales forecasting using moving averages and trend analysis
        
        Args:
            sales_data (pandas.DataFrame): Historical sales data
            periods (int): Number of periods to forecast
        
        Returns:
            pandas.DataFrame: Forecasted sales data
        """
        try:
            df = sales_data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate moving averages and trend
            df['ma_7'] = df['revenue'].rolling(window=7, min_periods=1).mean()
            df['ma_30'] = df['revenue'].rolling(window=30, min_periods=1).mean()
            
            # Simple linear trend
            if len(df) >= 10:
                x = np.arange(len(df))
                y = df['revenue'].values
                trend_coef = np.polyfit(x, y, 1)
                
                # Generate forecast dates
                last_date = df['date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
                
                # Simple forecast using trend and seasonal adjustment
                base_forecast = trend_coef[0] * np.arange(len(df), len(df) + periods) + trend_coef[1]
                
                # Add seasonal adjustment based on day of week
                seasonal_factors = df.groupby(df['date'].dt.day_name())['revenue'].mean() / df['revenue'].mean()
                
                forecast_values = []
                for i, date in enumerate(forecast_dates):
                    day_name = date.day_name()
                    seasonal_factor = seasonal_factors.get(day_name, 1.0)
                    forecast_value = base_forecast[i] * seasonal_factor
                    forecast_values.append(max(0, forecast_value))  # Ensure non-negative
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecasted_revenue': forecast_values,
                    'forecast_type': 'Trend + Seasonal'
                })
                
                return forecast_df
            
            else:
                # Insufficient data - use simple average
                avg_revenue = df['revenue'].mean()
                last_date = df['date'].iloc[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecasted_revenue': [avg_revenue] * periods,
                    'forecast_type': 'Average'
                })
                
                return forecast_df
                
        except Exception as e:
            print(f"Error in sales forecasting: {str(e)}")
            return pd.DataFrame()
