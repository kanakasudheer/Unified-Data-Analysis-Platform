import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

class DataProcessor:
    """Data processing and validation utilities"""
    
    def __init__(self):
        self.required_sales_columns = ['date', 'revenue']
        self.date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S'
        ]
    
    def validate_sales_data(self, data):
        """
        Validate sales data for completeness and correctness
        
        Args:
            data (pandas.DataFrame): Raw sales data
        
        Returns:
            dict: Validation results with is_valid flag and issues list
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            # Check if data is not empty
            if data.empty:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Data is empty")
                return validation_result
            
            # Check for required columns
            available_columns = data.columns.tolist()
            validation_result['summary']['available_columns'] = available_columns
            
            # Try to identify date and revenue columns
            date_column = self._identify_date_column(data)
            revenue_column = self._identify_revenue_column(data)
            
            if not date_column:
                validation_result['issues'].append("No date column found")
                validation_result['is_valid'] = False
            else:
                validation_result['summary']['date_column'] = date_column
            
            if not revenue_column:
                validation_result['issues'].append("No revenue/sales column found")
                validation_result['is_valid'] = False
            else:
                validation_result['summary']['revenue_column'] = revenue_column
            
            # If we have both columns, perform detailed validation
            if date_column and revenue_column:
                validation_result.update(self._validate_date_column(data, date_column))
                validation_result.update(self._validate_revenue_column(data, revenue_column))
                validation_result.update(self._check_data_quality(data, date_column, revenue_column))
            
            # Summary statistics
            validation_result['summary']['total_rows'] = len(data)
            validation_result['summary']['total_columns'] = len(data.columns)
            
            if validation_result['issues']:
                validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _identify_date_column(self, data):
        """Identify the date column in the data"""
        # Common date column names
        date_keywords = ['date', 'time', 'timestamp', 'created', 'order_date', 'sale_date', 'transaction_date']
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Check if column name contains date keywords
            if any(keyword in col_lower for keyword in date_keywords):
                return col
            
            # Check if column contains date-like data
            if self._is_date_column(data[col]):
                return col
        
        return None
    
    def _identify_revenue_column(self, data):
        """Identify the revenue/sales column in the data"""
        # Common revenue column names
        revenue_keywords = ['revenue', 'sales', 'amount', 'total', 'price', 'value', 'income', 'earnings']
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Check if column name contains revenue keywords
            if any(keyword in col_lower for keyword in revenue_keywords):
                # Verify it's numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    return col
        
        # If no keyword match, look for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            return numeric_columns[0]  # Return first numeric column
        
        return None
    
    def _is_date_column(self, series):
        """Check if a series contains date-like data"""
        try:
            # Try to parse a sample of the data
            sample_size = min(10, len(series))
            sample = series.dropna().head(sample_size)
            
            if sample.empty:
                return False
            
            # Check if it's already datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                return True
            
            # Try to parse string dates
            if series.dtype == 'object':
                parsed_count = 0
                for value in sample:
                    if self._try_parse_date(str(value)):
                        parsed_count += 1
                
                # If more than 70% can be parsed as dates, consider it a date column
                return (parsed_count / len(sample)) > 0.7
            
        except:
            pass
        
        return False
    
    def _try_parse_date(self, date_string):
        """Try to parse a date string using common formats"""
        for fmt in self.date_formats:
            try:
                datetime.strptime(date_string.strip(), fmt)
                return True
            except:
                continue
        
        # Try pandas date parsing
        try:
            pd.to_datetime(date_string)
            return True
        except:
            pass
        
        return False
    
    def _validate_date_column(self, data, date_column):
        """Validate the date column"""
        result = {'issues': [], 'warnings': []}
        
        try:
            # Check for missing values
            missing_dates = data[date_column].isna().sum()
            if missing_dates > 0:
                result['warnings'].append(f"{missing_dates} missing dates found")
            
            # Try to convert to datetime
            try:
                converted_dates = pd.to_datetime(data[date_column], errors='coerce')
                invalid_dates = converted_dates.isna().sum() - missing_dates
                
                if invalid_dates > 0:
                    result['issues'].append(f"{invalid_dates} invalid date formats found")
                
                # Check date range reasonableness
                valid_dates = converted_dates.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    # Check if dates are in reasonable range (not too far in past/future)
                    current_year = datetime.now().year
                    if min_date.year < current_year - 10:
                        result['warnings'].append(f"Dates go back to {min_date.year} - very old data")
                    if max_date.year > current_year + 1:
                        result['warnings'].append(f"Future dates found: {max_date}")
                
            except Exception as e:
                result['issues'].append(f"Cannot parse dates: {str(e)}")
        
        except Exception as e:
            result['issues'].append(f"Date validation error: {str(e)}")
        
        return result
    
    def _validate_revenue_column(self, data, revenue_column):
        """Validate the revenue column"""
        result = {'issues': [], 'warnings': []}
        
        try:
            revenue_data = data[revenue_column]
            
            # Check for missing values
            missing_revenue = revenue_data.isna().sum()
            if missing_revenue > 0:
                result['warnings'].append(f"{missing_revenue} missing revenue values found")
            
            # Check if numeric
            if not pd.api.types.is_numeric_dtype(revenue_data):
                # Try to convert to numeric
                try:
                    numeric_revenue = pd.to_numeric(revenue_data, errors='coerce')
                    conversion_failures = numeric_revenue.isna().sum() - missing_revenue
                    
                    if conversion_failures > 0:
                        result['issues'].append(f"{conversion_failures} non-numeric revenue values found")
                    
                    revenue_data = numeric_revenue
                except Exception as e:
                    result['issues'].append(f"Cannot convert revenue to numeric: {str(e)}")
                    return result
            
            # Check for negative values
            valid_revenue = revenue_data.dropna()
            if not valid_revenue.empty:
                negative_count = (valid_revenue < 0).sum()
                if negative_count > 0:
                    result['warnings'].append(f"{negative_count} negative revenue values found")
                
                # Check for zero values
                zero_count = (valid_revenue == 0).sum()
                if zero_count > len(valid_revenue) * 0.1:  # More than 10% zeros
                    result['warnings'].append(f"High number of zero revenue values: {zero_count}")
                
                # Check for outliers
                if len(valid_revenue) > 10:
                    q1 = valid_revenue.quantile(0.25)
                    q3 = valid_revenue.quantile(0.75)
                    iqr = q3 - q1
                    outlier_threshold = q3 + 1.5 * iqr
                    outliers = (valid_revenue > outlier_threshold).sum()
                    
                    if outliers > 0:
                        result['warnings'].append(f"{outliers} potential revenue outliers detected")
        
        except Exception as e:
            result['issues'].append(f"Revenue validation error: {str(e)}")
        
        return result
    
    def _check_data_quality(self, data, date_column, revenue_column):
        """Check overall data quality"""
        result = {'issues': [], 'warnings': []}
        
        try:
            # Check for duplicates
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                result['warnings'].append(f"{duplicates} duplicate rows found")
            
            # Check date duplicates (if dates should be unique)
            date_duplicates = data[date_column].duplicated().sum()
            if date_duplicates > 0:
                result['warnings'].append(f"{date_duplicates} duplicate dates found")
            
            # Check data coverage (gaps in dates)
            try:
                dates = pd.to_datetime(data[date_column], errors='coerce').dropna()
                if len(dates) > 1:
                    date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
                    missing_dates = len(date_range) - len(dates.unique())
                    
                    if missing_dates > 0:
                        result['warnings'].append(f"{missing_dates} days missing in date range")
            except:
                pass
            
        except Exception as e:
            result['warnings'].append(f"Data quality check error: {str(e)}")
        
        return result
    
    def process_sales_data(self, data, date_column=None, revenue_column=None):
        """
        Process and clean sales data for analysis
        
        Args:
            data (pandas.DataFrame): Raw sales data
            date_column (str): Name of date column (auto-detected if None)
            revenue_column (str): Name of revenue column (auto-detected if None)
        
        Returns:
            pandas.DataFrame: Processed sales data
        """
        try:
            # Auto-detect columns if not provided
            if not date_column:
                date_column = self._identify_date_column(data)
            if not revenue_column:
                revenue_column = self._identify_revenue_column(data)
            
            if not date_column or not revenue_column:
                raise ValueError("Cannot identify required date and revenue columns")
            
            # Create a copy for processing
            processed_data = data[[date_column, revenue_column]].copy()
            
            # Rename columns to standard names
            processed_data.columns = ['date', 'revenue']
            
            # Convert date column
            processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')
            
            # Convert revenue column
            processed_data['revenue'] = pd.to_numeric(processed_data['revenue'], errors='coerce')
            
            # Remove rows with invalid dates or revenue
            initial_count = len(processed_data)
            processed_data = processed_data.dropna()
            removed_count = initial_count - len(processed_data)
            
            if removed_count > 0:
                print(f"Removed {removed_count} rows with invalid data")
            
            # Sort by date
            processed_data = processed_data.sort_values('date').reset_index(drop=True)
            
            # Remove duplicates (keep last occurrence)
            processed_data = processed_data.drop_duplicates(subset=['date'], keep='last')
            
            # Handle negative revenues (convert to zero or remove based on preference)
            negative_revenues = (processed_data['revenue'] < 0).sum()
            if negative_revenues > 0:
                print(f"Found {negative_revenues} negative revenue values, setting to zero")
                processed_data['revenue'] = processed_data['revenue'].clip(lower=0)
            
            # Add derived columns for analysis
            processed_data = self._add_derived_columns(processed_data)
            
            return processed_data
            
        except Exception as e:
            print(f"Error processing sales data: {str(e)}")
            return pd.DataFrame()
    
    def _add_derived_columns(self, data):
        """Add derived columns for enhanced analysis"""
        try:
            # Time-based features
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['day'] = data['date'].dt.day
            data['day_of_week'] = data['date'].dt.dayofweek
            data['day_name'] = data['date'].dt.day_name()
            data['month_name'] = data['date'].dt.month_name()
            data['quarter'] = data['date'].dt.quarter
            data['week_of_year'] = data['date'].dt.isocalendar().week
            
            # Moving averages (if enough data)
            if len(data) >= 7:
                data['revenue_ma_7'] = data['revenue'].rolling(window=7, min_periods=1).mean()
            
            if len(data) >= 30:
                data['revenue_ma_30'] = data['revenue'].rolling(window=30, min_periods=1).mean()
            
            # Revenue changes
            data['revenue_change'] = data['revenue'].diff()
            data['revenue_pct_change'] = data['revenue'].pct_change().fillna(0)
            
            # Cumulative revenue
            data['cumulative_revenue'] = data['revenue'].cumsum()
            
            return data
            
        except Exception as e:
            print(f"Error adding derived columns: {str(e)}")
            return data
    
    def aggregate_data(self, data, frequency='D'):
        """
        Aggregate data by specified frequency
        
        Args:
            data (pandas.DataFrame): Processed sales data
            frequency (str): Aggregation frequency ('D', 'W', 'M', 'Q', 'Y')
        
        Returns:
            pandas.DataFrame: Aggregated data
        """
        try:
            # Set date as index for resampling
            df = data.set_index('date')
            
            # Aggregate revenue by frequency
            aggregated = df['revenue'].resample(frequency).agg({
                'total_revenue': 'sum',
                'avg_revenue': 'mean',
                'min_revenue': 'min',
                'max_revenue': 'max',
                'count': 'count'
            }).reset_index()
            
            # Calculate derived metrics
            aggregated['revenue_std'] = df['revenue'].resample(frequency).std().values
            aggregated['revenue_median'] = df['revenue'].resample(frequency).median().values
            
            # Add period information
            aggregated['period'] = aggregated['date'].dt.to_period(frequency)
            
            return aggregated
            
        except Exception as e:
            print(f"Error aggregating data: {str(e)}")
            return pd.DataFrame()
    
    def clean_outliers(self, data, method='iqr', factor=1.5):
        """
        Clean outliers from revenue data
        
        Args:
            data (pandas.DataFrame): Sales data
            method (str): Outlier detection method ('iqr', 'zscore', 'isolation')
            factor (float): Outlier threshold factor
        
        Returns:
            pandas.DataFrame: Data with outliers handled
        """
        try:
            cleaned_data = data.copy()
            revenue = cleaned_data['revenue']
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = revenue.quantile(0.25)
                Q3 = revenue.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Identify outliers
                outliers = (revenue < lower_bound) | (revenue > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((revenue - revenue.mean()) / revenue.std())
                outliers = z_scores > factor
                
            else:
                print(f"Unknown outlier detection method: {method}")
                return cleaned_data
            
            # Report outliers
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"Detected {outlier_count} outliers using {method} method")
                
                # Option 1: Remove outliers
                # cleaned_data = cleaned_data[~outliers]
                
                # Option 2: Cap outliers (preferred for sales data)
                if method == 'iqr':
                    cleaned_data.loc[outliers & (revenue < lower_bound), 'revenue'] = lower_bound
                    cleaned_data.loc[outliers & (revenue > upper_bound), 'revenue'] = upper_bound
                elif method == 'zscore':
                    # Cap at mean ± factor * std
                    mean_rev = revenue.mean()
                    std_rev = revenue.std()
                    cleaned_data.loc[outliers, 'revenue'] = np.clip(
                        cleaned_data.loc[outliers, 'revenue'],
                        mean_rev - factor * std_rev,
                        mean_rev + factor * std_rev
                    )
            
            return cleaned_data
            
        except Exception as e:
            print(f"Error cleaning outliers: {str(e)}")
            return data
    
    def fill_missing_dates(self, data, fill_method='interpolate'):
        """
        Fill missing dates in the data
        
        Args:
            data (pandas.DataFrame): Sales data with potential date gaps
            fill_method (str): Method to fill missing values ('zero', 'interpolate', 'forward_fill')
        
        Returns:
            pandas.DataFrame: Data with filled date gaps
        """
        try:
            if data.empty:
                return data
            
            # Create complete date range
            start_date = data['date'].min()
            end_date = data['date'].max()
            complete_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create complete dataframe
            complete_df = pd.DataFrame({'date': complete_dates})
            
            # Merge with existing data
            filled_data = complete_df.merge(data, on='date', how='left')
            
            # Fill missing revenue values
            if fill_method == 'zero':
                filled_data['revenue'] = filled_data['revenue'].fillna(0)
            elif fill_method == 'interpolate':
                filled_data['revenue'] = filled_data['revenue'].interpolate(method='linear')
            elif fill_method == 'forward_fill':
                filled_data['revenue'] = filled_data['revenue'].fillna(method='ffill')
            else:
                filled_data['revenue'] = filled_data['revenue'].fillna(0)
            
            # Re-add derived columns
            filled_data = self._add_derived_columns(filled_data[['date', 'revenue']])
            
            return filled_data
            
        except Exception as e:
            print(f"Error filling missing dates: {str(e)}")
            return data
