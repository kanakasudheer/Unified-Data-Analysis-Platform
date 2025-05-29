import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def format_currency(amount, currency_symbol="$"):
    """
    Format a number as currency
    
    Args:
        amount (float): Amount to format
        currency_symbol (str): Currency symbol to use
    
    Returns:
        str: Formatted currency string
    """
    try:
        if pd.isna(amount) or amount is None:
            return f"{currency_symbol}0.00"
        
        # Handle very large numbers
        if abs(amount) >= 1_000_000_000:
            return f"{currency_symbol}{amount/1_000_000_000:.2f}B"
        elif abs(amount) >= 1_000_000:
            return f"{currency_symbol}{amount/1_000_000:.2f}M"
        elif abs(amount) >= 1_000:
            return f"{currency_symbol}{amount/1_000:.2f}K"
        else:
            return f"{currency_symbol}{amount:.2f}"
    except:
        return f"{currency_symbol}0.00"

def format_percentage(value, decimal_places=2):
    """
    Format a number as percentage
    
    Args:
        value (float): Value to format (e.g., 0.15 for 15%)
        decimal_places (int): Number of decimal places
    
    Returns:
        str: Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None:
            return "0.00%"
        return f"{value:.{decimal_places}f}%"
    except:
        return "0.00%"

def format_number(value, decimal_places=2, use_commas=True):
    """
    Format a number with proper formatting
    
    Args:
        value (float): Number to format
        decimal_places (int): Number of decimal places
        use_commas (bool): Whether to use comma separators
    
    Returns:
        str: Formatted number string
    """
    try:
        if pd.isna(value) or value is None:
            return "0"
        
        if use_commas:
            return f"{value:,.{decimal_places}f}"
        else:
            return f"{value:.{decimal_places}f}"
    except:
        return "0"

def get_date_range_options():
    """
    Get common date range options for analysis
    
    Returns:
        dict: Dictionary of date range options
    """
    today = datetime.now().date()
    
    return {
        "Last 7 days": (today - timedelta(days=7), today),
        "Last 30 days": (today - timedelta(days=30), today),
        "Last 3 months": (today - timedelta(days=90), today),
        "Last 6 months": (today - timedelta(days=180), today),
        "Last year": (today - timedelta(days=365), today),
        "Year to date": (datetime(today.year, 1, 1).date(), today),
        "All time": (None, None)
    }

def calculate_growth_rate(current_value, previous_value):
    """
    Calculate growth rate between two values
    
    Args:
        current_value (float): Current period value
        previous_value (float): Previous period value
    
    Returns:
        float: Growth rate as percentage
    """
    try:
        if previous_value == 0 or pd.isna(previous_value) or pd.isna(current_value):
            return 0.0
        
        return ((current_value - previous_value) / previous_value) * 100
    except:
        return 0.0

def calculate_moving_average(data, window_size):
    """
    Calculate moving average for a data series
    
    Args:
        data (pandas.Series): Data series
        window_size (int): Window size for moving average
    
    Returns:
        pandas.Series: Moving average series
    """
    try:
        return data.rolling(window=window_size, min_periods=1).mean()
    except:
        return pd.Series([0] * len(data))

def detect_trend_direction(data, window_size=10):
    """
    Detect trend direction using linear regression
    
    Args:
        data (pandas.Series): Data series
        window_size (int): Number of recent points to consider
    
    Returns:
        str: Trend direction ('Increasing', 'Decreasing', 'Stable')
    """
    try:
        if len(data) < window_size:
            return "Insufficient Data"
        
        recent_data = data.tail(window_size)
        x = np.arange(len(recent_data))
        y = recent_data.values
        
        # Calculate slope using linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend based on slope
        if slope > 0.01:  # Threshold for increasing trend
            return "Increasing"
        elif slope < -0.01:  # Threshold for decreasing trend
            return "Decreasing"
        else:
            return "Stable"
    except:
        return "Unknown"

def calculate_volatility(data, window_size=20):
    """
    Calculate volatility (standard deviation) of data
    
    Args:
        data (pandas.Series): Price or value data
        window_size (int): Window size for calculation
    
    Returns:
        float: Volatility as percentage
    """
    try:
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=window_size).std().iloc[-1]
        
        # Annualize and convert to percentage
        annualized_volatility = volatility * np.sqrt(252) * 100
        
        return annualized_volatility if not pd.isna(annualized_volatility) else 0.0
    except:
        return 0.0

def create_summary_metrics(data, value_column):
    """
    Create summary metrics for a data series
    
    Args:
        data (pandas.DataFrame): Data with date and value columns
        value_column (str): Name of the value column
    
    Returns:
        dict: Summary metrics
    """
    try:
        values = data[value_column].dropna()
        
        if values.empty:
            return {}
        
        metrics = {
            'count': len(values),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'sum': values.sum(),
            'q25': values.quantile(0.25),
            'q75': values.quantile(0.75),
            'coefficient_of_variation': (values.std() / values.mean()) * 100 if values.mean() != 0 else 0
        }
        
        return metrics
    except:
        return {}

def create_correlation_matrix(data, columns):
    """
    Create correlation matrix for specified columns
    
    Args:
        data (pandas.DataFrame): Data
        columns (list): List of column names
    
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    try:
        numeric_data = data[columns].select_dtypes(include=[np.number])
        return numeric_data.corr()
    except:
        return pd.DataFrame()

def identify_outliers(data, column, method='iqr', factor=1.5):
    """
    Identify outliers in data using specified method
    
    Args:
        data (pandas.DataFrame): Data
        column (str): Column name to analyze
        method (str): Method to use ('iqr', 'zscore')
        factor (float): Threshold factor
    
    Returns:
        pandas.Series: Boolean series indicating outliers
    """
    try:
        values = data[column].dropna()
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data[column] - values.mean()) / values.std())
            outliers = z_scores > factor
            
        else:
            return pd.Series([False] * len(data))
        
        return outliers.fillna(False)
    except:
        return pd.Series([False] * len(data))

def create_time_series_chart(data, x_column, y_column, title="Time Series Chart"):
    """
    Create a time series chart using Plotly
    
    Args:
        data (pandas.DataFrame): Data with time series
        x_column (str): X-axis column (usually date/time)
        y_column (str): Y-axis column (values)
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        fig = px.line(data, x=x_column, y=y_column, title=title)
        
        # Customize layout
        fig.update_layout(
            xaxis_title=x_column.title(),
            yaxis_title=y_column.title(),
            hovermode='x unified',
            showlegend=True
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate=f'<b>{y_column.title()}</b>: %{{y}}<br>' +
                         f'<b>{x_column.title()}</b>: %{{x}}<br>' +
                         '<extra></extra>'
        )
        
        return fig
    except Exception as e:
        # Return empty figure if error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_distribution_chart(data, column, title="Distribution Chart"):
    """
    Create a distribution chart (histogram) using Plotly
    
    Args:
        data (pandas.DataFrame): Data
        column (str): Column to analyze
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        fig = px.histogram(data, x=column, title=title, nbins=30)
        
        # Add statistics
        mean_val = data[column].mean()
        median_val = data[column].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                     annotation_text=f"Median: {median_val:.2f}")
        
        fig.update_layout(
            xaxis_title=column.title(),
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    except Exception as e:
        # Return empty figure if error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_comparison_chart(data, categories, values, title="Comparison Chart"):
    """
    Create a comparison bar chart using Plotly
    
    Args:
        data (pandas.DataFrame): Data
        categories (str): Column for categories (x-axis)
        values (str): Column for values (y-axis)
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    try:
        fig = px.bar(data, x=categories, y=values, title=title)
        
        fig.update_layout(
            xaxis_title=categories.title(),
            yaxis_title=values.title(),
            showlegend=False
        )
        
        # Add value labels on bars
        fig.update_traces(texttemplate='%{y}', textposition='outside')
        
        return fig
    except Exception as e:
        # Return empty figure if error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def export_data_to_csv(data, filename=None):
    """
    Export data to CSV format
    
    Args:
        data (pandas.DataFrame): Data to export
        filename (str): Optional filename
    
    Returns:
        str: CSV string
    """
    try:
        return data.to_csv(index=False)
    except Exception as e:
        return f"Error exporting data: {str(e)}"

def validate_date_range(start_date, end_date):
    """
    Validate date range inputs
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        dict: Validation result
    """
    try:
        # Convert to datetime if strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Validation checks
        if start_date and end_date:
            if start_date > end_date:
                return {
                    'valid': False,
                    'message': 'Start date cannot be after end date'
                }
            
            if end_date > datetime.now().date():
                return {
                    'valid': False,
                    'message': 'End date cannot be in the future'
                }
            
            # Check if range is too large (more than 5 years)
            if (end_date - start_date).days > 1825:
                return {
                    'valid': False,
                    'message': 'Date range too large (maximum 5 years)'
                }
        
        return {
            'valid': True,
            'message': 'Valid date range'
        }
    
    except Exception as e:
        return {
            'valid': False,
            'message': f'Invalid date format: {str(e)}'
        }

def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, handling division by zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division fails
    
    Returns:
        float: Result of division or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def calculate_percentage_change(current, previous):
    """
    Calculate percentage change between two values
    
    Args:
        current: Current value
        previous: Previous value
    
    Returns:
        float: Percentage change
    """
    try:
        if previous == 0 or pd.isna(previous) or pd.isna(current):
            return 0.0
        return ((current - previous) / abs(previous)) * 100
    except:
        return 0.0

def get_color_palette(n_colors=10):
    """
    Get a color palette for charts
    
    Args:
        n_colors (int): Number of colors needed
    
    Returns:
        list: List of color codes
    """
    # Plotly default color sequence
    plotly_colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    
    # Extend if more colors needed
    if n_colors > len(plotly_colors):
        # Generate additional colors
        import colorsys
        additional_colors = []
        for i in range(n_colors - len(plotly_colors)):
            hue = (i * 0.618033988749895) % 1  # Golden ratio for nice distribution
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            additional_colors.append(hex_color)
        plotly_colors.extend(additional_colors)
    
    return plotly_colors[:n_colors]
