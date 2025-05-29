import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

class GeneralDataAnalyzer:
    """General data analyzer for any CSV file with Power BI-style visualizations"""
    
    def __init__(self):
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.date_columns = []
        
    def analyze_data_structure(self, data):
        """Analyze the structure of uploaded data"""
        self.data = data
        
        # Identify column types
        self.numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(data.select_dtypes(include=['object', 'category']).columns)
        
        # Try to identify date columns
        self.date_columns = []
        for col in self.categorical_columns:
            if self._is_likely_date_column(data[col]):
                self.date_columns.append(col)
                # Remove from categorical if it's a date
                if col in self.categorical_columns:
                    self.categorical_columns.remove(col)
        
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'date_columns': self.date_columns,
            'missing_values': data.isnull().sum().to_dict()
        }
    
    def _is_likely_date_column(self, series):
        """Check if a column likely contains dates"""
        try:
            # Try to parse a sample of the data
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            parsed_count = 0
            for value in sample:
                try:
                    pd.to_datetime(str(value))
                    parsed_count += 1
                except:
                    continue
            
            # If more than 70% can be parsed as dates, consider it a date column
            return (parsed_count / len(sample)) > 0.7
        except:
            return False
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        if self.data is None:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Overview', 'Column Types', 'Missing Values', 'Data Quality'),
            specs=[[{"type": "indicator"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Data overview indicator
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=len(self.data),
                title={"text": "Total Records"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Column types pie chart
        column_types = {
            'Numeric': len(self.numeric_columns),
            'Categorical': len(self.categorical_columns),
            'Date': len(self.date_columns)
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(column_types.keys()),
                values=list(column_types.values()),
                name="Column Types"
            ),
            row=1, col=2
        )
        
        # Missing values bar chart
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].head(10)
        
        if not missing_data.empty:
            fig.add_trace(
                go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    name="Missing Values"
                ),
                row=2, col=1
            )
        
        # Data quality metrics
        quality_metrics = self._calculate_quality_metrics()
        fig.add_trace(
            go.Bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                name="Quality Score"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Data Summary Dashboard")
        return fig
    
    def _calculate_quality_metrics(self):
        """Calculate data quality metrics"""
        metrics = {}
        
        # Completeness (percentage of non-null values)
        total_cells = len(self.data) * len(self.data.columns)
        non_null_cells = total_cells - self.data.isnull().sum().sum()
        metrics['Completeness'] = (non_null_cells / total_cells) * 100
        
        # Uniqueness for each column (average)
        uniqueness_scores = []
        for col in self.data.columns:
            unique_ratio = len(self.data[col].unique()) / len(self.data)
            uniqueness_scores.append(unique_ratio * 100)
        metrics['Uniqueness'] = np.mean(uniqueness_scores)
        
        # Validity (percentage of valid data types)
        metrics['Validity'] = 85  # Simplified for now
        
        return metrics
    
    def create_numeric_analysis(self):
        """Create analysis for numeric columns"""
        if not self.numeric_columns:
            return None
        
        charts = []
        
        # Distribution plots for each numeric column
        for col in self.numeric_columns[:6]:  # Limit to first 6 columns
            fig = px.histogram(
                self.data, 
                x=col, 
                title=f'Distribution of {col}',
                marginal="box"
            )
            charts.append(fig)
        
        # Correlation heatmap if multiple numeric columns
        if len(self.numeric_columns) > 1:
            correlation_matrix = self.data[self.numeric_columns].corr()
            fig = px.imshow(
                correlation_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            charts.append(fig)
        
        return charts
    
    def create_categorical_analysis(self):
        """Create analysis for categorical columns"""
        if not self.categorical_columns:
            return None
        
        charts = []
        
        # Top values for each categorical column
        for col in self.categorical_columns[:6]:  # Limit to first 6 columns
            value_counts = self.data[col].value_counts().head(10)
            
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f'Top 10 Values in {col}'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            charts.append(fig)
        
        return charts
    
    def create_time_series_analysis(self):
        """Create time series analysis if date columns exist"""
        if not self.date_columns or not self.numeric_columns:
            return None
        
        charts = []
        
        for date_col in self.date_columns[:2]:  # Limit to first 2 date columns
            # Convert to datetime
            try:
                self.data[f'{date_col}_parsed'] = pd.to_datetime(self.data[date_col])
                
                for numeric_col in self.numeric_columns[:3]:  # Limit to first 3 numeric columns
                    # Create time series plot
                    fig = px.line(
                        self.data.sort_values(f'{date_col}_parsed'),
                        x=f'{date_col}_parsed',
                        y=numeric_col,
                        title=f'{numeric_col} over {date_col}'
                    )
                    charts.append(fig)
                    
            except Exception as e:
                st.warning(f"Could not parse date column {date_col}: {str(e)}")
                continue
        
        return charts
    
    def create_cross_analysis(self):
        """Create cross-analysis between different column types"""
        charts = []
        
        # Numeric vs Categorical
        if self.numeric_columns and self.categorical_columns:
            for num_col in self.numeric_columns[:2]:
                for cat_col in self.categorical_columns[:2]:
                    # Box plot
                    fig = px.box(
                        self.data,
                        x=cat_col,
                        y=num_col,
                        title=f'{num_col} by {cat_col}'
                    )
                    charts.append(fig)
        
        # Scatter plots for numeric columns
        if len(self.numeric_columns) >= 2:
            for i in range(min(3, len(self.numeric_columns)-1)):
                fig = px.scatter(
                    self.data,
                    x=self.numeric_columns[i],
                    y=self.numeric_columns[i+1],
                    title=f'{self.numeric_columns[i]} vs {self.numeric_columns[i+1]}'
                )
                charts.append(fig)
        
        return charts
    
    def get_data_insights(self):
        """Generate text insights about the data"""
        insights = []
        
        if self.data is None:
            return insights
        
        # Basic insights
        insights.append(f"Dataset contains {len(self.data):,} rows and {len(self.data.columns)} columns")
        insights.append(f"Found {len(self.numeric_columns)} numeric columns, {len(self.categorical_columns)} categorical columns, and {len(self.date_columns)} date columns")
        
        # Missing data insights
        missing_percentage = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        if missing_percentage > 10:
            insights.append(f"⚠️ {missing_percentage:.1f}% of data is missing - consider data cleaning")
        elif missing_percentage > 0:
            insights.append(f"✓ Low missing data: {missing_percentage:.1f}%")
        else:
            insights.append("✓ No missing data detected")
        
        # Numeric insights
        if self.numeric_columns:
            for col in self.numeric_columns[:3]:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    insights.append(f"{col}: Range {col_data.min():.2f} to {col_data.max():.2f}, Average {col_data.mean():.2f}")
        
        # Categorical insights
        if self.categorical_columns:
            for col in self.categorical_columns[:3]:
                unique_count = self.data[col].nunique()
                total_count = len(self.data[col].dropna())
                if unique_count > 0:
                    insights.append(f"{col}: {unique_count} unique values out of {total_count} records")
        
        return insights
    
    def export_summary_report(self):
        """Export a summary report of the analysis"""
        if self.data is None:
            return None
        
        summary = {
            'dataset_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'size_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'column_analysis': {
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'date_columns': self.date_columns
            },
            'data_quality': self._calculate_quality_metrics(),
            'statistical_summary': self.data.describe().to_dict() if self.numeric_columns else {},
            'insights': self.get_data_insights()
        }
        
        return summary