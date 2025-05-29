import streamlit as st
import time
import json

class TooltipManager:
    """Manages animated tooltips and user guidance for data transformation steps"""
    
    def __init__(self):
        self.tooltip_steps = {
            'upload': [
                "📁 Click 'Browse files' to select your CSV file",
                "🔍 The system will automatically detect your data structure",
                "✅ Look for the success message confirming your upload"
            ],
            'structure_analysis': [
                "🧮 Numeric columns contain numbers for calculations and charts",
                "📝 Categorical columns contain text or categories for grouping",
                "📅 Date columns enable time-based analysis and trends"
            ],
            'visualization': [
                "📊 Summary dashboard shows overall data quality and structure",
                "📈 Distribution plots reveal patterns in your numeric data",
                "🔗 Correlation matrix shows relationships between variables",
                "📅 Time series charts track changes over time",
                "🎯 Cross-analysis compares different data dimensions"
            ],
            'insights': [
                "💡 Key insights highlight important patterns in your data",
                "📋 Statistical summary provides detailed numeric analysis",
                "⚠️ Quality metrics help identify data issues",
                "📤 Export options save your analysis and processed data"
            ]
        }
        
        self.transformation_steps = {
            'data_loading': "Loading and parsing your CSV file...",
            'column_detection': "Detecting column types and data structure...",
            'quality_check': "Analyzing data quality and completeness...",
            'chart_generation': "Generating interactive visualizations...",
            'insight_analysis': "Extracting key insights from your data...",
            'report_preparation': "Preparing comprehensive analysis report..."
        }
    
    def show_animated_tooltip(self, step_key, delay=2):
        """Display animated tooltips for a specific step"""
        if step_key not in self.tooltip_steps:
            return
        
        tooltip_container = st.container()
        
        with tooltip_container:
            for i, tip in enumerate(self.tooltip_steps[step_key]):
                with st.expander(f"💡 Tip {i+1}", expanded=True):
                    st.info(tip)
                    if i < len(self.tooltip_steps[step_key]) - 1:
                        time.sleep(delay)
                        st.rerun()
    
    def show_progress_tooltip(self, step_key, progress_value=0):
        """Show animated progress tooltip with steps"""
        if step_key not in self.transformation_steps:
            return None, None
        
        progress_bar = st.progress(progress_value)
        status_text = st.empty()
        
        status_text.text(self.transformation_steps[step_key])
        
        return progress_bar, status_text
    
    def show_guided_tour(self):
        """Show a guided tour of the data analysis features"""
        st.markdown("### 🎯 Quick Start Guide")
        
        with st.expander("📚 How to Use General Data Analysis", expanded=False):
            st.markdown("""
            **Step 1: Upload Your Data**
            - Click 'Browse files' and select any CSV file
            - Supported formats: Sales data, financial records, survey responses, etc.
            - File size limit: Up to 200MB
            
            **Step 2: Review Data Structure**
            - System automatically detects column types
            - Numeric: Numbers for calculations and metrics
            - Categorical: Text/categories for grouping
            - Date: Time-based data for trends
            
            **Step 3: Generate Analysis**
            - Click 'Generate Comprehensive Analysis'
            - View multiple chart types and insights
            - Export reports and processed data
            
            **Step 4: Interpret Results**
            - Review data quality metrics
            - Analyze patterns and correlations
            - Download reports for sharing
            """)
    
    def show_interactive_help(self, context):
        """Show context-sensitive help based on current user action"""
        help_messages = {
            'file_upload': {
                'title': "📁 File Upload Help",
                'content': """
                **Supported File Types:** CSV files only
                **Data Requirements:** 
                - First row should contain column headers
                - Data should be in tabular format
                - Text encoding: UTF-8 recommended
                
                **Common Issues:**
                - File too large? Try reducing rows or splitting data
                - Special characters? Ensure UTF-8 encoding
                - Date format problems? Use YYYY-MM-DD format
                """
            },
            'column_mapping': {
                'title': "🎯 Column Detection Help", 
                'content': """
                **Automatic Detection:**
                - Numeric: Automatically detected for calculations
                - Categorical: Text data for grouping and filtering
                - Date: Timestamps for time-series analysis
                
                **Manual Override:**
                - If detection is wrong, data will still process
                - Charts adapt to available column types
                - You can clean data before re-uploading
                """
            },
            'visualization': {
                'title': "📊 Visualization Guide",
                'content': """
                **Chart Types Generated:**
                - Distribution plots: Show data spread and patterns
                - Correlation matrix: Relationships between variables
                - Time series: Trends over time (if dates present)
                - Cross-analysis: Compare different dimensions
                
                **Interactive Features:**
                - Hover for detailed values
                - Zoom and pan on charts
                - Download charts as images
                """
            }
        }
        
        if context in help_messages:
            help_info = help_messages[context]
            with st.expander(help_info['title'], expanded=False):
                st.markdown(help_info['content'])
    
    def show_data_transformation_steps(self, data_info):
        """Show animated steps of data transformation process"""
        steps_container = st.container()
        
        with steps_container:
            st.markdown("### 🔄 Data Transformation Process")
            
            # Step 1: Data Loading
            with st.expander("Step 1: Data Loading ✅", expanded=False):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Rows Loaded", f"{data_info.get('rows', 0):,}")
                with col2:
                    st.success("✅ Data successfully loaded and parsed")
            
            # Step 2: Structure Analysis
            with st.expander("Step 2: Structure Analysis ✅", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Numeric Columns", len(data_info.get('numeric_columns', [])))
                with col2:
                    st.metric("Categorical Columns", len(data_info.get('categorical_columns', [])))
                with col3:
                    st.metric("Date Columns", len(data_info.get('date_columns', [])))
            
            # Step 3: Quality Assessment
            with st.expander("Step 3: Quality Assessment", expanded=False):
                quality_score = self._calculate_quality_score(data_info)
                st.progress(quality_score / 100)
                st.write(f"Overall Data Quality: {quality_score:.1f}%")
                
                if quality_score > 80:
                    st.success("🌟 Excellent data quality detected!")
                elif quality_score > 60:
                    st.warning("⚠️ Good data quality with minor issues")
                else:
                    st.error("🔧 Data quality needs improvement")
    
    def _calculate_quality_score(self, data_info):
        """Calculate a simple data quality score"""
        score = 0
        
        # Points for having different column types
        if data_info.get('numeric_columns'):
            score += 30
        if data_info.get('categorical_columns'):
            score += 20
        if data_info.get('date_columns'):
            score += 20
        
        # Points for data completeness (simplified)
        missing_ratio = data_info.get('missing_percentage', 0)
        completeness_score = max(0, 30 - missing_ratio)
        score += completeness_score
        
        return min(100, score)
    
    def show_chart_explanation(self, chart_type):
        """Show explanation for specific chart types"""
        explanations = {
            'histogram': {
                'icon': '📊',
                'title': 'Distribution Plot (Histogram)',
                'explanation': 'Shows how your data values are distributed. Peaks indicate common values, while gaps show rare ranges.'
            },
            'correlation': {
                'icon': '🔗',
                'title': 'Correlation Matrix',
                'explanation': 'Displays relationships between numeric variables. Values close to 1 or -1 indicate strong relationships.'
            },
            'time_series': {
                'icon': '📈',
                'title': 'Time Series Chart',
                'explanation': 'Shows how values change over time. Look for trends, seasonality, and sudden changes.'
            },
            'box_plot': {
                'icon': '📦',
                'title': 'Box Plot',
                'explanation': 'Compares distributions across categories. Shows median, quartiles, and outliers for each group.'
            },
            'scatter': {
                'icon': '⭐',
                'title': 'Scatter Plot',
                'explanation': 'Reveals relationships between two numeric variables. Patterns indicate correlations.'
            }
        }
        
        if chart_type in explanations:
            exp = explanations[chart_type]
            st.info(f"{exp['icon']} **{exp['title']}**: {exp['explanation']}")
    
    def show_insight_tooltip(self, insight_type, value):
        """Show contextual tooltips for insights"""
        tooltip_messages = {
            'high_correlation': f"🔗 Strong correlation detected ({value:.2f}). These variables move together.",
            'outliers_detected': f"⚠️ {value} outliers found. These unusual values may need investigation.",
            'missing_data': f"📝 {value:.1f}% missing data. Consider data cleaning strategies.",
            'high_variability': f"📈 High variability detected. Data shows significant spread.",
            'trend_detected': f"📊 Clear trend identified. Data shows consistent direction over time."
        }
        
        if insight_type in tooltip_messages:
            st.info(tooltip_messages[insight_type])
    
    def create_step_by_step_guide(self):
        """Create an interactive step-by-step guide"""
        st.markdown("### 📋 Interactive Guide")
        
        if 'guide_step' not in st.session_state:
            st.session_state.guide_step = 0
        
        steps = [
            {
                'title': '📁 Step 1: Upload Your Data',
                'content': 'Start by uploading a CSV file using the file uploader above.',
                'action': 'Upload a CSV file to continue'
            },
            {
                'title': '🔍 Step 2: Review Data Structure', 
                'content': 'Check how the system categorized your columns (numeric, categorical, date).',
                'action': 'Verify the column detection is correct'
            },
            {
                'title': '🚀 Step 3: Generate Analysis',
                'content': 'Click the "Generate Comprehensive Analysis" button to create visualizations.',
                'action': 'Generate your analysis to see charts and insights'
            },
            {
                'title': '📊 Step 4: Explore Results',
                'content': 'Review the charts, insights, and statistical summaries generated.',
                'action': 'Scroll through all the analysis sections'
            },
            {
                'title': '📤 Step 5: Export Results',
                'content': 'Download your analysis report or processed data for future use.',
                'action': 'Use the export options to save your work'
            }
        ]
        
        current_step = st.session_state.guide_step
        
        # Progress indicator
        progress = (current_step + 1) / len(steps)
        st.progress(progress)
        st.write(f"Step {current_step + 1} of {len(steps)}")
        
        # Current step display
        step = steps[current_step]
        st.markdown(f"#### {step['title']}")
        st.write(step['content'])
        st.info(f"**Action needed:** {step['action']}")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_step > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.guide_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset Guide"):
                st.session_state.guide_step = 0
                st.rerun()
        
        with col3:
            if current_step < len(steps) - 1:
                if st.button("Next ➡️"):
                    st.session_state.guide_step += 1
                    st.rerun()
            else:
                st.success("🎉 Guide Complete! You're ready to analyze data.")