import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from datetime import datetime, timedelta
import json

class ReportGenerator:
    """Generate comprehensive reports combining market and sales analysis"""
    
    def __init__(self):
        self.report_templates = {
            'executive': self._executive_template,
            'detailed': self._detailed_template,
            'custom': self._custom_template
        }
    
    def generate_combined_report(self, report_data):
        """
        Generate a combined report with market and sales analysis
        
        Args:
            report_data (dict): Contains market_results, sales_results, and config
        
        Returns:
            dict: Generated report content in various formats
        """
        try:
            config = report_data.get('config', {})
            report_type = config.get('report_type', 'Executive Summary').lower().replace(' ', '_')
            
            # Generate report based on type
            if report_type in self.report_templates:
                report_content = self.report_templates[report_type](report_data)
            else:
                report_content = self._executive_template(report_data)
            
            # Generate different formats
            result = {
                'summary': report_content['summary'],
                'html': self._generate_html_report(report_content),
                'csv_data': self._generate_csv_data(report_data),
                'charts': report_content.get('charts', [])
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating combined report: {str(e)}")
            return {
                'summary': 'Error generating report',
                'html': '<p>Error generating report</p>',
                'csv_data': '',
                'charts': []
            }
    
    def _executive_template(self, report_data):
        """Generate executive summary template"""
        content = {
            'title': 'Executive Summary Report',
            'sections': [],
            'charts': []
        }
        
        # Market analysis section
        market_results = report_data.get('market_results')
        if market_results:
            market_section = self._generate_market_summary(market_results)
            content['sections'].append(market_section)
        
        # Sales analysis section
        sales_results = report_data.get('sales_results')
        if sales_results:
            sales_section = self._generate_sales_summary(sales_results)
            content['sections'].append(sales_section)
        
        # Combined insights
        if market_results and sales_results:
            combined_section = self._generate_combined_insights(market_results, sales_results)
            content['sections'].append(combined_section)
        
        # Generate summary
        content['summary'] = self._create_executive_summary(content['sections'])
        
        return content
    
    def _detailed_template(self, report_data):
        """Generate detailed analysis template"""
        content = {
            'title': 'Detailed Analysis Report',
            'sections': [],
            'charts': []
        }
        
        # Detailed market analysis
        market_results = report_data.get('market_results')
        if market_results:
            market_section = self._generate_detailed_market_analysis(market_results)
            content['sections'].append(market_section)
        
        # Detailed sales analysis
        sales_results = report_data.get('sales_results')
        if sales_results:
            sales_section = self._generate_detailed_sales_analysis(sales_results)
            content['sections'].append(sales_section)
        
        # Technical details and methodology
        methodology_section = self._generate_methodology_section(report_data)
        content['sections'].append(methodology_section)
        
        content['summary'] = self._create_detailed_summary(content['sections'])
        
        return content
    
    def _custom_template(self, report_data):
        """Generate custom report template"""
        # For now, use executive template as base
        return self._executive_template(report_data)
    
    def _generate_market_summary(self, market_results):
        """Generate market analysis summary section"""
        section = {
            'title': '📈 Market Analysis Summary',
            'content': [],
            'metrics': {}
        }
        
        try:
            stock_data = market_results.get('stock_data')
            technical_analysis = market_results.get('technical_analysis', {})
            news_sentiment = market_results.get('news_sentiment', [])
            
            if stock_data is not None and not stock_data.empty:
                # Key metrics
                current_price = stock_data['Close'].iloc[-1]
                first_price = stock_data['Close'].iloc[0]
                price_change = ((current_price - first_price) / first_price) * 100
                
                section['metrics']['current_price'] = current_price
                section['metrics']['price_change'] = price_change
                section['metrics']['volatility'] = technical_analysis.get('volatility', 0)
                section['metrics']['rsi'] = technical_analysis.get('rsi', 0)
                
                # Content
                section['content'].append(f"**Current Stock Price:** ${current_price:.2f}")
                section['content'].append(f"**Period Change:** {price_change:.2f}%")
                section['content'].append(f"**Volatility:** {technical_analysis.get('volatility', 0):.2f}%")
                section['content'].append(f"**RSI:** {technical_analysis.get('rsi', 0):.2f}")
                
                # Trend analysis
                trend_direction = technical_analysis.get('trend_direction', 'Unknown')
                section['content'].append(f"**Trend Direction:** {trend_direction}")
                
                # Trading signals
                signals = technical_analysis.get('signals', [])
                if signals:
                    section['content'].append(f"**Key Signals:** {', '.join(signals[:3])}")
                
                # Sentiment analysis
                if news_sentiment:
                    avg_sentiment = sum([item.get('score', 0) for item in news_sentiment]) / len(news_sentiment)
                    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                    section['content'].append(f"**News Sentiment:** {sentiment_label} ({avg_sentiment:.2f})")
                    section['metrics']['sentiment_score'] = avg_sentiment
        
        except Exception as e:
            section['content'].append(f"Error generating market summary: {str(e)}")
        
        return section
    
    def _generate_sales_summary(self, sales_results):
        """Generate sales analysis summary section"""
        section = {
            'title': '🛍️ Sales Analysis Summary',
            'content': [],
            'metrics': {}
        }
        
        try:
            processed_data = sales_results.get('processed_data')
            sales_analysis = sales_results.get('sales_analysis', {})
            anomalies = sales_results.get('anomalies', pd.DataFrame())
            
            if processed_data is not None and not processed_data.empty:
                # Key metrics
                total_revenue = processed_data['revenue'].sum()
                avg_revenue = processed_data['revenue'].mean()
                data_points = len(processed_data)
                anomaly_count = len(anomalies) if not anomalies.empty else 0
                
                section['metrics']['total_revenue'] = total_revenue
                section['metrics']['avg_revenue'] = avg_revenue
                section['metrics']['data_points'] = data_points
                section['metrics']['anomaly_count'] = anomaly_count
                
                # Content
                section['content'].append(f"**Total Revenue:** ${total_revenue:,.2f}")
                section['content'].append(f"**Average Daily Revenue:** ${avg_revenue:,.2f}")
                section['content'].append(f"**Data Points:** {data_points:,}")
                section['content'].append(f"**Anomalies Detected:** {anomaly_count}")
                
                # Growth metrics
                growth_rate = sales_analysis.get('total_growth_rate', 0)
                section['content'].append(f"**Growth Rate:** {growth_rate:.2f}%")
                section['metrics']['growth_rate'] = growth_rate
                
                # Performance insights
                cv = sales_analysis.get('coefficient_of_variation', 0)
                consistency = "High" if cv < 20 else "Medium" if cv < 50 else "Low"
                section['content'].append(f"**Revenue Consistency:** {consistency} (CV: {cv:.1f}%)")
                
                # Trend direction
                trend_direction = sales_analysis.get('trend_direction', 'Unknown')
                section['content'].append(f"**Trend Direction:** {trend_direction}")
        
        except Exception as e:
            section['content'].append(f"Error generating sales summary: {str(e)}")
        
        return section
    
    def _generate_combined_insights(self, market_results, sales_results):
        """Generate combined insights section"""
        section = {
            'title': '🔗 Combined Business Insights',
            'content': [],
            'recommendations': []
        }
        
        try:
            # Extract key metrics
            market_metrics = {}
            sales_metrics = {}
            
            # Market metrics
            if market_results.get('stock_data') is not None:
                stock_data = market_results['stock_data']
                current_price = stock_data['Close'].iloc[-1]
                first_price = stock_data['Close'].iloc[0]
                market_change = ((current_price - first_price) / first_price) * 100
                market_metrics['price_change'] = market_change
                
                sentiment = market_results.get('news_sentiment', [])
                if sentiment:
                    avg_sentiment = sum([item.get('score', 0) for item in sentiment]) / len(sentiment)
                    market_metrics['sentiment'] = avg_sentiment
            
            # Sales metrics
            if sales_results.get('processed_data') is not None:
                sales_analysis = sales_results.get('sales_analysis', {})
                sales_metrics['growth_rate'] = sales_analysis.get('total_growth_rate', 0)
                sales_metrics['total_revenue'] = sales_results['processed_data']['revenue'].sum()
            
            # Generate insights
            insights = []
            
            # Correlation insights
            if 'price_change' in market_metrics and 'growth_rate' in sales_metrics:
                market_change = market_metrics['price_change']
                sales_growth = sales_metrics['growth_rate']
                
                if market_change > 10 and sales_growth > 10:
                    insights.append("Strong correlation: Both market performance and sales are showing positive growth")
                elif market_change < -10 and sales_growth < -10:
                    insights.append("Concerning correlation: Both market and sales performance are declining")
                elif market_change > 10 and sales_growth < -10:
                    insights.append("Market confidence high but sales declining - investigate operational issues")
                elif market_change < -10 and sales_growth > 10:
                    insights.append("Sales growth despite market concerns - strong business fundamentals")
            
            # Sentiment vs sales correlation
            if 'sentiment' in market_metrics and 'growth_rate' in sales_metrics:
                sentiment = market_metrics['sentiment']
                sales_growth = sales_metrics['growth_rate']
                
                if sentiment > 0.1 and sales_growth > 5:
                    insights.append("Positive market sentiment aligns with strong sales performance")
                elif sentiment < -0.1 and sales_growth > 5:
                    insights.append("Sales growth despite negative sentiment - resilient business model")
            
            # Generate recommendations
            recommendations = []
            
            if market_metrics.get('price_change', 0) > 15:
                recommendations.append("Consider capitalizing on strong market performance for expansion")
            
            if sales_metrics.get('growth_rate', 0) > 20:
                recommendations.append("Strong sales growth - ensure operational capacity can handle continued expansion")
            
            if market_metrics.get('sentiment', 0) < -0.2:
                recommendations.append("Negative market sentiment - focus on defensive strategies and cash flow management")
            
            # Add to section
            section['content'] = insights if insights else ["No significant correlations detected"]
            section['recommendations'] = recommendations if recommendations else ["Continue monitoring both market and sales trends"]
        
        except Exception as e:
            section['content'].append(f"Error generating combined insights: {str(e)}")
        
        return section
    
    def _generate_detailed_market_analysis(self, market_results):
        """Generate detailed market analysis section"""
        section = {
            'title': '📊 Detailed Market Analysis',
            'content': [],
            'technical_details': {}
        }
        
        try:
            stock_data = market_results.get('stock_data')
            technical_analysis = market_results.get('technical_analysis', {})
            
            if stock_data is not None:
                # Technical indicators
                section['content'].append("### Technical Indicators")
                section['content'].append(f"- RSI: {technical_analysis.get('rsi', 0):.2f}")
                section['content'].append(f"- Volatility: {technical_analysis.get('volatility', 0):.2f}%")
                section['content'].append(f"- Trend: {technical_analysis.get('trend_direction', 'Unknown')}")
                
                # Support and resistance
                support_resistance = technical_analysis.get('support_resistance', {})
                if support_resistance:
                    section['content'].append("### Support and Resistance Levels")
                    resistance = support_resistance.get('resistance', [])
                    support = support_resistance.get('support', [])
                    
                    if resistance:
                        section['content'].append(f"- Resistance levels: {', '.join([f'${r:.2f}' for r in resistance])}")
                    if support:
                        section['content'].append(f"- Support levels: {', '.join([f'${s:.2f}' for s in support])}")
                
                # Volume analysis
                volume_analysis = technical_analysis.get('volume_analysis', 'Normal Volume')
                section['content'].append(f"### Volume Analysis: {volume_analysis}")
                
                # Trading signals
                signals = technical_analysis.get('signals', [])
                if signals:
                    section['content'].append("### Trading Signals")
                    for signal in signals:
                        section['content'].append(f"- {signal}")
        
        except Exception as e:
            section['content'].append(f"Error in detailed market analysis: {str(e)}")
        
        return section
    
    def _generate_detailed_sales_analysis(self, sales_results):
        """Generate detailed sales analysis section"""
        section = {
            'title': '📈 Detailed Sales Analysis',
            'content': [],
            'analysis_details': {}
        }
        
        try:
            sales_analysis = sales_results.get('sales_analysis', {})
            anomalies = sales_results.get('anomalies', pd.DataFrame())
            
            # Time patterns
            if 'seasonal' in sales_analysis:
                seasonal = sales_analysis['seasonal']
                section['content'].append("### Seasonal Patterns")
                
                if 'monthly_patterns' in seasonal:
                    monthly = seasonal['monthly_patterns']
                    section['content'].append(f"- Peak month: {monthly.get('peak_month', 'Unknown')}")
                    section['content'].append(f"- Lowest month: {monthly.get('lowest_month', 'Unknown')}")
                
                if 'day_of_week_patterns' in seasonal:
                    dow = seasonal['day_of_week_patterns']
                    section['content'].append(f"- Best day: {dow.get('peak_day', 'Unknown')}")
                    section['content'].append(f"- Slowest day: {dow.get('lowest_day', 'Unknown')}")
            
            # Performance metrics
            section['content'].append("### Performance Metrics")
            cv = sales_analysis.get('coefficient_of_variation', 0)
            section['content'].append(f"- Coefficient of Variation: {cv:.2f}%")
            
            percentiles = sales_analysis.get('percentiles', {})
            if percentiles:
                section['content'].append("- Revenue Percentiles:")
                for p, value in percentiles.items():
                    section['content'].append(f"  - {p}: ${value:,.2f}")
            
            # Anomaly details
            if not anomalies.empty:
                section['content'].append("### Anomaly Analysis")
                section['content'].append(f"- Total anomalies detected: {len(anomalies)}")
                
                severity_counts = anomalies['severity'].value_counts()
                for severity, count in severity_counts.items():
                    section['content'].append(f"- {severity} severity: {count}")
        
        except Exception as e:
            section['content'].append(f"Error in detailed sales analysis: {str(e)}")
        
        return section
    
    def _generate_methodology_section(self, report_data):
        """Generate methodology and technical details section"""
        section = {
            'title': '🔬 Methodology & Technical Details',
            'content': []
        }
        
        # Market analysis methodology
        if report_data.get('market_results'):
            section['content'].append("### Market Analysis Methodology")
            section['content'].append("- Data source: Yahoo Finance API")
            section['content'].append("- Technical indicators: RSI, Moving Averages, Bollinger Bands")
            section['content'].append("- Sentiment analysis: TextBlob natural language processing")
            section['content'].append("- Volatility calculation: 20-day rolling standard deviation")
        
        # Sales analysis methodology
        if report_data.get('sales_results'):
            section['content'].append("### Sales Analysis Methodology")
            section['content'].append("- Anomaly detection: Isolation Forest algorithm")
            section['content'].append("- Time series analysis: Moving averages and trend decomposition")
            section['content'].append("- Seasonality detection: Monthly and weekly pattern analysis")
            section['content'].append("- Growth calculation: Period-over-period percentage change")
        
        # Report generation details
        section['content'].append("### Report Generation")
        generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        section['content'].append(f"- Generated on: {generation_time}")
        section['content'].append("- Analysis framework: Python with Pandas, Scikit-learn, and Plotly")
        
        return section
    
    def _create_executive_summary(self, sections):
        """Create executive summary from sections"""
        summary_parts = []
        
        for section in sections:
            if section['title'].startswith('📈 Market'):
                if section['content']:
                    summary_parts.append("**Market Overview:** " + "; ".join(section['content'][:3]))
            elif section['title'].startswith('🛍️ Sales'):
                if section['content']:
                    summary_parts.append("**Sales Overview:** " + "; ".join(section['content'][:3]))
            elif section['title'].startswith('🔗 Combined'):
                if section['content']:
                    summary_parts.append("**Key Insights:** " + "; ".join(section['content']))
        
        return "\n\n".join(summary_parts) if summary_parts else "No analysis data available"
    
    def _create_detailed_summary(self, sections):
        """Create detailed summary from sections"""
        summary_parts = []
        
        for section in sections:
            if section['content']:
                summary_parts.append(f"**{section['title']}**")
                summary_parts.append("\n".join(section['content'][:5]))  # First 5 points
                summary_parts.append("")  # Empty line
        
        return "\n".join(summary_parts) if summary_parts else "No detailed analysis available"
    
    def _generate_html_report(self, content):
        """Generate HTML version of the report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; }}
                h2 {{ color: #007bff; margin-top: 30px; }}
                h3 {{ color: #555; }}
                .metric {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
                .section {{ margin: 20px 0; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
            
            {sections_html}
            
            <hr>
            <p class="timestamp">Report generated by Unified Data Analysis Platform</p>
        </body>
        </html>
        """
        
        # Generate sections HTML
        sections_html = ""
        for section in content['sections']:
            sections_html += f"<div class='section'><h2>{section['title']}</h2>"
            
            for item in section['content']:
                if item.startswith('**') and item.endswith('**'):
                    # Bold items
                    sections_html += f"<div class='metric'>{item[2:-2]}</div>"
                elif item.startswith('###'):
                    # Subheadings
                    sections_html += f"<h3>{item[4:]}</h3>"
                elif item.startswith('- '):
                    # List items
                    if not sections_html.endswith('<ul>'):
                        sections_html += "<ul>"
                    sections_html += f"<li>{item[2:]}</li>"
                else:
                    # Close any open lists
                    if sections_html.endswith('</li>'):
                        sections_html += "</ul>"
                    sections_html += f"<p>{item}</p>"
            
            # Close any open lists
            if sections_html.endswith('</li>'):
                sections_html += "</ul>"
            
            sections_html += "</div>"
        
        return html_template.format(
            title=content['title'],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sections_html=sections_html
        )
    
    def _generate_csv_data(self, report_data):
        """Generate CSV data for download"""
        csv_parts = []
        
        # Market data
        market_results = report_data.get('market_results')
        if market_results and market_results.get('stock_data') is not None:
            stock_data = market_results['stock_data'].reset_index()
            csv_parts.append("Market Data")
            csv_parts.append(stock_data.to_csv(index=False))
            csv_parts.append("")
        
        # Sales data
        sales_results = report_data.get('sales_results')
        if sales_results and sales_results.get('processed_data') is not None:
            sales_data = sales_results['processed_data']
            csv_parts.append("Sales Data")
            csv_parts.append(sales_data.to_csv(index=False))
            csv_parts.append("")
        
        # Anomalies
        if sales_results and not sales_results.get('anomalies', pd.DataFrame()).empty:
            anomalies = sales_results['anomalies']
            csv_parts.append("Detected Anomalies")
            csv_parts.append(anomalies.to_csv(index=False))
        
        return "\n".join(csv_parts) if csv_parts else "No data available for export"
    
    def create_dashboard_charts(self, report_data):
        """Create charts for dashboard display"""
        charts = []
        
        try:
            # Market charts
            market_results = report_data.get('market_results')
            if market_results and market_results.get('stock_data') is not None:
                stock_data = market_results['stock_data'].reset_index()
                
                # Stock price chart
                fig_stock = px.line(stock_data, x='Date', y='Close', 
                                   title='Stock Price Trend')
                charts.append({
                    'type': 'market_price',
                    'title': 'Stock Price Trend',
                    'chart': fig_stock
                })
            
            # Sales charts
            sales_results = report_data.get('sales_results')
            if sales_results and sales_results.get('processed_data') is not None:
                sales_data = sales_results['processed_data']
                
                # Revenue trend chart
                fig_revenue = px.line(sales_data, x='date', y='revenue',
                                     title='Revenue Trend')
                charts.append({
                    'type': 'sales_revenue',
                    'title': 'Revenue Trend',
                    'chart': fig_revenue
                })
                
                # Anomalies overlay
                anomalies = sales_results.get('anomalies', pd.DataFrame())
                if not anomalies.empty:
                    fig_revenue.add_scatter(x=anomalies['date'], y=anomalies['revenue'],
                                          mode='markers', name='Anomalies',
                                          marker=dict(color='red', size=8))
        
        except Exception as e:
            print(f"Error creating dashboard charts: {str(e)}")
        
        return charts
