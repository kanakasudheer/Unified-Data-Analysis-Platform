import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, MetaData
import os
from datetime import datetime
import json

class DatabaseManager:
    """Database manager for storing and retrieving analysis data"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        self.engine = None
        self.metadata = MetaData()
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Connect to the PostgreSQL database"""
        try:
            if not self.database_url:
                raise ValueError("DATABASE_URL not found in environment variables")
            
            self.engine = create_engine(self.database_url)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("Database connection established successfully")
            
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            self.engine = None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.engine:
            return
        
        try:
            # Sales data table
            sales_data_table = Table(
                'sales_data', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('upload_id', String(50), nullable=False),
                Column('date', DateTime, nullable=False),
                Column('revenue', Float, nullable=False),
                Column('product', String(100)),
                Column('quantity', Integer),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            # Market data table
            market_data_table = Table(
                'market_data', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('symbol', String(10), nullable=False),
                Column('date', DateTime, nullable=False),
                Column('open_price', Float),
                Column('high_price', Float),
                Column('low_price', Float),
                Column('close_price', Float),
                Column('volume', Integer),
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            # Analysis results table
            analysis_results_table = Table(
                'analysis_results', self.metadata,
                Column('id', Integer, primary_key=True, autoincrement=True),
                Column('analysis_type', String(20), nullable=False),  # 'market' or 'sales'
                Column('analysis_id', String(50), nullable=False),
                Column('results_json', String),  # Store JSON results
                Column('created_at', DateTime, default=datetime.utcnow)
            )
            
            # Create all tables
            self.metadata.create_all(self.engine)
            print("Database tables created successfully")
            
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
    
    def save_sales_data(self, data, upload_id):
        """Save uploaded sales data to database"""
        if not self.engine:
            return False
        
        try:
            # Prepare data for insertion
            data_to_insert = data.copy()
            data_to_insert['upload_id'] = upload_id
            data_to_insert['created_at'] = datetime.utcnow()
            
            # Convert date column to datetime if it's not already
            if 'date' in data_to_insert.columns:
                data_to_insert['date'] = pd.to_datetime(data_to_insert['date'])
            
            # Insert data
            data_to_insert.to_sql('sales_data', self.engine, if_exists='append', index=False)
            print(f"Saved {len(data)} sales records to database")
            return True
            
        except Exception as e:
            print(f"Error saving sales data: {str(e)}")
            return False
    
    def save_market_data(self, data, symbol):
        """Save market data to database"""
        if not self.engine:
            return False
        
        try:
            # Prepare data for insertion
            data_to_insert = data.copy()
            data_to_insert = data_to_insert.reset_index()  # Reset index to get date as column
            
            # Rename columns to match database schema
            column_mapping = {
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            }
            
            data_to_insert = data_to_insert.rename(columns=column_mapping)
            data_to_insert['symbol'] = symbol
            data_to_insert['created_at'] = datetime.utcnow()
            
            # Select only the columns we need
            columns_to_save = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'created_at']
            data_to_insert = data_to_insert[columns_to_save]
            
            # Insert data
            data_to_insert.to_sql('market_data', self.engine, if_exists='append', index=False)
            print(f"Saved {len(data)} market records for {symbol} to database")
            return True
            
        except Exception as e:
            print(f"Error saving market data: {str(e)}")
            return False
    
    def save_analysis_results(self, analysis_type, analysis_id, results):
        """Save analysis results to database"""
        if not self.engine:
            return False
        
        try:
            # Convert results to JSON string
            results_json = json.dumps(results, default=str)
            
            # Prepare data for insertion
            data_to_insert = {
                'analysis_type': analysis_type,
                'analysis_id': analysis_id,
                'results_json': results_json,
                'created_at': datetime.utcnow()
            }
            
            # Insert data
            df = pd.DataFrame([data_to_insert])
            df.to_sql('analysis_results', self.engine, if_exists='append', index=False)
            print(f"Saved {analysis_type} analysis results to database")
            return True
            
        except Exception as e:
            print(f"Error saving analysis results: {str(e)}")
            return False
    
    def get_sales_data(self, upload_id=None, limit=1000):
        """Retrieve sales data from database"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = "SELECT * FROM sales_data"
            params = {}
            
            if upload_id:
                query += " WHERE upload_id = %(upload_id)s"
                params['upload_id'] = upload_id
            
            query += " ORDER BY date DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            data = pd.read_sql_query(query, self.engine, params=params)
            return data
            
        except Exception as e:
            print(f"Error retrieving sales data: {str(e)}")
            return pd.DataFrame()
    
    def get_market_data(self, symbol, limit=1000):
        """Retrieve market data from database"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = %(symbol)s 
                ORDER BY date DESC 
                LIMIT %(limit)s
            """
            params = {'symbol': symbol, 'limit': limit}
            
            data = pd.read_sql_query(query, self.engine, params=params)
            return data
            
        except Exception as e:
            print(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()
    
    def get_analysis_results(self, analysis_type=None, analysis_id=None):
        """Retrieve analysis results from database"""
        if not self.engine:
            return []
        
        try:
            query = "SELECT * FROM analysis_results"
            params = {}
            conditions = []
            
            if analysis_type:
                conditions.append("analysis_type = %(analysis_type)s")
                params['analysis_type'] = analysis_type
            
            if analysis_id:
                conditions.append("analysis_id = %(analysis_id)s")
                params['analysis_id'] = analysis_id
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            data = pd.read_sql_query(query, self.engine, params=params)
            
            # Parse JSON results
            results = []
            for _, row in data.iterrows():
                try:
                    parsed_results = json.loads(str(row['results_json']))
                    results.append({
                        'id': row['id'],
                        'analysis_type': row['analysis_type'],
                        'analysis_id': row['analysis_id'],
                        'results': parsed_results,
                        'created_at': row['created_at']
                    })
                except:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error retrieving analysis results: {str(e)}")
            return []
    
    def get_upload_history(self):
        """Get list of previous uploads"""
        if not self.engine:
            return []
        
        try:
            query = """
                SELECT upload_id, COUNT(*) as record_count, MIN(date) as start_date, 
                       MAX(date) as end_date, MAX(created_at) as uploaded_at
                FROM sales_data 
                GROUP BY upload_id 
                ORDER BY uploaded_at DESC
            """
            
            data = pd.read_sql_query(query, self.engine)
            return data.to_dict('records')
            
        except Exception as e:
            print(f"Error retrieving upload history: {str(e)}")
            return []
    
    def delete_sales_data(self, upload_id):
        """Delete sales data for a specific upload"""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("DELETE FROM sales_data WHERE upload_id = :upload_id"),
                    {"upload_id": upload_id}
                )
                conn.commit()
                print(f"Deleted {result.rowcount} records for upload {upload_id}")
                return True
                
        except Exception as e:
            print(f"Error deleting sales data: {str(e)}")
            return False
    
    def get_database_stats(self):
        """Get database statistics"""
        if not self.engine:
            return {}
        
        try:
            stats = {}
            
            # Sales data stats
            sales_query = "SELECT COUNT(*) as total_sales_records FROM sales_data"
            sales_result = pd.read_sql_query(sales_query, self.engine)
            stats['total_sales_records'] = sales_result['total_sales_records'].iloc[0]
            
            # Market data stats
            market_query = "SELECT COUNT(*) as total_market_records FROM market_data"
            market_result = pd.read_sql_query(market_query, self.engine)
            stats['total_market_records'] = market_result['total_market_records'].iloc[0]
            
            # Analysis results stats
            analysis_query = "SELECT COUNT(*) as total_analysis_results FROM analysis_results"
            analysis_result = pd.read_sql_query(analysis_query, self.engine)
            stats['total_analysis_results'] = analysis_result['total_analysis_results'].iloc[0]
            
            # Unique uploads
            uploads_query = "SELECT COUNT(DISTINCT upload_id) as unique_uploads FROM sales_data"
            uploads_result = pd.read_sql_query(uploads_query, self.engine)
            stats['unique_uploads'] = uploads_result['unique_uploads'].iloc[0]
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {str(e)}")
            return {}