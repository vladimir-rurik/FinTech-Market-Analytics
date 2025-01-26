"""
Data preprocessing and feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from sklearn.preprocessing import RobustScaler
import sqlite3
import os
from datetime import datetime

from .feature_engineering import FeatureEngineer

class DataPreprocessor:
    """Data preprocessing and feature engineering pipeline."""
    
    def __init__(self, db_path: str = 'data/market_data.db'):
        """Initialize preprocessor."""
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Create raw data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS raw_data (
                    date TEXT,
                    symbol TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (date, symbol)
                )
            ''')
            
            # Create processed data table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_data (
                    date TEXT,
                    symbol TEXT,
                    feature_name TEXT,
                    value REAL,
                    PRIMARY KEY (date, symbol, feature_name)
                )
            ''')
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling outliers and missing values.
        
        Args:
            data: Raw market data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        # Step 1: Handle missing values
        data = self._handle_missing_values(data)
        
        # Step 2: Remove outliers using RobustScaler
        data = self._remove_outliers(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        # Forward fill for missing values
        data = data.fillna(method='ffill')
        # Backward fill for any remaining missing values
        data = data.fillna(method='bfill')
        return data
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using RobustScaler."""
        scaler = RobustScaler()
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in cols_to_scale:
            if col in data.columns:
                scaled_data = scaler.fit_transform(data[col].values.reshape(-1, 1))
                # Remove data points that are more than 3 scaled units away from median
                mask = np.abs(scaled_data) <= 3.0
                data.loc[~mask.ravel(), col] = np.nan
                
        # Fill removed outliers
        data = self._handle_missing_values(data)
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create new features from market data."""
        return self.feature_engineer.process_features(data)
    
    def process_new_data(self, symbol: str, new_data: pd.DataFrame):
        """
        Process new data and store in database.
        
        Args:
            symbol: Asset symbol
            new_data: New market data
        """
        # Step 1: Clean the data
        cleaned_data = self.clean_data(new_data)
        
        # Step 2: Engineer features
        features = self.engineer_features(cleaned_data)
        
        # Step 3: Store data in database
        with sqlite3.connect(self.db_path) as conn:
            # Store raw data
            for idx, row in cleaned_data.iterrows():
                conn.execute('''
                    INSERT OR IGNORE INTO raw_data
                    (date, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (idx.strftime('%Y-%m-%d %H:%M:%S'), symbol,
                      row['Open'], row['High'], row['Low'],
                      row['Close'], row['Volume']))
            
            # Store processed features
            for col in features.columns:
                for idx, value in features[col].items():
                    if pd.notna(value):
                        conn.execute('''
                            INSERT OR IGNORE INTO processed_data
                            (date, symbol, feature_name, value)
                            VALUES (?, ?, ?, ?)
                        ''', (idx.strftime('%Y-%m-%d %H:%M:%S'),
                              symbol, col, value))
    
    def get_latest_data(self, symbol: str, 
                       start_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get latest data from database.
        
        Args:
            symbol: Asset symbol
            start_date: Optional start date for filtering
            
        Returns:
            Tuple of (raw_data, processed_features)
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get raw data
            query = '''
                SELECT date, open, high, low, close, volume
                FROM raw_data
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
                
            raw_df = pd.read_sql_query(
                query + ' ORDER BY date',
                conn,
                params=params,
                index_col='date',
                parse_dates=['date']
            )
            
            # Get processed features
            query = '''
                SELECT date, feature_name, value
                FROM processed_data
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
                
            features_df = pd.read_sql_query(
                query + ' ORDER BY date',
                conn,
                params=params,
                parse_dates=['date']
            )
            
            # Pivot features table
            features_df = features_df.pivot(
                index='date',
                columns='feature_name',
                values='value'
            )
            
            return raw_df, features_df