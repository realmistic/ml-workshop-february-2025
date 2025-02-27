#!/usr/bin/env python3
"""
Script to initialize SQLite Cloud database.
"""

import os
import sys
import sqlite3
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connection import get_db_connection

def init_cloud_db():
    """Initialize SQLite Cloud database."""
    if "SQLITECLOUD_URL" not in os.environ:
        print("Error: SQLITECLOUD_URL environment variable not set")
        return False
    
    if os.environ.get("USE_SQLITECLOUD", "0").lower() not in ("1", "true", "yes"):
        print("Error: USE_SQLITECLOUD environment variable not set to '1'")
        return False
    
    # Connect to cloud database
    cloud_conn = get_db_connection(use_cloud=True)
    
    # Tables to create
    tables = [
        # Raw market data
        """
        CREATE TABLE IF NOT EXISTS raw_market_data (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # ARIMA features
        """
        CREATE TABLE IF NOT EXISTS arima_features (
            date TEXT,
            ticker TEXT,
            close REAL,
            returns REAL,
            ma5 REAL,
            ma20 REAL,
            ma50 REAL,
            rsi14 REAL,
            bb_upper REAL,
            bb_lower REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # Prophet features
        """
        CREATE TABLE IF NOT EXISTS prophet_features (
            date TEXT,
            ticker TEXT,
            close REAL,
            ds TEXT,
            y REAL,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # DNN features
        """
        CREATE TABLE IF NOT EXISTS dnn_features (
            date TEXT,
            ticker TEXT,
            close REAL,
            returns REAL,
            returns_3d REAL,
            returns_5d REAL,
            returns_10d REAL,
            returns_20d REAL,
            volatility_5d REAL,
            volatility_10d REAL,
            volatility_20d REAL,
            ma5 REAL,
            ma20 REAL,
            ma50 REAL,
            rsi14 REAL,
            bb_upper REAL,
            bb_lower REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # ARIMA predictions
        """
        CREATE TABLE IF NOT EXISTS arima_predictions (
            date TEXT,
            ticker TEXT,
            predicted_value REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            is_future INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # Prophet predictions
        """
        CREATE TABLE IF NOT EXISTS prophet_predictions (
            date TEXT,
            ticker TEXT,
            predicted_value REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            is_future INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # DNN predictions
        """
        CREATE TABLE IF NOT EXISTS dnn_predictions (
            date TEXT,
            ticker TEXT,
            predicted_value REAL,
            confidence_lower REAL,
            confidence_upper REAL,
            is_future INTEGER,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # Model performance
        """
        CREATE TABLE IF NOT EXISTS model_performance (
            date TEXT,
            model TEXT,
            ticker TEXT,
            mae REAL,
            rmse REAL,
            win_rate REAL,
            loss_rate REAL,
            uncond_win_rate REAL,
            uncond_loss_rate REAL,
            avg_return REAL,
            pl_ratio REAL,
            trading_freq REAL,
            n_trades INTEGER,
            PRIMARY KEY (date, model, ticker)
        )
        """
    ]
    
    # Create tables
    cursor = cloud_conn.cursor()
    for table_sql in tables:
        try:
            cursor.execute(table_sql)
        except Exception as e:
            print(f"Error creating table: {str(e)}")
            print(f"SQL: {table_sql}")
    
    # Commit changes
    cloud_conn.commit()
    cloud_conn.close()
    
    print("Cloud database initialized successfully!")
    return True

if __name__ == "__main__":
    if not init_cloud_db():
        sys.exit(1)
