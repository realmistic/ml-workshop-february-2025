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

def init_cloud_db(force=False):
    """Initialize SQLite Cloud database."""
    if "SQLITECLOUD_URL" not in os.environ:
        print("Error: SQLITECLOUD_URL environment variable not set")
        return False
    
    if not force and os.environ.get("USE_SQLITECLOUD", "0").lower() not in ("1", "true", "yes"):
        print("Error: USE_SQLITECLOUD environment variable not set to '1'")
        print("Use --force to override this check")
        return False
    
    print(f"Initializing SQLite Cloud database with URL: {os.environ['SQLITECLOUD_URL'][:20]}...")
    
    # Connect to cloud database
    try:
        cloud_conn = get_db_connection(use_cloud=True)
        print("Successfully connected to SQLite Cloud")
    except Exception as e:
        print(f"Error connecting to SQLite Cloud: {str(e)}")
        return False
    
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker)
        )
        """,
        
        # ARIMA features
        """
        CREATE TABLE IF NOT EXISTS arima_features (
            date TEXT,
            ticker TEXT,
            returns REAL,
            volatility REAL,
            ma_5 REAL,
            ma_20 REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
        )
        """,
        
        # Prophet features
        """
        CREATE TABLE IF NOT EXISTS prophet_features (
            date TEXT,
            ticker TEXT,
            y REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
        )
        """,
        
        # DNN features
        """
        CREATE TABLE IF NOT EXISTS dnn_features (
            date TEXT,
            ticker TEXT,
            returns REAL,
            volatility REAL,
            ma_5 REAL,
            ma_20 REAL,
            rsi REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
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
            is_future BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker, created_at),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
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
            is_future BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker, created_at),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
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
            is_future BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker, created_at),
            FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
        )
        """,
        
        # Model performance
        """
        CREATE TABLE IF NOT EXISTS model_performance (
            date TEXT,
            ticker TEXT,
            model TEXT,
            mae REAL,
            rmse REAL,
            accuracy REAL,
            win_rate REAL,
            loss_rate REAL,
            uncond_win_rate REAL,
            uncond_loss_rate REAL,
            avg_return REAL,
            n_trades INTEGER,
            trading_freq REAL,
            pl_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (date, ticker, model)
        )
        """
    ]
    
    # Drop existing tables and create new ones
    cursor = cloud_conn.cursor()
    
    # Get existing tables
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        print(f"Existing tables: {existing_tables}")
        
        # Drop tables in reverse order to avoid foreign key constraints
        for table in reversed(['raw_market_data', 'arima_features', 'prophet_features', 'dnn_features',
                              'arima_predictions', 'prophet_predictions', 'dnn_predictions', 'model_performance']):
            if table in existing_tables:
                print(f"Dropping table: {table}")
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
    except Exception as e:
        print(f"Error dropping tables: {str(e)}")
    
    # Create tables
    for table_sql in tables:
        try:
            print(f"Executing: {table_sql[:50]}...")
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
