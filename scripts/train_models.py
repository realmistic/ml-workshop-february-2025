import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Try to find .env in multiple locations
env_paths = [
    Path('.') / '.env',  # Current directory
    Path(__file__).parent / '.env',  # Same directory as this file
    Path(__file__).parent.parent / '.env'  # Project root directory
]

for env_path in env_paths:
    if env_path.exists():
        print(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        break

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_db_connection
from models.arima_model import ARIMAPredictor
from models.prophet_model import ProphetPredictor
from models.dnn_model import DNNPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_available_tickers(conn):
    """Get list of tickers available in the database."""
    try:
        # Check if this is a SQLite Cloud connection
        is_cloud_conn = 'sqlitecloud' in str(type(conn))
        
        if is_cloud_conn:
            # Import the read_data function from db_connection
            from db_connection import read_data
            
            # Use the read_data function for SQLite Cloud
            result = read_data(
                "SELECT DISTINCT ticker FROM raw_market_data",
                conn=conn,
                close_conn=False
            )
            return result['ticker'].tolist()
        elif hasattr(conn, 'execute') and callable(conn.execute):
            # SQLAlchemy engine
            from sqlalchemy import text
            result = pd.read_sql_query(
                "SELECT DISTINCT ticker FROM raw_market_data",
                conn
            )
            return result['ticker'].tolist()
        else:
            # Direct connection
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM raw_market_data")
            return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error getting available tickers: {str(e)}")
        # Return default ticker if we can't get the list
        return ['QQQ']

def clear_predictions(conn):
    """Clear all prediction tables."""
    tables = ['arima_predictions', 'prophet_predictions', 'dnn_predictions', 'model_performance']
    
    try:
        # Check if this is a SQLite Cloud connection
        is_cloud_conn = 'sqlitecloud' in str(type(conn))
        
        if is_cloud_conn:
            # Import the read_data function from db_connection
            from db_connection import read_data
            
            # Use the read_data function for SQLite Cloud
            for table in tables:
                read_data(
                    f"DELETE FROM {table}",
                    conn=conn,
                    close_conn=False
                )
        elif hasattr(conn, 'execute') and callable(conn.execute):
            # SQLAlchemy engine
            from sqlalchemy import text
            for table in tables:
                conn.execute(text(f"DELETE FROM {table}"))
            conn.commit()
        else:
            # Direct connection
            cursor = conn.cursor()
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
            conn.commit()
        
        logger.info("Cleared prediction tables")
    except Exception as e:
        logger.error(f"Error clearing prediction tables: {str(e)}")
        # Continue execution even if clearing fails

def train_and_update_all_models(tickers=None, use_cloud=None):
    """
    Train all models and update predictions for specified tickers.
    
    Args:
        tickers: List of tickers to update (default: all tickers in database)
        use_cloud: Whether to use SQLite Cloud (default: use environment variable)
    """
    try:
        # Check if we should use SQLite Cloud
        if use_cloud is None:
            use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
        
        # Connect to database with direct connection to ensure compatibility with all models
        conn = get_db_connection(use_cloud=use_cloud, direct_connection=True)
        
        # Print connection info
        if use_cloud:
            logger.info("Connected to SQLite Cloud database")
        else:
            logger.info("Connected to local SQLite database")
        
        # Clear previous predictions
        clear_predictions(conn)
        
        # Get available tickers if none specified
        if tickers is None:
            tickers = get_available_tickers(conn)
        elif isinstance(tickers, str):
            tickers = [tickers]
        
        # Initialize models
        models = {
            'ARIMA': ARIMAPredictor(),
            'Prophet': ProphetPredictor(),
            'DNN': DNNPredictor()
        }
        
        results = {}
        
        # Train and update predictions for each ticker and model
        for ticker in tickers:
            results[ticker] = {}
            logger.info(f"Processing ticker: {ticker}")
            
            for name, model in models.items():
                try:
                    logger.info(f"Training {name} model for {ticker}...")
                    predictions, metrics = model.update_predictions(conn, ticker)
                    
                    results[ticker][name] = {
                        'predictions': predictions,
                        'metrics': metrics
                    }
                    
                    logger.info(f"{name} model updated successfully for {ticker}")
                    logger.info(f"{name} metrics for {ticker}: {metrics}")
                    
                except Exception as e:
                    logger.error(f"Error updating {name} model for {ticker}: {str(e)}")
                    continue
        
        # Close connection based on its type
        if hasattr(conn, 'dispose') and callable(conn.dispose):
            # SQLAlchemy engine
            conn.dispose()
        elif hasattr(conn, 'close') and callable(conn.close):
            # Direct connection
            conn.close()
        
        logger.info("All models updated successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in train_and_update_all_models: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting model training and updates...")
    
    # Print environment variables for debugging (without exposing values)
    logger.info(f"USE_SQLITECLOUD is set: {'yes' if 'USE_SQLITECLOUD' in os.environ else 'no'}")
    logger.info(f"SQLITECLOUD_URL is set: {'yes' if 'SQLITECLOUD_URL' in os.environ else 'no'}")
    
    results = train_and_update_all_models(['QQQ'])  # Default to QQQ
    logger.info("Completed model training and updates")
