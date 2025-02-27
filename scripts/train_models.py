import os
import sys
import logging
from datetime import datetime

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
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM raw_market_data")
    return [row[0] for row in cursor.fetchall()]

def clear_predictions(conn):
    """Clear all prediction tables."""
    cursor = conn.cursor()
    tables = ['arima_predictions', 'prophet_predictions', 'dnn_predictions', 'model_performance']
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
    conn.commit()
    logger.info("Cleared prediction tables")

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
        
        # Connect to database
        conn = get_db_connection(use_cloud=use_cloud)
        
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
        
        conn.close()
        logger.info("All models updated successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in train_and_update_all_models: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting model training and updates...")
    results = train_and_update_all_models(['QQQ'])  # Default to QQQ
    logger.info("Completed model training and updates")
