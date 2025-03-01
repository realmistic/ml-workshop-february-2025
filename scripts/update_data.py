import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import logging
import yfinance as yf
import ssl
import urllib3
import certifi
from io import StringIO
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_connection import get_db_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_stock_data(symbol, start_date):
    """
    Download stock data with fallback mechanism:
    1. Try yfinance first
    2. If fails, try stooq with SSL verification disabled
    """
    def try_yfinance():
        try:
            yticker = yf.Ticker(symbol)
            df = yticker.history(start=start_date)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"yfinance error: {str(e)}")
        return None
    
    def try_stooq():
        try:
            # Create a custom SSL context that doesn't verify certificates
            http = urllib3.PoolManager(
                cert_reqs='CERT_NONE',
                ca_certs=certifi.where()
            )
            
            # Convert symbol to stooq format if needed
            stooq_symbol = f"{symbol.lower()}.us" if not symbol.endswith('.us') else symbol.lower()
            url = f'https://stooq.com/q/d/l/?s={stooq_symbol}&i=d'
            
            response = http.request('GET', url)
            
            if response.status == 200:
                # Use StringIO to create a file-like object from the response data
                csv_data = StringIO(response.data.decode('utf-8'))
                df = pd.read_csv(csv_data)
                
                if not df.empty:
                    # Rename columns to match our expected format
                    df.columns = [col.capitalize() for col in df.columns]
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    # Sort index in ascending order (oldest to newest)
                    df.sort_index(inplace=True)
                    return df
        except Exception as e:
            logger.error(f"stooq error: {str(e)}")
        return None
    
    try:
        # Try yfinance first
        logger.info("Attempting to download from yfinance...")
        df = try_yfinance()
        
        # If yfinance fails, try stooq
        if df is None:
            logger.info("yfinance failed, trying stooq...")
            df = try_stooq()
        
        if df is None:
            raise ValueError(f"Failed to download data for {symbol} from all sources")
        
        logger.info(f"Successfully downloaded {len(df)} days of {symbol} data")
        
        # Basic validation
        required_columns = ['Open', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {str(e)}")
        return None

def calculate_features(df):
    """Calculate features for all models."""
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values (first few rows will have NaN due to rolling calculations)
    df = df.dropna()
    
    return df

def update_market_data(tickers=None, use_cloud=None):
    """
    Fetch and update market data for specified tickers.
    
    Args:
        tickers: List of tickers to update (default: ['QQQ'])
        use_cloud: Whether to use SQLite Cloud (default: use environment variable)
    """
    if tickers is None:
        tickers = ['QQQ']  # Default to QQQ
    elif isinstance(tickers, str):
        tickers = [tickers]
    
    # Check if we should use SQLite Cloud
    if use_cloud is None:
        use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
    
    # Connect to database
    conn = get_db_connection(use_cloud=use_cloud)
    
    # Print connection info
    if use_cloud:
        print(f"Connected to SQLite Cloud database")
    else:
        print(f"Connected to local SQLite database")
    
    for ticker in tickers:
        try:
            # Get the latest date for this ticker in our database
            latest_date = pd.read_sql_query(
                "SELECT MAX(date) as max_date FROM raw_market_data WHERE ticker = ?",
                conn,
                params=(ticker,)
            ).iloc[0]['max_date']
            
            # Always get at least the last 7 days of data to ensure we have the latest
            if latest_date:
                latest_date_obj = datetime.strptime(latest_date, '%Y-%m-%d').date()
                today = datetime.now().date()
                days_diff = (today - latest_date_obj).days
                
                if days_diff <= 7:
                    # If the latest date is within the last 7 days, get data from 7 days before the latest date
                    start_date = latest_date_obj - timedelta(days=7)
                else:
                    # Otherwise, get data from the day after the latest date
                    start_date = latest_date_obj + timedelta(days=1)
            else:
                start_date = '1999-01-01'
            
            print(f"Using start date: {start_date}")
        except Exception as e:
            print(f"Error determining start date: {str(e)}")
            start_date = '1999-01-01'
        
        # Get market data
        logger.info(f"Downloading {ticker} data from {start_date}")
        market_data = download_stock_data(ticker, start_date)
        
        if market_data is None or len(market_data) == 0:
            logger.warning(f"No new data to update for {ticker}")
            continue
        
        # Prepare raw data
        raw_data = market_data.reset_index()
        raw_data.columns = ['date'] + [col.lower() for col in raw_data.columns[1:]]
        raw_data['date'] = raw_data['date'].dt.strftime('%Y-%m-%d')
        raw_data['ticker'] = ticker
        
        # Calculate features
        features_df = calculate_features(raw_data.copy())
        
        # Sync raw_data with features_df after NaN removal
        raw_data = raw_data[raw_data['date'].isin(features_df['date'])]
        
        # Get existing dates for this ticker
        existing_dates = pd.read_sql_query(
            "SELECT date FROM raw_market_data WHERE ticker = ?",
            conn,
            params=(ticker,)
        )['date'].tolist()
        
        # Filter out existing dates
        new_data = raw_data[~raw_data['date'].isin(existing_dates)]
        new_features = features_df[~features_df['date'].isin(existing_dates)]
        new_features['ticker'] = ticker
        
        if len(new_data) == 0:
            logger.info(f"No new data to store for {ticker}")
            continue
        
        logger.info(f"Storing {len(new_data)} new records for {ticker}")
        
        # Use batch inserts for better performance
        print("Using batch inserts...")
        
        cursor = conn.cursor()
        
        # Define batch size
        BATCH_SIZE = 500
        
        try:
            # Insert raw data in batches
            raw_data_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
            raw_data_values = new_data[raw_data_columns].values.tolist()
            
            print(f"Processing {len(raw_data_values)} rows in batches of {BATCH_SIZE}...")
            
            for i in range(0, len(raw_data_values), BATCH_SIZE):
                batch = raw_data_values[i:i+BATCH_SIZE]
                placeholders = ','.join(['(?, ?, ?, ?, ?, ?, ?)'] * len(batch))
                flattened_values = [val for row in batch for val in row]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO raw_market_data 
                    (date, ticker, open, high, low, close, volume) 
                    VALUES {placeholders}
                """, flattened_values)
                
                print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(raw_data_values) + BATCH_SIZE - 1)//BATCH_SIZE} for raw_market_data")
            
            # Prepare and insert ARIMA features in batches
            arima_features = new_features[['date', 'ticker', 'returns', 'volatility', 'ma_5', 'ma_20']].copy()
            arima_values = arima_features.values.tolist()
            
            for i in range(0, len(arima_values), BATCH_SIZE):
                batch = arima_values[i:i+BATCH_SIZE]
                placeholders = ','.join(['(?, ?, ?, ?, ?, ?)'] * len(batch))
                flattened_values = [val for row in batch for val in row]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO arima_features 
                    (date, ticker, returns, volatility, ma_5, ma_20) 
                    VALUES {placeholders}
                """, flattened_values)
                
                print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(arima_values) + BATCH_SIZE - 1)//BATCH_SIZE} for arima_features")
            
            # Prepare and insert Prophet features in batches
            prophet_features = new_features[['date', 'ticker', 'close']].copy()
            prophet_features.columns = ['date', 'ticker', 'y']
            prophet_values = prophet_features.values.tolist()
            
            for i in range(0, len(prophet_values), BATCH_SIZE):
                batch = prophet_values[i:i+BATCH_SIZE]
                placeholders = ','.join(['(?, ?, ?)'] * len(batch))
                flattened_values = [val for row in batch for val in row]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO prophet_features 
                    (date, ticker, y) 
                    VALUES {placeholders}
                """, flattened_values)
                
                print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(prophet_values) + BATCH_SIZE - 1)//BATCH_SIZE} for prophet_features")
            
            # Prepare and insert DNN features in batches
            dnn_features = new_features[['date', 'ticker', 'returns', 'volatility', 'ma_5', 'ma_20', 'rsi']].copy()
            dnn_values = dnn_features.values.tolist()
            
            for i in range(0, len(dnn_values), BATCH_SIZE):
                batch = dnn_values[i:i+BATCH_SIZE]
                placeholders = ','.join(['(?, ?, ?, ?, ?, ?, ?)'] * len(batch))
                flattened_values = [val for row in batch for val in row]
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO dnn_features 
                    (date, ticker, returns, volatility, ma_5, ma_20, rsi) 
                    VALUES {placeholders}
                """, flattened_values)
                
                print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(dnn_values) + BATCH_SIZE - 1)//BATCH_SIZE} for dnn_features")
            
            # Commit all changes
            conn.commit()
            print(f"Successfully inserted {len(new_data)} records using batch inserts")
            
        except Exception as e:
            print(f"Error during batch insert: {str(e)}")
            conn.rollback()
            
            # Fallback to individual inserts if batch insert fails
            print("Falling back to individual inserts...")
            
            # Insert raw data
            for _, row in new_data[raw_data_columns].iterrows():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO raw_market_data (date, ticker, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row['date'], row['ticker'], row['open'], row['high'], row['low'], row['close'], row['volume'])
                    )
                except Exception as insert_error:
                    print(f"Error inserting raw data: {str(insert_error)}")
            
            # Insert ARIMA features
            for _, row in arima_features.iterrows():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO arima_features (date, ticker, returns, volatility, ma_5, ma_20) VALUES (?, ?, ?, ?, ?, ?)",
                        (row['date'], row['ticker'], row['returns'], row['volatility'], row['ma_5'], row['ma_20'])
                    )
                except Exception as insert_error:
                    print(f"Error inserting ARIMA features: {str(insert_error)}")
            
            # Insert Prophet features
            for _, row in prophet_features.iterrows():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO prophet_features (date, ticker, y) VALUES (?, ?, ?)",
                        (row['date'], row['ticker'], row['y'])
                    )
                except Exception as insert_error:
                    print(f"Error inserting Prophet features: {str(insert_error)}")
            
            # Insert DNN features
            for _, row in dnn_features.iterrows():
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO dnn_features (date, ticker, returns, volatility, ma_5, ma_20, rsi) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row['date'], row['ticker'], row['returns'], row['volatility'], row['ma_5'], row['ma_20'], row['rsi'])
                    )
                except Exception as insert_error:
                    print(f"Error inserting DNN features: {str(insert_error)}")
            
            # Commit all changes
            conn.commit()
            print(f"Inserted {len(new_data)} records using individual inserts (fallback)")
        
        print(f"Updated {ticker} data from {start_date} to {raw_data['date'].iloc[-1]}")
    
    conn.close()

if __name__ == "__main__":
    update_market_data()  # Will use QQQ by default
