import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import sqlite3
from datetime import datetime
import warnings

warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

def download_stock_data(ticker_symbol):
    """Download stock data with error handling and validation"""
    try:
        # Create ticker object and get history
        yticker = yf.Ticker(ticker_symbol)
        df = yticker.history(period='max')
        
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker_symbol}")
            
        print(f"Downloaded {len(df)} days of {ticker_symbol} data")
        
        # Basic validation
        required_columns = ['Open', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    except Exception as e:
        print(f"Error downloading {ticker_symbol}: {str(e)}")
        return None

def transform_stock_data(df):
    """Transform stock data for ARIMA"""
    if df is None:
        print("Error: df is None in transform_stock_data")
        return None
    
    print(f"DataFrame in transform_stock_data has {len(df)} rows")
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Reset index to make the date a column
    df = df.reset_index()
    
    # Create DataFrame with Prophet-like structure
    arima_df = pd.DataFrame()
    arima_df['ds'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone
    arima_df['y'] = df['close'].shift(-3).astype(float)  # 3-day ahead target
    
    # Add additional features
    arima_df['volume'] = df['volume'].astype(float)
    arima_df['open'] = df['open'].astype(float)
    arima_df['close'] = df['close'].astype(float)
    arima_df['range'] = df['open'].astype(float) - df['close'].astype(float)
    
    # Calculate moving averages
    arima_df['ma20'] = df['close'].rolling(window=20).mean()
    arima_df['ma50'] = df['close'].rolling(window=50).mean()
    
    # Calculate volatility (20-day rolling standard deviation)
    arima_df['volatility'] = df['close'].rolling(window=20).std()
    
    # Add day of week as a feature
    arima_df['day_of_week'] = arima_df['ds'].dt.dayofweek
    
    # Clean data by removing NaN values (e.g. first dates with MA non-defined)
    arima_df = arima_df.dropna()

    # Split data into train/validation/test sets (70/20/10)
    total_days = len(arima_df)
    train_end = int(total_days * 0.7)
    val_end = int(total_days * 0.9)
    
    # Initialize split column
    arima_df['split'] = 'train'
    arima_df.loc[train_end:val_end-1, 'split'] = 'validation'
    arima_df.loc[val_end:, 'split'] = 'test'
    
    return arima_df

def train_arima_model(df, order=(2,1,2)):
    """Train ARIMA model on price data"""
    # Get training data
    train_data = df[df['split'].isin(['train', 'validation'])]
    
    # Prepare exogenous variables
    exog = train_data[['volume', 'range', 'ma20', 'ma50', 'volatility', 'day_of_week']]
    
    # Train on prices
    model = ARIMA(train_data['close'], 
                  order=order,
                  exog=exog)
    
    return model.fit()

def make_predictions(model, df):
    """Make price predictions"""
    # Split data into train/validation/test
    train_mask = df['split'].isin(['train', 'validation'])
    train_data = df[train_mask]
    test_data = df[~train_mask]
    
    # Prepare exogenous variables
    train_exog = train_data[['volume', 'range', 'ma20', 'ma50', 'volatility', 'day_of_week']]
    test_exog = test_data[['volume', 'range', 'ma20', 'ma50', 'volatility', 'day_of_week']]

    # Get in-sample predictions for train/validation
    train_pred = model.predict(start=0, end=len(train_data)-1, exog=train_exog)

    # Get out-of-sample predictions for test
    test_pred = model.forecast(steps=len(test_data), exog=test_exog)

    #make sure index is the same for test_data and test_pred 
    test_pred.index = test_data.index  # Ensure index alignment
    # test_pred = pd.Series(test_pred, index=test_data.index)

    # Combine predictions
    predictions = pd.concat([train_pred, pd.Series(test_pred, index=test_data.index)])
    
    # Shift predictions forward by 3 days to match target
    predictions = predictions.shift(3)
    
    return predictions

class ARIMAPredictor:
    def __init__(self):
        self.model = None
        
    def predict(self, conn, ticker):
        """Generate predictions for ticker"""
        # Get raw data
        query = """
        SELECT
            r.date,
            r.open,
            r.close,
            r.volume
        FROM raw_market_data r
        WHERE r.ticker = ?
        ORDER BY r.date;
        """
        print(f"Executing query: {query} with params: {ticker}")
        try:
            # Try using pandas read_sql_query first
            try:
                df = pd.read_sql_query(query, conn, params=(ticker,))
            except Exception as pandas_error:
                print(f"Pandas read_sql_query failed: {str(pandas_error)}")
                print("Falling back to direct SQL execution...")
                
                # Fallback to direct SQL execution
                cursor = conn.cursor()
                cursor.execute(query, (ticker,))
                rows = cursor.fetchall()
                
                if not rows:
                    print("Warning: Query returned 0 rows")
                    return None
                
                # Get column names
                column_names = [description[0] for description in cursor.description]
                
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=column_names)
                
            print(f"Query returned {len(df)} rows")
            if len(df) == 0:
                print("Warning: Query returned 0 rows")
                return None
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None
        
        # Transform data
        df = transform_stock_data(df)

        # Train model if needed
        if self.model is None:
            self.model = train_arima_model(df)

        # Make predictions
        predictions = make_predictions(self.model, df)
        
        # Add predictions to DataFrame
        df['yhat'] = predictions
        df['yhat_lower'] = predictions * 0.95
        df['yhat_upper'] = predictions * 1.05

        # Store predictions for test set
        test_data = df[df['split'] == 'test']
        if len(test_data) > 0:
            # Store in-sample predictions
            predictions_df = pd.DataFrame({
                'date': test_data['ds'].dt.strftime(f'%Y-%m-%d'),
                'ticker': ticker,
                'predicted_value': test_data['yhat'],
                'confidence_lower': test_data['yhat_lower'],
                'confidence_upper': test_data['yhat_upper'],
                'is_future': False
            })
                       
            # Get future dates
            future_dates = pd.date_range(
                start=test_data['ds'].iloc[-1] + pd.Timedelta(days=1),
                periods=3,
                freq='D'
            )
            
            # Prepare future exogenous variables (use last known values)
            future_exog = df.iloc[-1:][['volume', 'range', 'ma20', 'ma50', 'volatility', 'day_of_week']].values.repeat(3, axis=0)
            future_exog[:, -1] = [(d.dayofweek) for d in future_dates]  # Update day_of_week
            
            # Get future predictions
            future_pred = self.model.forecast(steps=3, exog=future_exog)
            
            # Store out-of-sample predictions
            future_df = pd.DataFrame({
                'date': future_dates.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'predicted_value': future_pred,
                'confidence_lower': future_pred * 0.95,
                'confidence_upper': future_pred * 1.05,
                'is_future': True
            })
            
            # Combine predictions
            predictions_df = pd.concat([predictions_df, future_df])
            
            # Delete existing predictions
            cursor = conn.cursor()
            cursor.execute("DELETE FROM arima_predictions WHERE ticker = ?", (ticker,))
            conn.commit()
            
            # Insert new predictions using batch inserts
            # Define batch size
            BATCH_SIZE = 500
            
            # Convert DataFrame to list of tuples
            prediction_values = predictions_df.values.tolist()
            
            print(f"Processing {len(prediction_values)} predictions in batches of {BATCH_SIZE}...")
            
            cursor = conn.cursor()
            
            for i in range(0, len(prediction_values), BATCH_SIZE):
                batch = prediction_values[i:i+BATCH_SIZE]
                placeholders = ','.join(['(?, ?, ?, ?, ?, ?)'] * len(batch))
                flattened_values = [val for row in batch for val in row]
                
                try:
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO arima_predictions 
                        (date, ticker, predicted_value, confidence_lower, confidence_upper, is_future) 
                        VALUES {placeholders}
                    """, flattened_values)
                    
                    print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(prediction_values) + BATCH_SIZE - 1)//BATCH_SIZE} for arima_predictions")
                except Exception as batch_error:
                    print(f"Error during batch insert: {str(batch_error)}")
                    conn.rollback()
                    raise  # Re-raise the exception to handle it at a higher level
            
            conn.commit()
            print(f"Successfully inserted {len(prediction_values)} predictions using batch inserts")
        
        return df

    def evaluate(self, df):
        """Evaluate model performance using trading metrics"""
        metrics = {}
        for split in ['train', 'validation', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) == 0:
                continue
            
            # Trading signals
            signals = split_data['yhat'] > split_data['close']
            returns = (split_data['y'] - split_data['close']) / split_data['close'] * 100
            
            # Model Win/Loss metrics
            win_rate = ((signals) & (returns > 0)).sum() / signals.sum() * 100
            loss_rate = ((signals) & (returns < 0)).sum() / signals.sum() * 100
            
            # Unconditional metrics
            uncond_win_rate = (returns > 0).sum() / len(returns) * 100
            uncond_loss_rate = (returns < 0).sum() / len(returns) * 100
            
            # Returns and activity metrics
            avg_return = returns[signals].mean()
            n_trades = signals.sum()
            trading_freq = (signals.sum() / len(signals)) * 100
            
            # Risk metrics
            wins = returns[(signals) & (returns > 0)]
            losses = returns[(signals) & (returns < 0)]
            pl_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 else float('inf')
            
            # Standard metrics
            mae = np.mean(np.abs(split_data['y'] - split_data['yhat']))
            rmse = np.sqrt(np.mean((split_data['y'] - split_data['yhat']) ** 2))
            
            metrics[split] = {
                'mae': mae,
                'rmse': rmse,
                'accuracy': win_rate,  # Use win rate as accuracy
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'uncond_win_rate': uncond_win_rate,
                'uncond_loss_rate': uncond_loss_rate,
                'avg_return': avg_return,
                'n_trades': n_trades,
                'trading_freq': trading_freq,
                'pl_ratio': pl_ratio
            }
            
        return metrics
    
    def update_predictions(self, conn, ticker):
        """Update predictions in database"""
        # Generate predictions
        df = self.predict(conn, ticker)
        
        if df is None:
            print(f"Error: predict returned None for ticker {ticker}")
            return None, None
        
        # Update performance metrics
        try:
            metrics = self.evaluate(df)
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return df, None
        
        # Delete existing metrics
        cursor = conn.cursor()
        cursor.execute("DELETE FROM model_performance WHERE ticker = ? AND model = 'arima'", (ticker,))
        conn.commit()
        
        # Store new metrics using batch insert
        metrics_data = [{
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'model': 'arima',
            'mae': metrics['test']['mae'],
            'rmse': metrics['test']['rmse'],
            'accuracy': metrics['test']['accuracy'],
            'win_rate': metrics['test']['win_rate'],
            'loss_rate': metrics['test']['loss_rate'],
            'uncond_win_rate': metrics['test']['uncond_win_rate'],
            'uncond_loss_rate': metrics['test']['uncond_loss_rate'],
            'avg_return': metrics['test']['avg_return'],
            'n_trades': metrics['test']['n_trades'],
            'trading_freq': metrics['test']['trading_freq'],
            'pl_ratio': metrics['test']['pl_ratio']
        }]
        
        # Convert to list of values
        metrics_values = [[
            m['date'], m['ticker'], m['model'], 
            m['mae'], m['rmse'], m['accuracy'], 
            m['win_rate'], m['loss_rate'], 
            m['uncond_win_rate'], m['uncond_loss_rate'], 
            m['avg_return'], m['n_trades'], 
            m['trading_freq'], m['pl_ratio']
        ] for m in metrics_data]
        
        # Insert metrics using batch insert
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO model_performance 
                (date, ticker, model, mae, rmse, accuracy, win_rate, loss_rate, 
                uncond_win_rate, uncond_loss_rate, avg_return, n_trades, 
                trading_freq, pl_ratio) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, metrics_values[0])
            
            conn.commit()
            print(f"Successfully inserted metrics for ARIMA model")
        except Exception as e:
            print(f"Error inserting metrics: {str(e)}")
            conn.rollback()
            raise
        
        return df, metrics['test']

if __name__ == "__main__":
    import os
    import warnings
    import statsmodels.tools.sm_exceptions
    from db_connection import get_db_connection

    # Suppress specific warnings
    warnings.simplefilter("ignore", category=UserWarning)  # Suppresses ValueWarning in statsmodels
    warnings.simplefilter("ignore", category=FutureWarning)  # Suppresses FutureWarnings
    warnings.simplefilter("ignore", category=statsmodels.tools.sm_exceptions.ConvergenceWarning)  # Suppresses ConvergenceWarning
    
    # Use SQLite Cloud if environment variable is set
    use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
    
    # Connect to database
    conn = get_db_connection(use_cloud=use_cloud)
    
    model = ARIMAPredictor()
    predictions, metrics = model.update_predictions(conn, 'QQQ')
    print("Predictions:", predictions)
    print("Metrics:", metrics)
    conn.close()
