import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sqlite3
from datetime import datetime, timedelta

class ProphetPredictor:
    def __init__(self):
        self.model = None
        
    def prepare_data(self, conn, ticker):
        """Prepare data for Prophet model."""
        query = """
        SELECT
            r.date as ds,
            r.open,
            r.close,
            r.volume
        FROM raw_market_data r
        WHERE r.ticker = ?
        ORDER BY r.date;
        """
        try:
            # Check if connection is SQLAlchemy engine or direct connection
            if hasattr(conn, 'execute') and callable(conn.execute):
                # SQLAlchemy engine
                try:
                    # Use pandas read_sql_query with SQLAlchemy
                    df = pd.read_sql_query(
                        query.replace('?', ':ticker'), 
                        conn, 
                        params={"ticker": ticker}
                    )
                except Exception as e:
                    print(f"Error using SQLAlchemy: {str(e)}")
                    # Fallback to direct execution
                    cursor = conn.connect().execute(
                        query.replace('?', ':ticker'), 
                        {"ticker": ticker}
                    )
                    rows = cursor.fetchall()
                    if not rows:
                        print("Warning: Query returned 0 rows")
                        return None
                    column_names = [col[0] for col in cursor.description]
                    df = pd.DataFrame(rows, columns=column_names)
            else:
                # Direct connection
                cursor = conn.cursor()
                cursor.execute(query, (ticker,))
                rows = cursor.fetchall()
                if not rows:
                    print("Warning: Query returned 0 rows")
                    return None
                column_names = [description[0] for description in cursor.description]
                df = pd.DataFrame(rows, columns=column_names)
            
            df['ds'] = pd.to_datetime(df['ds'])
            print(f"Query returned {len(df)} rows")
            
            if len(df) == 0:
                print("Warning: Query returned 0 rows")
                return None
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None
        
        # Create target variable (1-day ahead price)
        df['y'] = df['close'].shift(-1).astype(float)
        
        # Calculate additional features
        df['range'] = df['open'].astype(float) - df['close'].astype(float)
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['volatility'] = df['close'].rolling(window=20).std()
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['volume'] = df['volume'].astype(float)
        
        # Prophet requires no missing values
        df = df.dropna()
        
        # Get raw data dates
        dates_query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM raw_market_data"
        dates = pd.read_sql_query(dates_query, conn)
        min_date = pd.to_datetime(dates['min_date'].iloc[0])
        max_date = pd.to_datetime(dates['max_date'].iloc[0])
        
        # Calculate split dates
        total_days = (max_date - min_date).days
        train_end = min_date + pd.Timedelta(days=int(total_days * 0.7))
        val_end = min_date + pd.Timedelta(days=int(total_days * 0.9))
        
        # Apply splits
        df['split'] = 'train'
        mask = (df['ds'] > train_end)
        df.loc[mask, 'split'] = 'validation'
        mask = (df['ds'] > val_end)
        df.loc[mask, 'split'] = 'test'
        
        return df
        
    def train(self, conn, ticker):
        """Train Prophet model."""
        # Prepare data
        df = self.prepare_data(conn, ticker)
        
        # Get training data
        train_data = df[df['split'].isin(['train', 'validation'])]
        
        # Initialize model with settings from experiment
        self.model = Prophet(
            seasonality_mode='multiplicative',
            changepoints=['2008-10-13', '2020-03-16'],  # Major market events
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add US holidays
        self.model.add_country_holidays(country_name='US')
        
        # Add regressors
        self.model.add_regressor('volatility')
        self.model.add_regressor('range')
        self.model.add_regressor('ma20')
        self.model.add_regressor('ma50')
        self.model.add_regressor('day_of_week')
        
        # Fit model
        self.model.fit(train_data[['ds', 'y', 'volatility', 'range', 'ma20', 'ma50', 'day_of_week']])
        
        return self.model
    
    def predict(self, conn, ticker):
        """Generate predictions for all splits."""
        if self.model is None:
            self.train(conn, ticker)
        
        # Get data
        df = self.prepare_data(conn, ticker)
        
        # Make predictions for each split
        predictions = {}
        for split in ['train', 'validation', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) == 0:
                continue
            
            # Get the latest date from raw data
            latest_date = df['ds'].iloc[-1]
            
            # Create future dates starting from next day
            future_dates = pd.date_range(
                start=latest_date + pd.Timedelta(days=1),
                periods=3,
                freq='D'
            )
            
            # Create future dataframe for current period plus future dates
            in_sample_dates = split_data['ds']
            future = pd.DataFrame({'ds': pd.concat([
                pd.Series(in_sample_dates),
                pd.Series(future_dates)
            ])})
            
            # Add features for in-sample dates
            future_in_sample = future[future['ds'].isin(in_sample_dates)]
            future_in_sample = pd.merge(
                future_in_sample,
                split_data[['ds', 'volume', 'range', 'ma20', 'ma50', 'volatility', 'day_of_week']],
                on='ds',
                how='left'
            )
            
            # Add features for future dates
            future_out_sample = future[future['ds'].isin(future_dates)].copy()
            future_out_sample['volume'] = split_data['volume'].iloc[-1]
            future_out_sample['range'] = split_data['range'].iloc[-1]
            future_out_sample['ma20'] = split_data['ma20'].iloc[-1]
            future_out_sample['ma50'] = split_data['ma50'].iloc[-1]
            future_out_sample['volatility'] = split_data['volatility'].iloc[-1]
            future_out_sample['day_of_week'] = future_out_sample['ds'].dt.dayofweek
            
            # Combine and make predictions
            future_all = pd.concat([future_in_sample, future_out_sample])
            forecast = self.model.predict(future_all)
            
            # Split predictions into in-sample and out-of-sample
            in_sample_mask = forecast['ds'].isin(in_sample_dates)
            out_sample_mask = forecast['ds'].isin(future_dates)
            
            predictions[split] = {
                'in_sample': {
                    'dates': forecast[in_sample_mask]['ds'],
                    'predicted_value': forecast[in_sample_mask]['yhat'],
                    'confidence_lower': forecast[in_sample_mask]['yhat_lower'],
                    'confidence_upper': forecast[in_sample_mask]['yhat_upper']
                },
                'out_of_sample': {
                    'dates': forecast[out_sample_mask]['ds'],
                    'predicted_value': forecast[out_sample_mask]['yhat'],
                    'confidence_lower': forecast[out_sample_mask]['yhat_lower'],
                    'confidence_upper': forecast[out_sample_mask]['yhat_upper']
                }
            }
        
        return predictions, df
    
    def evaluate(self, conn, ticker):
        """Evaluate model performance using trading metrics"""
        predictions, df = self.predict(conn, ticker)
        
        metrics = {}
        for split in ['train', 'validation', 'test']:
            if split not in predictions:
                continue
                
            split_data = df[df['split'] == split]
            pred = pd.Series(predictions[split]['in_sample']['predicted_value'].values,
                           index=split_data.index)
            actual = split_data['y']
            close = split_data['close']
            
            # Ensure same length and index alignment
            common_idx = pred.index.intersection(actual.index)
            pred = pred[common_idx]
            actual = actual[common_idx]
            close = close[common_idx]
            
            # Trading signals
            signals = pred > close
            returns = (actual - close) / close * 100
            
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
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            
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
        """Update predictions in database."""
        # Clear existing predictions and metrics
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prophet_predictions WHERE ticker = ?", (ticker,))
        cursor.execute("DELETE FROM model_performance WHERE ticker = ? AND model = 'prophet'", (ticker,))
        conn.commit()
        
        # Generate predictions
        predictions, _ = self.predict(conn, ticker)
        
        # Store predictions for test set
        if 'test' in predictions:
            test_pred = predictions['test']
            
            # Store in-sample predictions
            in_sample_df = pd.DataFrame({
                'date': [d.strftime('%Y-%m-%d') for d in test_pred['in_sample']['dates']],
                'ticker': ticker,
                'predicted_value': test_pred['in_sample']['predicted_value'],
                'confidence_lower': test_pred['in_sample']['confidence_lower'],
                'confidence_upper': test_pred['in_sample']['confidence_upper'],
                'is_future': False
            })
            
            # Store out-of-sample predictions
            out_sample_df = pd.DataFrame({
                'date': [d.strftime('%Y-%m-%d') for d in test_pred['out_of_sample']['dates']],
                'ticker': ticker,
                'predicted_value': test_pred['out_of_sample']['predicted_value'],
                'confidence_lower': test_pred['out_of_sample']['confidence_lower'],
                'confidence_upper': test_pred['out_of_sample']['confidence_upper'],
                'is_future': True
            })
            
            # Combine predictions
            predictions_df = pd.concat([in_sample_df, out_sample_df])
            
            # Delete existing predictions for this ticker
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prophet_predictions WHERE ticker = ?", (ticker,))
            conn.commit()
            
            # Insert new predictions using batch inserts
            # Define batch size
            BATCH_SIZE = 500
            
            # Convert DataFrame to list of tuples
            prediction_values = predictions_df.values.tolist()
            
            print(f"Processing {len(prediction_values)} rows in batches of {BATCH_SIZE}...")
            
            cursor = conn.cursor()
            
            try:
                for i in range(0, len(prediction_values), BATCH_SIZE):
                    batch = prediction_values[i:i+BATCH_SIZE]
                    placeholders = ','.join(['(?, ?, ?, ?, ?, ?)'] * len(batch))
                    flattened_values = [val for row in batch for val in row]
                    
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO prophet_predictions 
                        (date, ticker, predicted_value, confidence_lower, confidence_upper, is_future) 
                        VALUES {placeholders}
                    """, flattened_values)
                    
                    print(f"Inserted batch {i//BATCH_SIZE + 1}/{(len(prediction_values) + BATCH_SIZE - 1)//BATCH_SIZE} for prophet_predictions")
                
                # Commit all changes
                conn.commit()
                print(f"Successfully inserted {len(prediction_values)} predictions using batch inserts")
                
            except Exception as e:
                print(f"Error during batch insert: {str(e)}")
                conn.rollback()
                raise  # Re-raise the exception to handle it at a higher level
        
        # Update performance metrics
        metrics = self.evaluate(conn, ticker)
        
        # Store metrics using batch insert
        metrics_data = [{
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'model': 'prophet',
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
            print(f"Successfully inserted metrics for Prophet model")
        except Exception as e:
            print(f"Error inserting metrics: {str(e)}")
            conn.rollback()
            raise
        
        return predictions_df, metrics['test']

if __name__ == "__main__":
    import os
    import warnings
    from db_connection import get_db_connection
    
    # Use SQLite Cloud if environment variable is set
    use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
    
    # Connect to database
    conn = get_db_connection(use_cloud=use_cloud)
    
    model = ProphetPredictor()
    predictions, metrics = model.update_predictions(conn, 'QQQ')
    print("Predictions:", predictions)
    print("Metrics:", metrics)
    conn.close()
