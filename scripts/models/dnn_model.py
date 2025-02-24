import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sqlite3
import os
from datetime import datetime, timedelta

class DNNPredictor:
    def __init__(self):
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def prepare_data(self, conn, ticker):
        """Prepare data for DNN model."""
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
        df = pd.read_sql_query(query, conn, params=(ticker,))
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Create feature columns
        feature_columns = []
        
        # Returns features (1-10, 30, 60 days)
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 60]:
            df[f'return_{i}d'] = df['close'].pct_change(periods=i) * 100
            feature_columns.append(f'return_{i}d')
        
        # Moving average returns (5, 10, 20, 30, 60 days)
        for window in [5, 10, 20, 30, 60]:
            ma = df['close'].rolling(window=window).mean()
            df[f'ma{window}_return'] = (df['close'] / ma - 1) * 100
            feature_columns.append(f'ma{window}_return')
        
        # Volatility features (5, 10, 20, 30, 60 days)
        for window in [5, 10, 20, 30, 60]:
            df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window=window).std() * 100
            df[f'volume_volatility_{window}d'] = df['volume'].pct_change().rolling(window=window).std() * 100
            feature_columns.append(f'volatility_{window}d')
            feature_columns.append(f'volume_volatility_{window}d')
        
        # Volume indicators
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume_ma5'] / df['volume_ma20']
        feature_columns.append('volume_ratio')
        
        # Target: 3-day future return
        df['y_return'] = (df['close'].shift(-3) / df['close'] - 1) * 100
        df['y'] = df['close'].shift(-3)  # Store actual future price for evaluation
        
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
        df.loc[df.index > train_end, 'split'] = 'validation'
        df.loc[df.index > val_end, 'split'] = 'test'
        
        # Drop rows with NaN
        df = df.dropna()
        
        # Scale features
        train_data = df[df['split'] == 'train']
        X = self.feature_scaler.fit_transform(train_data[feature_columns])
        y = self.target_scaler.fit_transform(train_data[['y_return']])
        
        return df, feature_columns
    
    def build_model(self, input_shape):
        """Build simple two-layer DNN model."""
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, conn, ticker):
        """Train DNN model with cyclic learning rate."""
        # Prepare data
        df, feature_columns = self.prepare_data(conn, ticker)
        
        # Get training data
        train_data = df[df['split'] == 'train']
        X_train = self.feature_scaler.transform(train_data[feature_columns])
        y_train = self.target_scaler.transform(train_data[['y_return']])
        
        # Get validation data
        val_data = df[df['split'] == 'validation']
        X_val = self.feature_scaler.transform(val_data[feature_columns])
        y_val = self.target_scaler.transform(val_data[['y_return']])
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            mode='min'
        )
        
        # Cyclic learning rate
        initial_learning_rate = 0.0005
        maximal_learning_rate = 0.005
        step_size = 8
        
        def cyclic_lr(epoch):
            cycle = np.floor(1 + epoch/(2 * step_size))
            x = np.abs(epoch/step_size - 2 * cycle + 1)
            lr = initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * max(0, (1-x))
            return lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cyclic_lr)
        
        # Build and train model
        self.model = self.build_model(len(feature_columns))
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,
            batch_size=32,
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )
        
        return self.model
    
    def predict(self, conn, ticker):
        """Generate predictions for all splits."""
        if self.model is None:
            self.train(conn, ticker)
        
        # Get data
        df, feature_columns = self.prepare_data(conn, ticker)
        
        # Make predictions for each split
        predictions = {}
        for split in ['train', 'validation', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) == 0:
                continue
                
            try:
                # Get features for current period
                X = self.feature_scaler.transform(split_data[feature_columns])
                
                # Predict returns
                pred_returns = self.target_scaler.inverse_transform(
                    self.model.predict(X, verbose=0)
                )
                
                # Store predicted returns
                predictions[split] = {
                    'in_sample': {
                        'dates': split_data.index,
                        'predicted_value': pred_returns.flatten(),
                        'confidence_lower': pred_returns.flatten() - 1.96 * np.std(pred_returns),
                        'confidence_upper': pred_returns.flatten() + 1.96 * np.std(pred_returns)
                    },
                    'out_of_sample': {
                        'dates': pd.date_range(
                            start=split_data.index[-1] + pd.Timedelta(days=1),
                            periods=3,
                            freq='D'
                        ),
                        'predicted_value': pred_returns[-1] * np.ones(3),
                        'confidence_lower': (pred_returns[-1] - 1.96 * np.std(pred_returns)) * np.ones(3),
                        'confidence_upper': (pred_returns[-1] + 1.96 * np.std(pred_returns)) * np.ones(3)
                    }
                }
            except Exception as e:
                print(f"Warning: Failed to generate predictions for {split} split: {str(e)}")
                predictions[split] = {
                    'in_sample': {
                        'dates': split_data.index,
                        'predicted_value': np.zeros(len(split_data)),
                        'confidence_lower': np.zeros(len(split_data)),
                        'confidence_upper': np.zeros(len(split_data))
                    },
                    'out_of_sample': {
                        'dates': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=3, freq='D'),
                        'predicted_value': np.zeros(3),
                        'confidence_lower': np.zeros(3),
                        'confidence_upper': np.zeros(3)
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
            pred = predictions[split]['in_sample']['predicted_value']
            actual = split_data['y_return']
            close = split_data['close']
            
            # Ensure same length
            min_len = min(len(actual), len(pred))
            actual = actual[:min_len]
            pred = pred[:min_len]
            close = close[:min_len]
            
            # Trading signals based on predicted returns
            signals = pred > 0  # Buy when predicted return is positive
            returns = actual  # actual is already in percentage form
            
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
            cursor.execute("DELETE FROM dnn_predictions WHERE ticker = ?", (ticker,))
            conn.commit()
            
            # Insert new predictions
            predictions_df.to_sql(
                'dnn_predictions',
                conn,
                if_exists='append',
                index=False
            )
        
        # Update performance metrics
        metrics = self.evaluate(conn, ticker)
        
        # Delete existing metrics
        cursor = conn.cursor()
        cursor.execute("DELETE FROM model_performance WHERE ticker = ? AND model = 'dnn'", (ticker,))
        conn.commit()
        
        # Store metrics
        metrics_df = pd.DataFrame([{
            'date': datetime.now().strftime('%Y-%m-%d'),
            'ticker': ticker,
            'model': 'dnn',
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
        }])
        
        metrics_df.to_sql(
            'model_performance',
            conn,
            if_exists='append',
            index=False
        )
        
        return predictions_df, metrics['test']

if __name__ == "__main__":
    # Get the absolute path to the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construct database path relative to project root
    db_path = os.path.join(project_root, 'data', 'market_data.db')
    
    # Initialize database if needed
    if not os.path.exists(db_path):
        from scripts.init_db import create_database
        create_database()
    
    conn = sqlite3.connect(db_path)
    model = DNNPredictor()
    predictions, metrics = model.update_predictions(conn, 'QQQ')
    print("Predictions:", predictions)
    print("Metrics:", metrics)
    conn.close()
