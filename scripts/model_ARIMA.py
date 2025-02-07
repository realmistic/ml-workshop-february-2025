import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import itertools
import pickle
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

def download_stock_data(ticker_symbol):
    """
    Download stock data with error handling and validation
    """
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

def transform_stock_data(df, ticker, h=5):
    """
    Transform stock data by adding various technical indicators and features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data DataFrame
    ticker : str
        Stock ticker symbol
    h : int
        Number of periods to predict (default: 5)
    """
    if df is None:
        return None
    
    # Ensure index has frequency information and handle missing values
    df = df.asfreq('B').fillna(method='ffill')  # Forward fill missing values
    
    # Calculate returns instead of using raw prices
    df['Returns'] = df['Close'].pct_change()
    
    # Add Date and Ticker columns
    df['Date'] = df.index
    df['Ticker'] = ticker
    
    # Split data into train/validation/test sets (60/20/20)
    total_days = len(df)
    train_end = int(total_days * 0.6)
    val_end = int(total_days * 0.8)
    
    df['split'] = 'train'
    df.iloc[train_end:val_end, df.columns.get_loc('split')] = 'validation'
    df.iloc[val_end:, df.columns.get_loc('split')] = 'test'
    
    print(f"Data split sizes:")
    print(df['split'].value_counts())
    
    # Generate features for historical prices
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday
    
    # Calculate historical returns for different periods
    for i in [1, 3, 7, 30, 90, 365]:
        df[f'growth_{i}d'] = df['Close'] / df['Close'].shift(i)
    
    # Calculate future growth (h days ahead)
    df[f'future_growth_{h}d'] = df['Close'].shift(-h) / df['Close']
    
    # Calculate 30-day rolling volatility (annualized)
    df['volatility'] = df['Close'].rolling(30).std() * np.sqrt(252)
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate YoY growth
    df['YoY_growth'] = (df['Close'] / df['Close'].shift(252)) - 1
    
    return df

def plot_stock_analysis(df, ticker, predictions=None):
    """
    Create and display various plots for stock analysis
    """
    if df is None:
        return
    
    # Create price chart with moving averages and data splits
    fig = go.Figure()
    
    # Plot data for each split separately
    colors = {'train': 'blue', 'validation': 'orange', 'test': 'red'}
    for split in ['train', 'validation', 'test']:
        mask = df['split'] == split
        fig.add_trace(go.Scatter(
            x=df[mask].index, 
            y=df[mask]['Close'], 
            name=f'Close ({split})', 
            line=dict(color=colors[split])
        ))
    
    # Add moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                            line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', 
                            line=dict(color='black', dash='dash')))
    
    fig.update_layout(
        title=f'{ticker} Price with Moving Averages and Data Splits',
        yaxis_title='Price',
        template='presentation'
    )
    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions,
            name='ARIMA Predictions',
            line=dict(color='green', dash='dot')
        ))
    
    fig.show()
    
    # Create volume chart
    fig_volume = px.bar(
        df,
        x=df.index,
        y='Volume',
        title=f'{ticker} Trading Volume'
    )
    fig_volume.update_layout(template='presentation')
    fig_volume.show()
    
    # Create YoY growth chart
    fig_yoy = px.bar(
        df,
        x=df.index,
        y='YoY_growth',
        title=f'{ticker} Year-over-Year Growth',
        color='YoY_growth',
        color_continuous_scale=['red', 'green']
    )
    fig_yoy.update_layout(
        yaxis_title='YoY Growth %',
        template='presentation'
    )
    fig_yoy.show()

def evaluate_arima_model(model, df, exog=None, split='validation', max_horizon=5):
    """
    Evaluate ARIMA model performance on validation/test set for multiple forecast horizons
    
    Parameters:
    -----------
    model : ARIMA model
        Trained ARIMA model
    df : pandas.DataFrame
        Full dataset with splits
    exog : pandas.DataFrame
        Exogenous variables
    split : str
        'validation' or 'test'
    max_horizon : int
        Maximum number of periods ahead to predict
    """
    split_mask = df['split'] == split
    split_data = df[split_mask]
    
    # Initialize dictionaries to store predictions and metrics
    predictions = {h: [] for h in range(1, max_horizon + 1)}
    actuals = {h: [] for h in range(1, max_horizon + 1)}
    
    # For each day in the split set (except last max_horizon days)
    for i in range(len(split_data) - max_horizon):
        # Get current exog data if provided
        current_exog = None if exog is None else exog[split_mask].iloc[i:i+max_horizon]
        
        # Make predictions for different horizons
        forecast = model.forecast(steps=max_horizon, exog=current_exog)
        
        # Store predictions and actuals for each horizon
        for h in range(1, max_horizon + 1):
            if i + h < len(split_data):
                predictions[h].append(forecast[h-1])
                actuals[h].append(split_data['Returns'].iloc[i + h])
    
    # Calculate metrics for each horizon
    metrics = {}
    for h in range(1, max_horizon + 1):
        if len(predictions[h]) > 0:
            mse = np.mean((np.array(actuals[h]) - np.array(predictions[h])) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((np.array(actuals[h]) - np.array(predictions[h])) / np.array(actuals[h]))) * 100
            metrics[h] = {'rmse': rmse, 'mape': mape}
    
    # Print results
    print(f"\nModel performance on {split} set by forecast horizon:")
    for h in range(1, max_horizon + 1):
        if h in metrics:
            print(f"Horizon {h}:")
            print(f"  RMSE: {metrics[h]['rmse']:.4f}")
            print(f"  MAPE: {metrics[h]['mape']:.2f}%")
    
    return metrics

def prepare_exog_variables(df, reference_columns=None):
    """
    Prepare exogenous variables for ARIMA model
    """
    # Create dummy variables for month and weekday
    month_dummies = pd.get_dummies(df['Month'], prefix='month')
    weekday_dummies = pd.get_dummies(df['Weekday'], prefix='weekday')
    
    # Select technical indicators
    tech_indicators = df[['growth_1d', 'growth_3d', 'growth_7d',
                         'volatility', 'MA20', 'MA50']]
    
    # Fill any missing values
    tech_indicators = tech_indicators.ffill().bfill()
    
    # Combine all exogenous variables
    exog = pd.concat([month_dummies, weekday_dummies, tech_indicators], axis=1)
    
    # If reference columns provided, ensure same columns in same order
    if reference_columns is not None:
        missing_cols = set(reference_columns) - set(exog.columns)
        if missing_cols:
            for col in missing_cols:
                exog[col] = 0
        exog = exog[reference_columns]
    
    # Ensure all columns are float type
    exog = exog.astype(float)
    
    return exog

def find_best_arima_params(train_data, val_data, train_exog, val_exog, max_p=2, max_d=1, max_q=2):
    """
    Find the best ARIMA parameters using grid search and validation set RMSE
    """
    # Suppress convergence warnings
    warnings.filterwarnings('ignore', 'Maximum Likelihood optimization failed to converge')
    warnings.filterwarnings('ignore', 'invalid value encountered')
    
    best_rmse = float('inf')
    best_params = None
    
    # Create all possible combinations of parameters
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)
    combinations = list(itertools.product(p, d, q))
    
    print(f"Testing {len(combinations)} ARIMA parameter combinations...")
    
    for params in combinations:
        try:
            # Convert data to numpy arrays
            train_data_np = np.asarray(train_data, dtype=float)
            train_exog_np = np.asarray(train_exog, dtype=float)
            val_exog_np = np.asarray(val_exog, dtype=float)
            
            # Train model
            model = ARIMA(train_data_np, order=params, exog=train_exog_np)
            model_fit = model.fit()
            
            # Evaluate on validation set
            predictions = model_fit.forecast(steps=len(val_data), exog=val_exog_np)
            mse = np.mean((val_data - predictions) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"ARIMA{params} - Validation RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
            
        except Exception as e:
            print(f"Skipping ARIMA{params} - {str(e)}")
            continue
    
    if best_params is not None:
        print(f"\nBest parameters (p,d,q) = {best_params} with validation RMSE = {best_rmse:.4f}")
    
    return best_params

def train_arima_model(df, ticker, max_horizon=5):
    """
    Train ARIMA model on returns and save it to file
    """
    try:
        # Prepare data splits
        train_mask = df['split'] == 'train'
        val_mask = df['split'] == 'validation'
        
        # Prepare returns data (exclude first row since it has NaN return)
        train_returns = df[train_mask]['Returns'].iloc[1:]
        val_returns = df[val_mask]['Returns']
        
        # Prepare exogenous variables for training
        exog = prepare_exog_variables(df)
        train_exog = exog[train_mask].iloc[1:]  # Align with returns
        val_exog = exog[val_mask]
        
        # Save the column names for future predictions
        exog_columns = exog.columns.tolist()
        
        print("Finding best ARIMA parameters using validation RMSE...")
        best_params = find_best_arima_params(train_returns, val_returns, train_exog, val_exog)
        
        if best_params is None:
            print("Could not find suitable ARIMA parameters")
            return None
            
        print(f"Best ARIMA parameters: {best_params}")
        
        print("Training final model...")
        # Convert data to numpy arrays
        train_returns_np = np.asarray(train_returns, dtype=float)
        train_exog_np = np.asarray(train_exog, dtype=float)
        
        model = ARIMA(train_returns_np, order=best_params, exog=train_exog_np)
        model_fit = model.fit()
        
        # Evaluate model on validation set with multiple horizons
        evaluate_arima_model(model_fit, df[train_mask | val_mask], val_exog, 'validation', max_horizon)
    
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model
        model_path = f'models/arima_{ticker}.pkl'
        with open(model_path, 'wb') as f:
            # Save both model and exog columns
            pickle.dump({'model': model_fit, 'exog_columns': exog_columns}, f)
        print(f"Model saved to {model_path}")
        
        return model_fit
        
    except Exception as e:
        print(f"Error training ARIMA model: {str(e)}")
        return None

def load_arima_model(ticker):
    """
    Load trained ARIMA model and exog columns from file
    """
    model_path = f'models/arima_{ticker}.pkl'
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            exog_columns = data['exog_columns']
        print(f"Loaded model from {model_path}")
        return model, exog_columns
    except FileNotFoundError:
        print(f"No trained model found at {model_path}")
        return None

def make_predictions(model, df, h, exog_columns=None):
    """
    Make predictions using the trained ARIMA model and convert returns to prices
    
    Parameters:
    -----------
    model : ARIMA model
        Trained ARIMA model
    df : pandas.DataFrame
        Stock data DataFrame
    h : int
        Number of periods to predict
    """
    try:
        if model is None:
            return None
    
        # Prepare future exogenous variables
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=h, freq='B')
        future_df = pd.DataFrame(index=future_dates)
        future_df['Month'] = future_df.index.month
        future_df['Weekday'] = future_df.index.weekday
        
        # Use last values for technical indicators
        for col in ['growth_1d', 'growth_3d', 'growth_7d', 'growth_30d', 'growth_90d', 'growth_365d',
                   'volatility', 'MA20', 'MA50', 'YoY_growth']:
            future_df[col] = df[col].iloc[-1]
        
        # Prepare exogenous variables for prediction using same columns as training
        future_exog = prepare_exog_variables(future_df, reference_columns=exog_columns)
        
        # Convert exogenous variables to numpy array
        future_exog_np = np.asarray(future_exog, dtype=float)
        
        # Predict returns using exogenous variables
        return_predictions = model.forecast(steps=h, exog=future_exog_np)
        
        # Convert returns predictions to price predictions
        last_price = df['Close'].iloc[-1]
        price_predictions = [last_price]
        for ret in return_predictions:
            price_predictions.append(price_predictions[-1] * (1 + ret))
        
        # Create price predictions series
        price_predictions = pd.Series(price_predictions[1:])  # Remove initial price
        
        # Create date index for predictions
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='B')
        price_predictions.index = pred_dates
        
        print(f"\nPredictions for the next {h} days:")
        for date, pred in zip(pred_dates, price_predictions):
            print(f"{date.date()}: {pred:.2f}")
        
        return price_predictions
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

def main(ticker="QQQ", h=5, train_model=False, max_horizon=5, force_retrain=True):
    """
    Main function to run the stock analysis
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (default: "QQQ")
    h : int
        Number of periods to predict (default: 5)
    train_model : bool
        Whether to train a new ARIMA model (default: False)
    """
    print(f"Analyzing {ticker} with up to {max_horizon} periods forecast horizon")
    df = download_stock_data(ticker)
    
    if df is None:
        print(f"Could not proceed with analysis for {ticker} due to data download failure")
        return None, None, None
        
    df = transform_stock_data(df, ticker, max_horizon)
    
    if train_model or force_retrain:
        model = train_arima_model(df, ticker, max_horizon)
        if model is None:
            print("Model training failed. Cannot proceed with evaluation.")
            return None, None, None
        _, exog_columns = load_arima_model(ticker)  # Get columns after training
    else:
        model, exog_columns = load_arima_model(ticker)
        if model is None:
            print("No trained model found. Training new model...")
            model = train_arima_model(df, ticker, max_horizon)
            if model is None:
                print("Model training failed. Cannot proceed with evaluation.")
                return None, None, None
            _, exog_columns = load_arima_model(ticker)  # Get columns after training
    
    # Make predictions
    predictions = make_predictions(model, df, h, exog_columns)
    
    # Evaluate on test set
    test_mask = df['split'] == 'test'
    test_exog = prepare_exog_variables(df[test_mask], reference_columns=exog_columns)
    print("\nEvaluating on test set:")
    evaluate_arima_model(model, df[test_mask], test_exog, 'test', max_horizon)
    
    # Plot analysis with predictions
    plot_stock_analysis(df, ticker, predictions)
    return df, model, predictions

if __name__ == "__main__":
    import sys
    
    # Get command line arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    train_model = "--train" in sys.argv
    force_retrain = "--force-retrain" in sys.argv
    
    main(ticker=ticker, h=h, train_model=train_model, max_horizon=max_horizon, force_retrain=force_retrain)
