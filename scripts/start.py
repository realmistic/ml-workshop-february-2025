import yfinance as yf
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
# from scripts.model_ARIMA import download_stock_data
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pickle

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

def plot_analysis(df, ticker):
    """
    Create and display comprehensive stock analysis plots
    """
    # Price chart with moving averages
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='orange')))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='red')))
    fig_price.update_layout(
        title=f'{ticker} Price with Moving Averages',
        yaxis_title='Price',
        template='presentation'
    )
    fig_price.show()

    # Volume chart
    fig_volume = px.bar(
        df,
        x=df.index,
        y='Volume',
        title=f'{ticker} Trading Volume'
    )
    fig_volume.update_layout(template='presentation')
    fig_volume.show()

    # Calculate confidence intervals using return volatility
    df_2024 = df[df.index >= '2024-01-01']
    returns_std = df_2024['growth_1d'].apply(lambda x: x - 1).std()
    df_2024['ci_upper'] = df_2024['Close'] * (1 + 2 * returns_std)  # 95% CI
    df_2024['ci_lower'] = df_2024['Close'] * (1 - 2 * returns_std)  # 95% CI
    
    # Price chart with confidence intervals
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_2024.index, y=df_2024['Close'], name='Close', line=dict(color='blue')))
    fig_price.add_trace(go.Scatter(x=df_2024.index, y=df_2024['ci_upper'], name='Upper CI (95%)', line=dict(color='lightgray', dash='dash')))
    fig_price.add_trace(go.Scatter(x=df_2024.index, y=df_2024['ci_lower'], name='Lower CI (95%)', line=dict(color='lightgray', dash='dash')))
    fig_price.update_layout(
        title=f'{ticker} Price with 95% Confidence Intervals (2024)',
        yaxis_title='Price',
        template='presentation'
    )
    fig_price.show()
    
    # Calculate Bollinger Bands (20-day SMA ± 2 standard deviations)
    df_2024['BB_middle'] = df_2024['Close'].rolling(window=20).mean()
    rolling_std = df_2024['Close'].rolling(window=20).std()
    df_2024['BB_upper'] = df_2024['BB_middle'] + (rolling_std * 2)
    df_2024['BB_lower'] = df_2024['BB_middle'] - (rolling_std * 2)
    
    # Price chart with Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df_2024.index, y=df_2024['Close'], name='Close', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=df_2024.index, y=df_2024['BB_upper'], name='Upper BB', line=dict(color='gray', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=df_2024.index, y=df_2024['BB_middle'], name='20-day MA', line=dict(color='orange')))
    fig_bb.add_trace(go.Scatter(x=df_2024.index, y=df_2024['BB_lower'], name='Lower BB', line=dict(color='gray', dash='dash')))
    fig_bb.update_layout(
        title=f'{ticker} Price with Bollinger Bands (2024)',
        yaxis_title='Price',
        template='presentation'
    )
    fig_bb.show()

    # Growth distributions
    df_melted = df[['growth_1d', 'growth_5d', 'growth_30d']].melt(var_name="Period", value_name="Growth")
    fig_dist = px.histogram(
        df_melted, 
        x="Growth", 
        color="Period", 
        barmode="overlay", 
        nbins=200, 
        histnorm="percent"
    )
    fig_dist.update_layout(
        title="Growth Distribution Comparison",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(title="Percentage of Total", tickformat=".1f%"),
        template='presentation'
    )
    fig_dist.show()
    
    # Plot ACF and PACF
    plt.figure(figsize=(12, 8))
    
    # ACF plot
    plt.subplot(211)
    returns = df['growth_1d'].dropna().apply(lambda x: x - 1)  # Convert to actual returns
    plot_acf(returns, lags=40, ax=plt.gca(), title='Autocorrelation Function of Daily Returns')
    
    # PACF plot
    plt.subplot(212)
    plot_pacf(returns, lags=40, ax=plt.gca(), title='Partial Autocorrelation Function of Daily Returns')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze and print ACF/PACF interpretation
    analyze_autocorrelation(returns)

def analyze_autocorrelation(series):
    """
    Analyze ACF and PACF to suggest ARIMA parameters
    """
    # Calculate ACF values
    acf_values = acf(series, nlags=40)
    
    # Find significant lags (using 95% confidence interval)
    confidence_interval = 1.96/np.sqrt(len(series))
    significant_lags_acf = [i for i, v in enumerate(acf_values) if abs(v) > confidence_interval and i > 0]
    
    # Analyze patterns
    print("\nARIMA Parameter Analysis:")
    print("------------------------")
    
    # Check for stationarity
    if abs(acf_values[1]) > 0.9:
        print("Series shows high persistence - differencing (d=1) may be needed")
        d_recommendation = 1
    else:
        print("Series appears stationary - differencing may not be needed")
        d_recommendation = 0
    
    # Analyze ACF pattern
    if len(significant_lags_acf) == 0:
        print("No significant autocorrelation found - suggesting MA(0)")
        q_recommendation = 0
    else:
        last_significant = max(significant_lags_acf)
        if last_significant > 2:
            print(f"Significant autocorrelation up to lag {last_significant}")
            q_recommendation = min(last_significant, 2)  # Cap at 2 for practicality
        else:
            q_recommendation = last_significant
    
    # Analyze PACF pattern for AR term
    pacf_values = pacf(series, nlags=40)
    significant_lags_pacf = [i for i, v in enumerate(pacf_values) if abs(v) > confidence_interval and i > 0]
    
    if len(significant_lags_pacf) == 0:
        print("No significant partial autocorrelation - suggesting AR(0)")
        p_recommendation = 0
    else:
        last_significant = max(significant_lags_pacf)
        if last_significant > 2:
            print(f"Significant partial autocorrelation up to lag {last_significant}")
            p_recommendation = min(last_significant, 2)  # Cap at 2 for practicality
        else:
            p_recommendation = last_significant
    
    # Create a DataFrame for parameters
    params_df = pd.DataFrame({
        'Parameter': ['p (AR term)', 'd (Differencing)', 'q (MA term)'],
        'Value': [p_recommendation, d_recommendation, q_recommendation],
        'Description': [
            'Number of autoregressive terms',
            'Number of differences needed for stationarity',
            'Number of moving average terms'
        ]
    })
    
    print("\nRecommended ARIMA Parameters:")
    print("------------------------")
    print(params_df.to_string(index=False))
    print("\nNote: These are statistical suggestions. Model performance should be validated through testing.")
    
    return p_recommendation, d_recommendation, q_recommendation

def prepare_exog_variables(df):
    """
    Prepare exogenous variables for ARIMA model
    """
    # Create dummy variables for calendar effects
    month_dummies = pd.get_dummies(df['Month'], prefix='month')
    weekday_dummies = pd.get_dummies(df['Weekday'], prefix='weekday')
    
    # Normalize volume (using rolling mean to avoid look-ahead bias)
    volume_ma = df['Volume'].rolling(window=20).mean()
    normalized_volume = df['Volume'] / volume_ma
    
    # Technical indicators (already calculated in transform_data)
    tech_indicators = df[['volatility', 'MA20', 'MA50']]
    
    # Lagged returns (excluding growth_1d as it's our target)
    lagged_returns = df[[f'growth_{i}d' for i in [3, 5, 30, 90, 365]]]
    
    # Combine all features
    exog = pd.concat([
        month_dummies,
        weekday_dummies,
        normalized_volume.rename('norm_volume'),
        tech_indicators,
        lagged_returns
    ], axis=1)
    
    # Fill any missing values with forward fill then backward fill
    exog = exog.ffill().bfill()
    
    # Ensure all data is float type
    for col in exog.columns:
        exog[col] = pd.to_numeric(exog[col], errors='coerce')
    
    # Convert to numpy array
    exog = exog.astype(float)
    
    return exog

def estimate_arima_model(df, ticker):
    """
    Estimate ARIMA model using recommended parameters and exogenous variables
    """
    # Prepare data splits
    train_val_mask = df['split'].isin(['train', 'validation'])
    train_val_data = df[train_val_mask]
    
    # Prepare returns data
    returns = train_val_data['growth_1d'].dropna() - 1  # Convert to actual returns
    returns = returns.sort_index()
    returns = pd.to_numeric(returns, errors='coerce')  # Ensure numeric type
    
    # Reindex with business day frequency, but only for dates we actually have
    min_date = returns.index.min()
    max_date = returns.index.max()
    bday_index = pd.date_range(start=min_date, end=max_date, freq='B')
    bday_index = bday_index[bday_index.isin(returns.index)]  # Keep only trading days
    returns = returns.reindex(bday_index)
    
    # Prepare exogenous variables
    exog = prepare_exog_variables(train_val_data)
    exog = exog.reindex(bday_index)  # Align with returns data using business day index
    
    try:
        # Define model orders to try
        orders = [
            (10, 0, 10),  # Higher order model
            (2, 0, 2)     # Original model for comparison
        ]
        
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        print("\nComparing different ARIMA specifications:")
        print("----------------------------------------")
        
        for order in orders:
            p, d, q = order
            print(f"\nTrying ARIMA{order}")
            
            # Train model with exogenous variables
            model = ARIMA(returns, order=order, exog=exog)
            model_fit = model.fit(method='innovations_mle')
            
            current_aic = model_fit.aic
            print(f"AIC: {current_aic:.2f}")
            
            if current_aic < best_aic:
                best_aic = current_aic
                best_model = model_fit
                best_order = order
        
        print(f"\nSelected ARIMA {best_order} based on AIC")
        print("\nModel Summary:")
        print("-------------")
        print(best_model.summary())
        
        # Prepare future exogenous variables for forecasting
        future_exog = prepare_exog_variables(df)
        future_exog = future_exog.iloc[-5:]  # Last 5 days as a proxy for future values
        
        # Make predictions for next 5 days
        forecast = best_model.forecast(steps=5, exog=future_exog)
        forecast_std = best_model.forecast(steps=5, exog=future_exog, method='std_forecast')
        
        print("\nPredictions for next 5 days (daily returns):")
        last_price = train_val_data['Close'].iloc[-1]
        cumulative_price = last_price
        for i, (pred, std) in enumerate(zip(forecast, forecast_std), 1):
            pred_return = pred + 1  # Convert back from returns to growth factor
            pred_price = cumulative_price * pred_return
            ci_lower = pred - 1.96 * std
            ci_upper = pred + 1.96 * std
            print(f"Day {i}:")
            print(f"  Return: {pred:.2%} (95% CI: [{ci_lower:.2%}, {ci_upper:.2%}])")
            print(f"  Price: ${pred_price:.2f}")
            cumulative_price = pred_price
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/arima_{ticker}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model,
                'order': best_order,
                'exog_columns': exog.columns.tolist()
            }, f)
        print(f"\nModel saved to {model_path}")
        
        return best_model, forecast
        
    except Exception as e:
        print(f"Error estimating ARIMA model: {str(e)}")
        return None, None

def analyze_growth_stats(df):
    """
    Analyze and display growth statistics
    """
    stats = df[['growth_1d', 'growth_5d', 'growth_30d']].describe().T
    print("\nGrowth Statistics:")
    print(stats)
    return stats

def transform_data(df, ticker):
    """
    Transform stock data by adding various indicators and split into train/validation/test sets
    """
    if df is None:
        return None
        
    # Add Date and Ticker columns
    df['Date'] = df.index
    df['Ticker'] = ticker
    
    # Generate features for historical prices
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Weekday'] = df.index.weekday
    
    # Calculate historical returns for different periods
    for i in [1, 3, 5, 30, 90, 365]:
        df[f'growth_{i}d'] = df['Close'] / df['Close'].shift(i)
    
    # Calculate future growth (3 days ahead)
    df['future_growth_3d'] = df['Close'].shift(-3) / df['Close']
    
    # Calculate 30-day rolling volatility (annualized)
    df['volatility'] = df['Close'].rolling(30).std() * np.sqrt(252)
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate YoY growth
    df['YoY_growth'] = (df['Close'] / df['Close'].shift(252)) - 1
    
    # Split data into train/validation/test sets (60/20/20)
    total_days = len(df)
    train_end = int(total_days * 0.6)
    val_end = int(total_days * 0.8)
    
    df['split'] = 'train'
    df.iloc[train_end:val_end, df.columns.get_loc('split')] = 'validation'
    df.iloc[val_end:, df.columns.get_loc('split')] = 'test'
    
    print(f"\nData split sizes:")
    print(df['split'].value_counts())
    
    return df

# Run default analysis
ticker = "QQQ"
df = download_stock_data(ticker)
df = transform_data(df, ticker)

df.head()

if df is not None:
    print("\nMost recent date:")
    print(df.Date.max())
    stats = analyze_growth_stats(df)
    plot_analysis(df, ticker)
    
    # Estimate model with recommended parameters
    model, forecast = estimate_arima_model(df, ticker)
