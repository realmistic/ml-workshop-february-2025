import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connection import get_db_connection

def create_database():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Connect to database (local by default)
    conn = get_db_connection(use_cloud=False)
    cursor = conn.cursor()

    # Create raw market data table with ticker column
    cursor.execute('''
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
    ''')

    # Create processed features tables for each model
    cursor.execute('''
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
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prophet_features (
        date TEXT,
        ticker TEXT,
        y REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, ticker),
        FOREIGN KEY (date, ticker) REFERENCES raw_market_data (date, ticker)
    )
    ''')

    cursor.execute('''
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
    ''')

    # Create predictions tables for each model
    for model in ['arima', 'prophet', 'dnn']:
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {model}_predictions (
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
        ''')

    # Create performance metrics table
    cursor.execute('''
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
    ''')

    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database initialized successfully!")
