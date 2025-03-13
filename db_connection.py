import os
import sqlite3
import pandas as pd
import sqlitecloud
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
        load_dotenv(dotenv_path=env_path, override=True)
        break

# Manually read the .env file to ensure we get the correct values
def read_env_file():
    """Read environment variables directly from .env file."""
    env_vars = {}
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
            break
    return env_vars

def get_db_connection(use_cloud=None, pandas_friendly=True, direct_connection=False):
    """
    Get database connection based on configuration.
    
    Args:
        use_cloud: Override environment setting (True=cloud, False=local, None=use environment)
        pandas_friendly: If True, return a connection object that works well with pandas
        direct_connection: If True, always return a direct SQLite connection (not SQLAlchemy engine)
    
    Returns:
        Database connection object
    """
    # Determine whether to use cloud
    if use_cloud is None:
        # Check environment variable, default to local if not set
        use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
        print(f"USE_SQLITECLOUD environment variable: {os.environ.get('USE_SQLITECLOUD', 'not set')}")
        print(f"Using cloud: {use_cloud}")
    
    if use_cloud:
        try:
            # Import sqlitecloud only when needed
            import sqlitecloud
            
            # Print version information for debugging
            try:
                version = getattr(sqlitecloud, '__version__', 'unknown')
                print(f"Using SQLite Cloud SDK version: {version}")
            except:
                print("Using SQLite Cloud SDK (version information not available)")
            
            # Get URL directly from .env file
            env_vars = read_env_file()
            url = env_vars.get('SQLITECLOUD_URL', os.environ.get('SQLITECLOUD_URL', ''))
            
            if not url:
                print("SQLITECLOUD_URL not found in .env file or environment variables")
                print("Falling back to local database.")
                return sqlite3.connect('data/market_data.db')
                
            print(f"Using SQLite Cloud URL from .env file")
            try:
                if '@' in url:
                    server_part = url.split('@')[1].split('?')[0]
                else:
                    server_part = url.replace('sqlitecloud://', '').split('?')[0]
                print(f"Connecting to: {server_part}")
            except Exception as e:
                print(f"Could not parse URL (but will try to connect anyway): {str(e)}")
            
            # Connect to SQLite Cloud
            
            # Create a custom connection function that uses an unverified SSL context
            import socket
            import ssl
            
            # Monkey patch the socket.create_connection function
            original_create_connection = socket.create_connection
            
            def patched_create_connection(*args, **kwargs):
                print("Using patched socket.create_connection")
                return original_create_connection(*args, **kwargs)
            
            # Monkey patch the ssl.create_default_context function
            original_create_default_context = ssl.create_default_context
            
            def patched_create_default_context(*args, **kwargs):
                print("Using patched ssl.create_default_context")
                context = ssl._create_unverified_context(*args, **kwargs)
                return context
            
            # Apply the patches
            socket.create_connection = patched_create_connection
            ssl.create_default_context = patched_create_default_context
            
            try:
                print("Connecting with SSL verification disabled...")
                cloud_conn = sqlitecloud.connect(url)
                print("SQLite Cloud connection successful")
            finally:
                # Restore the original functions
                socket.create_connection = original_create_connection
                ssl.create_default_context = original_create_default_context
            
            print("SQLite Cloud connection successful")
            
            # Return the cloud connection directly
            # This is the most reliable approach for SQLite Cloud
            return cloud_conn
            
        except ImportError as e:
            print(f"SQLite Cloud SDK not installed or import error: {str(e)}")
            print("Falling back to local database.")
            return sqlite3.connect('data/market_data.db')
        except Exception as e:
            print(f"Error connecting to SQLite Cloud: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            
            # Try to get more information about the error
            import traceback
            traceback.print_exc()
            
            print("Falling back to local database.")
            return sqlite3.connect('data/market_data.db')
    else:
        # Use local SQLite
        if direct_connection:
            # Always return a direct SQLite connection
            return sqlite3.connect('data/market_data.db')
        elif pandas_friendly:
            try:
                # Import SQLAlchemy
                from sqlalchemy import create_engine
                
                # Create a SQLAlchemy engine
                engine = create_engine(f"sqlite:///data/market_data.db")
                
                # Return the engine
                return engine
            except ImportError:
                print("SQLAlchemy not installed, falling back to direct connection")
                return sqlite3.connect('data/market_data.db')
        else:
            return sqlite3.connect('data/market_data.db')

def read_data(query, params=None, conn=None, close_conn=False):
    """
    Read data from the database using a SQL query.
    This function is specifically designed for SQLite Cloud connections.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query (optional)
        conn: Database connection (optional, will create one if not provided)
        close_conn: Whether to close the connection after execution (default: False)
    
    Returns:
        pandas DataFrame with the query results
    """
    # Get connection if not provided
    if conn is None:
        conn = get_db_connection(use_cloud=True)
        close_conn = True
    
    try:
        # Execute query and convert to pandas DataFrame
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        # Get column names from cursor description
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=columns)
            return df
        else:
            # Query didn't return any data (e.g., INSERT, UPDATE, DELETE)
            return pd.DataFrame()
    finally:
        if close_conn:
            conn.close()
