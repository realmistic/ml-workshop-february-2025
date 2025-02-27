import os
import sqlite3
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

def get_db_connection(use_cloud=None, pandas_friendly=True):
    """
    Get database connection based on configuration.
    
    Args:
        use_cloud: Override environment setting (True=cloud, False=local, None=use environment)
        pandas_friendly: If True, return a connection object that works well with pandas
    
    Returns:
        Database connection object
    """
    # Determine whether to use cloud
    if use_cloud is None:
        # Check environment variable, default to local if not set
        use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
        print(f"USE_SQLITECLOUD environment variable: {os.environ.get('USE_SQLITECLOUD', 'not set')}")
        print(f"Using cloud: {use_cloud}")
    
    if use_cloud and "SQLITECLOUD_URL" in os.environ:
        try:
            # Import sqlitecloud only when needed
            import sqlitecloud
            
            # Print version information for debugging
            try:
                version = getattr(sqlitecloud, '__version__', 'unknown')
                print(f"Using SQLite Cloud SDK version: {version}")
            except:
                print("Using SQLite Cloud SDK (version information not available)")
            # Safely extract server info from URL
            url = os.environ['SQLITECLOUD_URL']
            try:
                if '@' in url:
                    server_part = url.split('@')[1].split('?')[0]
                else:
                    server_part = url.replace('sqlitecloud://', '').split('?')[0]
                print(f"Connecting to: {server_part}")
            except Exception as e:
                print(f"Could not parse URL (but will try to connect anyway): {str(e)}")
            
            # Connect to SQLite Cloud
            url = os.environ["SQLITECLOUD_URL"]
            
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
            
            # If pandas_friendly is True, create a local SQLite database in memory
            # and use it as a proxy for the cloud database
            if pandas_friendly:
                # Create a wrapper class that provides a pandas-friendly interface
                class PandasFriendlyConnection:
                    def __init__(self, cloud_conn):
                        self.cloud_conn = cloud_conn
                        self.is_cloud = True

                    def cursor(self):
                        return self.cloud_conn.cursor()

                    def commit(self):
                        return self.cloud_conn.commit()

                    def rollback(self):
                        return self.cloud_conn.rollback()

                    def close(self):
                        return self.cloud_conn.close()

                    def execute(self, sql, params=None):
                        cursor = self.cloud_conn.cursor()
                        if params:
                            return cursor.execute(sql, params)
                        else:
                            return cursor.execute(sql)
                
                return PandasFriendlyConnection(cloud_conn)
            else:
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
        return sqlite3.connect('data/market_data.db')
