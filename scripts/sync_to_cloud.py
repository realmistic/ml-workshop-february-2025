import os
import sys
import sqlite3
import pandas as pd
import time
import socket

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connection import get_db_connection

# Increase socket timeout
socket.setdefaulttimeout(60)  # 60 seconds timeout

def sync_to_cloud(max_retries=3, batch_size=500, force_init=True):
    """Sync local database to SQLite Cloud."""
    if "SQLITECLOUD_URL" not in os.environ:
        print("Error: SQLITECLOUD_URL environment variable not set")
        return False
    
    print(f"Syncing to SQLite Cloud database with URL: {os.environ['SQLITECLOUD_URL'][:20]}...")
    
    # Connect to local database
    try:
        local_conn = sqlite3.connect('data/market_data.db')
        print("Successfully connected to local database")
    except Exception as e:
        print(f"Error connecting to local database: {str(e)}")
        return False
    
    # Connect to cloud database
    try:
        cloud_conn = get_db_connection(use_cloud=True)
        print("Successfully connected to SQLite Cloud")
    except Exception as e:
        print(f"Error connecting to SQLite Cloud: {str(e)}")
        return False
    
    # Initialize cloud database if needed
    if force_init:
        print("Initializing cloud database tables...")
        from init_cloud_db import init_cloud_db
        init_cloud_db(force=True)
    
    # Disable foreign key constraints temporarily
    try:
        cursor = cloud_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        print("Foreign key constraints disabled for sync")
    except Exception as e:
        print(f"Warning: Could not disable foreign key constraints: {str(e)}")
    
    # Tables to sync
    tables = [
        'raw_market_data', 
        'arima_features', 
        'prophet_features', 
        'dnn_features',
        'arima_predictions', 
        'prophet_predictions', 
        'dnn_predictions',
        'model_performance'
    ]
    
    for table in tables:
        try:
            print(f"Syncing table: {table}")
            
            # Force a full sync by getting all data from the local database
            print(f"Forcing full sync for table: {table}")
            query = f"SELECT * FROM {table}"
            local_data = pd.read_sql_query(query, local_conn)
            
            if local_data.empty:
                print(f"No data to sync for table: {table}")
                continue
            
            print(f"Syncing {len(local_data)} rows for table: {table}")
            
            # Insert data into cloud database using a more efficient approach with retries
            for retry in range(max_retries):
                try:
                    # Use the SQLite Cloud connection directly
                    cursor = cloud_conn.cursor()
                    
                    # Begin a transaction explicitly
                    cursor.execute("BEGIN TRANSACTION")
                    
                    # First, delete all existing data in the table
                    print(f"Deleting existing data from {table}...")
                    cursor.execute(f"DELETE FROM {table}")
                    
                    # Get column names
                    columns = local_data.columns.tolist()
                    column_str = ', '.join(columns)
                    
                    # Generate placeholders for the SQL query
                    placeholders = ', '.join(['?' for _ in columns])
                    
                    # Convert DataFrame to list of tuples for executemany
                    data_tuples = [tuple(x) for x in local_data.to_numpy()]
                    
                    # Process in batches to avoid timeouts
                    print(f"Processing {len(data_tuples)} rows in batches of {batch_size}...")
                    
                    # Direct insert with REPLACE to handle duplicates
                    insert_sql = f"INSERT OR REPLACE INTO {table} ({column_str}) VALUES ({placeholders})"
                    
                    # Process in batches
                    for i in range(0, len(data_tuples), batch_size):
                        batch = data_tuples[i:i+batch_size]
                        print(f"Inserting batch {i//batch_size + 1}/{(len(data_tuples) + batch_size - 1)//batch_size}...")
                        cursor.executemany(insert_sql, batch)
                    
                    # Commit changes
                    cloud_conn.commit()
                    print(f"Successfully synced {len(data_tuples)} rows to {table}")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"Error on attempt {retry+1}/{max_retries}: {str(e)}")
                    
                    try:
                        # Try to rollback the transaction
                        cloud_conn.rollback()
                    except:
                        pass
                    
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry  # Exponential backoff
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to sync table {table} after {max_retries} attempts")
                        import traceback
                        traceback.print_exc()
            
        except Exception as e:
            print(f"Error syncing table {table}: {str(e)}")
    
    # Re-enable foreign key constraints
    try:
        cursor = cloud_conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        print("Foreign key constraints re-enabled")
    except Exception as e:
        print(f"Warning: Could not re-enable foreign key constraints: {str(e)}")
    
    local_conn.close()
    cloud_conn.close()
    print("Database synced to cloud successfully!")
    return True

if __name__ == "__main__":
    if not sync_to_cloud():
        sys.exit(1)
