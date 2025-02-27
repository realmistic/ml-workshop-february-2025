import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connection import get_db_connection

def test_connection(use_cloud=True):
    """Test database connection."""
    try:
        # Set environment variable for testing if not already set
        if use_cloud and "SQLITECLOUD_URL" not in os.environ:
            print("Error: SQLITECLOUD_URL environment variable not set")
            print("Please set it with: export SQLITECLOUD_URL='your-connection-url'")
            return False
        
        # Force specific connection type
        conn = get_db_connection(use_cloud=use_cloud)
        
        # Try a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT sqlite_version()")
        version = cursor.fetchone()
        
        # Close connection
        conn.close()
        
        # Print result
        if use_cloud:
            print(f"✅ Successfully connected to SQLite Cloud (SQLite version: {version[0]})")
        else:
            print(f"✅ Successfully connected to local SQLite database (SQLite version: {version[0]})")
        
        return True
    
    except Exception as e:
        if use_cloud:
            print(f"❌ Failed to connect to SQLite Cloud: {str(e)}")
        else:
            print(f"❌ Failed to connect to local SQLite database: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test database connection')
    parser.add_argument('--local', action='store_true', help='Test local SQLite connection')
    parser.add_argument('--env-file', type=str, help='Path to .env file (default: .env in project root)')
    args = parser.parse_args()
    
    # Check for .env file if specified
    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.exists():
            print(f"Error: Specified .env file not found: {args.env_file}")
            sys.exit(1)
        print(f"Using environment variables from: {args.env_file}")
        load_dotenv(dotenv_path=env_path)  # Load the specified .env file
    
    if args.local:
        test_connection(use_cloud=False)
    else:
        # Set USE_SQLITECLOUD environment variable for testing if not using .env file
        if "USE_SQLITECLOUD" not in os.environ:
            os.environ["USE_SQLITECLOUD"] = "1"
        test_connection(use_cloud=True)
