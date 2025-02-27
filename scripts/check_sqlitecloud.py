#!/usr/bin/env python3
"""
Script to check SQLite Cloud installation and connection URL format.
"""

import os
import sys
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

def check_sqlitecloud_installation():
    """Check if SQLite Cloud package is installed correctly."""
    print("\n=== Checking SQLite Cloud Installation ===")
    
    try:
        import sqlitecloud
        try:
            version = getattr(sqlitecloud, '__version__', 'unknown')
            print(f"✅ SQLite Cloud SDK is installed (version: {version})")
        except:
            print(f"✅ SQLite Cloud SDK is installed (version information not available)")
        
        print(f"   Module location: {sqlitecloud.__file__}")
        return True
    except ImportError as e:
        print(f"❌ SQLite Cloud SDK is not installed or cannot be imported: {str(e)}")
        print("   Try reinstalling with: pip install sqlitecloud==0.0.83")
        return False

def check_connection_url():
    """Check if SQLite Cloud connection URL is properly formatted."""
    print("\n=== Checking SQLite Cloud Connection URL ===")
    
    url = os.environ.get("SQLITECLOUD_URL")
    if not url:
        print("❌ SQLITECLOUD_URL environment variable is not set")
        return False
    
    # Basic format check
    if not url.startswith("sqlitecloud://"):
        print("❌ URL should start with 'sqlitecloud://'")
        return False
    
    # Parse URL components
    try:
        # Remove protocol
        url_without_protocol = url.replace("sqlitecloud://", "")
        
        # Check if there's authentication info
        if "@" in url_without_protocol:
            auth_part, server_part = url_without_protocol.split("@", 1)
            print(f"✅ Authentication information found")
        else:
            auth_part = ""
            server_part = url_without_protocol
        
        # Check server and port
        if ":" in server_part:
            server, port_and_path = server_part.split(":", 1)
            if "/" in port_and_path:
                port, path = port_and_path.split("/", 1)
                print(f"✅ Server: {server}")
                print(f"✅ Port: {port}")
                print(f"✅ Database path: {path.split('?')[0] if '?' in path else path}")
            else:
                print(f"✅ Server: {server}")
                print(f"✅ Port: {port_and_path}")
                print("❌ Missing database path")
                return False
        else:
            print(f"✅ Server: {server_part}")
            print("❌ Missing port specification")
            return False
        
        # Check for API key
        if "apikey=" in url:
            print("✅ API key found")
        else:
            print("⚠️ No API key found in URL")
        
        print("✅ URL format appears valid")
        return True
    
    except Exception as e:
        print(f"❌ Error parsing URL: {str(e)}")
        return False

def main():
    """Main function."""
    print("SQLite Cloud Configuration Check")
    print("===============================")
    
    # Check if USE_SQLITECLOUD is enabled
    use_cloud = os.environ.get("USE_SQLITECLOUD", "0").lower() in ("1", "true", "yes")
    if use_cloud:
        print("✅ USE_SQLITECLOUD is enabled")
    else:
        print("❌ USE_SQLITECLOUD is not enabled (set to '1' to enable)")
    
    # Check SQLite Cloud installation
    installation_ok = check_sqlitecloud_installation()
    
    # Check connection URL
    url_ok = check_connection_url()
    
    # Summary
    print("\n=== Summary ===")
    if installation_ok and url_ok and use_cloud:
        print("✅ SQLite Cloud configuration appears valid")
        print("   Try running: python scripts/test_cloud_connection.py")
    else:
        print("❌ SQLite Cloud configuration has issues that need to be fixed")

if __name__ == "__main__":
    main()
