#!/usr/bin/env python3
"""
Script to perform pre-commit checks for SQLite Cloud integration.
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and check its exit code."""
    print(f"\n=== {description} ===")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n=== Checking Environment Variables ===")
    
    if "SQLITECLOUD_URL" not in os.environ:
        print("❌ SQLITECLOUD_URL environment variable not set")
        print("Please set it in your .env file or environment")
        return False
    
    if os.environ.get("USE_SQLITECLOUD", "0").lower() not in ("1", "true", "yes"):
        print("❌ USE_SQLITECLOUD environment variable not set to '1'")
        print("Please set it in your .env file or environment")
        return False
    
    print("✅ Environment variables are set correctly")
    return True

def check_files_exist():
    """Check if all required files exist."""
    print("\n=== Checking Required Files ===")
    
    required_files = [
        "db_connection.py",
        "scripts/check_sqlitecloud.py",
        "scripts/test_cloud_connection.py",
        "scripts/init_cloud_db.py",
        "scripts/sync_to_cloud.py",
        ".github/workflows/daily_update.yml",
        "scripts/setup_github_secrets.md",
        "scripts/setup_streamlit_cloud.md",
        ".env.sample"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} does not exist")
            all_exist = False
    
    return all_exist

def main():
    """Run all pre-commit checks."""
    print("SQLite Cloud Pre-Commit Checks")
    print("==============================")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found")
        print("Please create a .env file with your SQLite Cloud credentials")
        print("You can copy .env.sample and update it with your values")
        return False
    
    # Check environment variables
    if not check_environment_variables():
        return False
    
    # Check if required files exist
    if not check_files_exist():
        return False
    
    # Run SQLite Cloud checks
    checks = [
        ("python scripts/check_sqlitecloud.py", "Checking SQLite Cloud Configuration"),
        ("python scripts/test_cloud_connection.py", "Testing SQLite Cloud Connection"),
        ("python scripts/init_cloud_db.py", "Initializing SQLite Cloud Database"),
        ("python scripts/sync_to_cloud.py", "Syncing Data to SQLite Cloud")
    ]
    
    for command, description in checks:
        if not run_command(command, description):
            print(f"\n❌ {description} failed")
            print("Please fix the issues before committing")
            return False
        # Add a small delay between commands to avoid rate limiting
        time.sleep(1)
    
    # Final check: Run the Streamlit app with SQLite Cloud
    print("\n=== Final Check: Streamlit App ===")
    print("To test the Streamlit app with SQLite Cloud, run:")
    print("USE_SQLITECLOUD=1 streamlit run app/main.py")
    print("Verify that the app shows 'Using SQLite Cloud database' in the UI")
    
    print("\n=== All Checks Passed ===")
    print("You can now commit your changes to GitHub")
    return True

if __name__ == "__main__":
    if not main():
        sys.exit(1)
