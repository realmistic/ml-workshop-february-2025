# Setting Up Streamlit Cloud with SQLite Cloud

This guide explains how to deploy your Streamlit app to Streamlit Cloud while using SQLite Cloud for database storage.

## Prerequisites

1. You have created a SQLite Cloud account at [sqlitecloud.io](https://sqlitecloud.io)
2. You have created a database in your SQLite Cloud dashboard
3. You have your SQLite Cloud connection URL
4. You have a GitHub repository with your Streamlit app

## Steps to Set Up Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in
2. Click on "New app"
3. Select your GitHub repository, branch, and the path to your main app file (e.g., `app/main.py`)
4. Click on "Advanced settings"
5. Add the following secrets:
   - `SQLITECLOUD_URL`: Your SQLite Cloud connection URL in the format `sqlitecloud://username:password@hostname:port/database?apikey=your-api-key`
   - `USE_SQLITECLOUD`: Set to `1` to enable SQLite Cloud
6. Click "Deploy"

## Verifying the Setup

1. Wait for the app to deploy
2. Visit the app URL provided by Streamlit Cloud
3. Check that the app is displaying data from your SQLite Cloud database
4. Look for the "Using SQLite Cloud database" indicator in the app

## Troubleshooting

If you encounter issues with the Streamlit Cloud deployment:

1. Check the app logs in the Streamlit Cloud dashboard
2. Verify that your SQLite Cloud connection URL is correct
3. Ensure that your SQLite Cloud account has the necessary permissions
4. Check that the `USE_SQLITECLOUD` secret is set to `1`
5. Verify that the `sqlitecloud` package is included in your `requirements.txt` file

## Security Considerations

- Never commit your SQLite Cloud connection URL directly to your repository
- Use Streamlit Cloud secrets to securely store sensitive information
- Consider using a dedicated API key with limited permissions for Streamlit Cloud
- Regularly rotate your API keys and update the Streamlit Cloud secrets accordingly

## Updating the Database

The database will be automatically updated by the GitHub Actions workflow. The Streamlit Cloud app will always display the latest data from your SQLite Cloud database.

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [SQLite Cloud Documentation](https://docs.sqlitecloud.io/)
