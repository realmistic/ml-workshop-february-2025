# Setting Up GitHub Secrets for SQLite Cloud

This guide explains how to set up GitHub Secrets to enable automatic database updates to SQLite Cloud using GitHub Actions.

## Prerequisites

1. You have created a SQLite Cloud account at [sqlitecloud.io](https://sqlitecloud.io)
2. You have created a database in your SQLite Cloud dashboard
3. You have your SQLite Cloud connection URL

## Steps to Set Up GitHub Secrets

1. Go to your GitHub repository page
2. Click on "Settings" in the top navigation bar
3. In the left sidebar, click on "Secrets and variables" and then "Actions"
4. Click on "New repository secret"
5. Add the following secret:
   - Name: `SQLITECLOUD_URL`
   - Value: Your SQLite Cloud connection URL in the format `sqlitecloud://username:password@hostname:port/database?apikey=your-api-key`
6. Click "Add secret"

## Verifying the Setup

1. Go to the "Actions" tab in your repository
2. Click on the "Daily Database Update" workflow
3. Click on "Run workflow" and select the branch to run on (usually `main`)
4. Wait for the workflow to complete
5. Check the logs to ensure that the database was successfully synced to SQLite Cloud

## Troubleshooting

If you encounter issues with the GitHub Actions workflow:

1. Check that your SQLite Cloud connection URL is correct
2. Ensure that your SQLite Cloud account has the necessary permissions
3. Verify that the GitHub Actions workflow has access to the secret
4. Check the workflow logs for any error messages

## Security Considerations

- Never commit your SQLite Cloud connection URL directly to your repository
- Use GitHub Secrets to securely store sensitive information
- Consider using a dedicated API key with limited permissions for GitHub Actions
- Regularly rotate your API keys and update the GitHub Secret accordingly
