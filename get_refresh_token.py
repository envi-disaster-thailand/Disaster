from google_auth_oauthlib.flow import InstalledAppFlow

# Write-capable Google Drive scope.
# Required for creating the ECMWF run folder and uploading PNG products
# to the existing Shared Drive PNG folder.
SCOPES = ["https://www.googleapis.com/auth/drive"]

flow = InstalledAppFlow.from_client_secrets_file(
    "client_secret.json",
    SCOPES,
)

creds = flow.run_local_server(
    port=0,
    access_type="offline",
    prompt="consent",
)

print("\nOAuth completed successfully.")
print("Refresh token:")
print(creds.refresh_token)
