from google.oauth2.credentials import Credentials
import os
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

def main():
    token_path = 'token.json'
    
    if not os.path.exists(token_path):
        print(f"Error: {token_path} not found.")
        print("You need to sign in first to generate this file.")
        return

    try:
        # 1. Load the existing credentials
        creds = Credentials.from_authorized_user_file(token_path, scopes)
            
        print("--- Diagnostic: Using All Granted OAuth Permissions ---")
        
        # 2. Check Scopes
        scopes = getattr(creds, 'scopes', [])
        print(f"Active Scopes in Token: {', '.join(scopes)}")

        # 3. Refresh (if possible)
        if creds and creds.expired and creds.refresh_token:
            print("\nToken expired. Attempting silent refresh...")
            try:
                creds.refresh(Request())
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
                print("Refresh Successful!")
            except Exception as e:
                print(f"Refresh Failed: {e}")
                print("TIP: This usually means the session has been revoked or expired permanently.")
                print("Action: Delete 'token.json' and run a login script to re-authorize.")
                return

        # 4. Use Drive Scope (list files)
        if any('drive' in s for s in scopes):
            print("\n--- Testing Google Drive Permissions ---")
            try:
                drive_service = build('drive', 'v3', credentials=creds)
                results = drive_service.files().list(pageSize=10).execute()
                items = results.get('files', [])
                print(f"Drive Access: SUCCESS. Found {len(items)} files:")
                for item in items:
                    print(f"  - {item['name']}")
            except Exception as e:
                print(f"Drive Access: FAILED. Error: {e}")

        # 5. Use Analytics Scope (if present)
        if any('analytics' in s for s in scopes):
            print("\n--- Testing Google Analytics Permissions ---")
            try:
                # v1beta for Admin API
                analytics_service = build('analyticsadmin', 'v1beta', credentials=creds)
                results = analytics_service.accounts().list().execute()
                items = results.get('accounts', [])
                print(f"Analytics Access: SUCCESS. Found {len(items)} accounts.")
            except Exception as e:
                print(f"Analytics Access: FAILED. Error: {e}")
        else:
            print("\nAnalytics: Not authorized in this token (Scope missing).")

        # 6. Use Ads Scope (if present)
        if any('adwords' in s for s in scopes):
            print("\n--- Testing Google Ads Permissions ---")
            print("Google Ads requires a Developer Token, which is not set in this test script.")
        else:
            print("\nAds: Not authorized in this token (Scope missing).")

    except Exception as e:
        print(f"\nExecution Error: {e}")
        if 'invalid_grant' in str(e):
            print("\nCRITICAL: Your current 'token.json' is invalid or revoked.")
            print("To fix this, you MUST delete 'token.json' and re-login.")

if __name__ == '__main__':
    main()
