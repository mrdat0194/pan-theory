import os
import pickle
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

def main():
    # Path to the token file from previous login
    token_path = 'token.pickle'
    
    if not os.path.exists(token_path):
        print(f"Error: {token_path} not found in this directory.")
        return

    try:
        # Load the credentials
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
            
        # Refresh the token if it has expired (silently, without browser)
        if creds and creds.expired and creds.refresh_token:
            print("Access token expired. Refreshing using the saved refresh token...")
            creds.refresh(Request())
            # Save the refreshed credentials back to the pickle file
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

        # Build the Drive service
        service = build('drive', 'v3', credentials=creds)

        print("\n--- Listing Google Drive Files (using OAuth Token) ---")
        
        # Fetch the most recent 20 files
        # fields="files(id, name, mimeType)" retrieves only the necessary data
        results = service.files().list(
            pageSize=20, 
            fields="nextPageToken, files(id, name, mimeType)",
            orderBy="modifiedTime desc" # Show most recent first
        ).execute()
        
        items = results.get('files', [])

        if not items:
            print("No files found in your account.")
        else:
            print(f"Success! Found {len(items)} most recent files:")
            for item in items:
                mime = item.get('mimeType', 'unknown')
                print(f"- {item['name']} (ID: {item['id']}) [{mime}]")

    except Exception as e:
        print(f"Execution Error: {e}")
        print("\nTIP: If the token is corrupted, you might need to delete token.pickle and sign in again.")

if __name__ == '__main__':
    main()
