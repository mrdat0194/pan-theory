import os
import json
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Scope for reading Drive files
# Note: service accounts only see files they own or that are shared with them.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the service account JSON that worked previously
    json_path = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            creds_data = json.load(f)
        if creds_data.get('type') == 'service_account':
            # Normalize private key line breaks
            private_key = creds_data.get('private_key', '')
            if private_key and '\\n' in private_key:
                creds_data['private_key'] = private_key.replace('\\n', '\n')
            return service_account.Credentials.from_service_account_info(creds_data, scopes=SCOPES)
    
    raise Exception(f"Credentials not found at {json_path}")

def main():
    # Specific folder ID from the user
    FOLDER_ID = '1nnhP2Xf9gLoJgXHTeDwLmsg9cnnoIEoH'
    
    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        print(f"Listing all files in Google Drive folder: {FOLDER_ID}...")
        
        all_files = []
        page_token = None
        
        # Query to list files in the specific folder
        query = f"'{FOLDER_ID}' in parents and trashed = false"
        
        while True:
            # Call the Drive v3 API
            results = service.files().list(
                q=query,
                pageSize=100, 
                pageToken=page_token,
                fields="nextPageToken, files(id, name, mimeType)"
            ).execute()
            
            items = results.get('files', [])
            all_files.extend(items)
            
            page_token = results.get('nextPageToken')
            if not page_token:
                break

        if not all_files:
            print('\nNo files found in the specified folder.')
            print("\nTIP: Service accounts can only see files/folders that have been shared with them.")
            print("Shared with: ga4-api@bubbly-cascade-398303.iam.gserviceaccount.com")
        else:
            # Create DataFrame
            df = pd.DataFrame(all_files)
            
            # Display results summary
            print(f'\nTotal files found: {len(all_files)}')
            print(df.head(20).to_string(index=False))
            
            # Save to CSV
            output_file = 'gdrive_folder_files_list.csv'
            df.to_csv(output_file, index=False)
            print(f"\nSuccess! List of {len(all_files)} files saved to {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTIP: Make sure the service account email is shared with the folder.")
        print("Service Account Email: ga4-api@bubbly-cascade-398303.iam.gserviceaccount.com")

if __name__ == '__main__':
    main()
