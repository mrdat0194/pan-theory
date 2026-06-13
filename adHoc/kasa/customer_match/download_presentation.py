import os
import json
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# Scopes for reading and downloading Drive files
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

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
    # The ID of the Google Presentation to download
    FILE_ID = '1gKt7uJ2rxsZ9ppANBP_wlv_AG5j3seb70_CQ2MdKxNQ'
    
    # MIME type for PDF
    EXPORT_MIME_TYPE = 'application/pdf'
    
    # Target local filename
    OUTPUT_FILENAME = 'google_presentation.pdf'

    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)

        print(f"Exporting Google Presentation (ID: {FILE_ID}) as {OUTPUT_FILENAME}...")
                
        # For Google Slides/Docs/Sheets, use export_media instead of get_media
        request = service.files().export_media(fileId=FILE_ID, mimeType=EXPORT_MIME_TYPE)
        
        # Binary stream to save the file
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download Progress: {int(status.progress() * 100)}%.")
        
        # Write the binary stream to a local file
        with open(OUTPUT_FILENAME, "wb") as f:
            f.write(fh.getbuffer())
        
        print(f"\nSuccess! Presentation downloaded to {os.path.abspath(OUTPUT_FILENAME)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTIP: Make sure the service account email is shared with the presentation.")
        print("Shared with: ga4-api@bubbly-cascade-398303.iam.gserviceaccount.com")

if __name__ == '__main__':
    main()
