import os
import json
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Scopes for reading spreadsheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

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
    # The ID of the Google Spreadsheet to "download"
    SPREADSHEET_ID = '1--copNwH0SNuv0rTmwpiCzVvJ1K_D82QMvku0HPfPtQ'
    
    # Target local filename
    OUTPUT_FILENAME = 'vinpearl_data_model_reconstructed.xlsx'

    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        # 1. Get spreadsheet metadata to list all sheets
        print(f"Reading spreadsheet metadata for: {SPREADSHEET_ID}...")
        spreadsheet = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()
        sheets = spreadsheet.get('sheets', [])
        
        print(f"Found {len(sheets)} sheets. Reconstructing into {OUTPUT_FILENAME}...")

        # 2. Iterate through all sheets and fetch data
        # Note: We use openpyxl as the engine for multi-sheet Excel files
        with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
            for s in sheets:
                title = s.get('properties', {}).get('title', 'Sheet')
                print(f"  Fetching data from sheet: {title}...")
                
                # Fetch all values from this sheet (A1:Z1000 range for safety)
                # If the sheet is very large, this may need adjustment
                result = service.spreadsheets().values().get(
                    spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:Z1000").execute()
                values = result.get('values', [])
                
                if values:
                    # Create DataFrame from raw values
                    df = pd.DataFrame(values)
                    # Use sheet title as tab name (Excel allows max 31 chars)
                    sheet_name = title[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                else:
                    print(f"    (Sheet '{title}' is empty)")

        print(f"\nSuccess! Spreadsheet reconstructed at: {os.path.abspath(OUTPUT_FILENAME)}")
        print("\nNOTE: Because full download was restricted by the owner, this script ")
        print("reconstructed the file by reading all available data via the Sheets API.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTIP: Make sure the service account email is shared with the spreadsheet.")
        print("Shared with: ga4-api@bubbly-cascade-398303.iam.gserviceaccount.com")

if __name__ == '__main__':
    main()
