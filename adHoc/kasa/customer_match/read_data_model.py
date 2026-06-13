import os
import json
import pandas as pd
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Scope for reading spreadsheets
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
    SPREADSHEET_ID = '1--copNwH0SNuv0rTmwpiCzVvJ1K_D82QMvku0HPfPtQ'
    # Based on the screenshot: table starts at B6 and ends around E17
    RANGE_NAME = "'Dimensions & Metrics'!B6:E30"

    print(f"Fetching data from spreadsheet: {SPREADSHEET_ID}...")
    
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                    range=RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            print('No data found in the specified range.')
            return

        # Use the first row as headers
        headers = values[0]
        data = values[1:]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Clean up: remove rows that are entirely empty or purely whitespace
        df = df.dropna(how='all')
        df = df[df[headers[0]].str.strip() != '']
        
        # Display the table
        print("\n--- Data Model Table ---\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        
        # Save to CSV for convenience
        output_file = 'data_model_output.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSuccess! Table saved to {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTIP: Make sure the service account email is shared with the spreadsheet.")
        print("Service Account Email: ga4-api@bubbly-cascade-398303.iam.gserviceaccount.com")

if __name__ == '__main__':
    main()
