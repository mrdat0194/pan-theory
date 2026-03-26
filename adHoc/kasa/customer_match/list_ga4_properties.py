import os
import json
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scope required to read properties
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']

def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the ga4-gtm-automation service account as requested
    json_path = os.path.join(BASE_DIR, 'core-arena.json')
    
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
    # ID for "Vortex Data" account
    ACCOUNT_ID = '364115693' 
    ACCOUNT_NAME = f"accounts/{ACCOUNT_ID}"

    try:
        creds = get_credentials()
        # Using GA4 Admin API v1beta for property listing
        service = build('analyticsadmin', 'v1beta', credentials=creds)

        print(f"Listing properties for account: {ACCOUNT_NAME}...")
        
        # List properties with filter
        # Reference: https://developers.google.com/analytics/devguides/config/admin/v1/rest/v1beta/properties/list
        results = service.properties().list(
            filter=f"parent:{ACCOUNT_NAME}"
        ).execute()
        
        properties = results.get('properties', [])
        
        if not properties:
            print("\nNo properties found for this account.")
        else:
            print(f"\nSuccess! Found {len(properties)} properties:")
            
            data = []
            for prop in properties:
                name = prop.get('name')
                display_name = prop.get('displayName')
                property_id = name.split('/')[-1]
                create_time = prop.get('createTime')
                industry = prop.get('industryCategory', 'N/A')
                
                print(f"- {display_name} (ID: {property_id})")
                
                data.append({
                    'Property ID': property_id,
                    'Display Name': display_name,
                    'Created': create_time,
                    'Industry': industry,
                    'Resource Name': name
                })
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            output_file = 'ga4_properties_list.csv'
            df.to_csv(output_file, index=False)
            print(f"\nFull property list saved to: {os.path.abspath(output_file)}")

    except Exception as e:
        print(f"Execution Error: {e}")
        print("\nTIP: Make sure the service account has at least 'Viewer' access to the account.")

if __name__ == '__main__':
    main()
