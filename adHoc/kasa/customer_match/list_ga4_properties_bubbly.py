import os
import json
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scope required to read properties
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']

def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the bubbly-cascade service account JSON as requested
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
    try:
        creds = get_credentials()
        # Using GA4 Admin API v1beta
        service = build('analyticsadmin', 'v1beta', credentials=creds)

        print(f"Bubbly SA: Listing all GA4 accounts and Properties...")
        
        # 1. List all accounts visible to the service account
        accounts_results = service.accounts().list().execute()
        accounts = accounts_results.get('accounts', [])
        
        if not accounts:
            print("No GA4 accounts found visible to this service account.")
            return

        print(f"Found {len(accounts)} accounts. Scanning properties for each...")
        
        all_properties = []
        
        for acc in accounts:
            acc_name = acc.get('name') # format: accounts/ID
            acc_display = acc.get('displayName')
            print(f"\n--- Account: {acc_display} ({acc_name}) ---")
            
            # 2. List properties for this account
            try:
                # Use filter parent:accounts/ID
                prop_results = service.properties().list(
                    filter=f"parent:{acc_name}"
                ).execute()
                
                properties = prop_results.get('properties', [])
                if not properties:
                    print("  No properties found.")
                else:
                    print(f"  Found {len(properties)} properties:")
                    for prop in properties:
                        p_name = prop.get('name')
                        p_display = prop.get('displayName')
                        p_id = p_name.split('/')[-1]
                        service_level = prop.get('serviceLevel', 'N/A')
                        print(f"  - {p_display} (ID: {p_id}) [License: {service_level}]")
                        
                        all_properties.append({
                            'Account Name': acc_display,
                            'Account ID': acc_name,
                            'Property Name': p_display,
                            'Property ID': p_id,
                            'License': service_level,
                            'Created': prop.get('createTime'),
                            'Industry': prop.get('industryCategory', 'N/A'),
                            'Resource Name': p_name
                        })
            except Exception as e:
                print(f"  Error fetching properties for {acc_display}: {e}")

        # 3. Aggregated Export
        if all_properties:
            df = pd.DataFrame(all_properties)
            output_file = 'ga4_all_accounts_properties_bubbly.csv'
            df.to_csv(output_file, index=False)
            print(f"\nScan Complete! Total properties found: {len(all_properties)}")
            print(f"Aggregated list saved to: {os.path.abspath(output_file)}")
        else:
            print("\nFinal Result: No properties found across any accessible accounts.")

    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == '__main__':
    main()
