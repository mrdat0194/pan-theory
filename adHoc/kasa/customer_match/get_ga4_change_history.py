import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scopes required to read change history
SCOPES = [
    'https://www.googleapis.com/auth/analytics.readonly',
    'https://www.googleapis.com/auth/analytics.edit',
    'https://www.googleapis.com/auth/analytics.manage.users'
]

def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the ga4-gtm-automation service account as requested
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
    # ID for "Vortex Data" account
    ACCOUNT_ID = '231663578' 
    ACCOUNT_NAME = f"accounts/{ACCOUNT_ID}"

    try:
        creds = get_credentials()
        # Using GA4 Admin API v1alpha
        service = build('analyticsadmin', 'v1alpha', credentials=creds)

        print(f"Searching Change History for: {ACCOUNT_NAME}...")
        
        # Body for searchChangeHistoryEvents
        # Reference: https://developers.google.com/analytics/devguides/config/admin/v1/rest/v1beta/accounts/searchChangeHistoryEvents
        body = {
            # You can add filters like windowStart or windowEnd here
            # "windowStart": "2026-03-01T00:00:00Z"
        }
        
        request = service.accounts().searchChangeHistoryEvents(
            account=ACCOUNT_NAME,
            body=body
        )
        
        results = request.execute()
        
        events = results.get('changeHistoryEvents', [])
        
        if not events:
            print("\nNo change history events found for this account recently.")
        else:
            print(f"\nSuccess! Found {len(events)} change history events:")
            
            for event in events:
                change_time = event.get('changeTime')
                user_email = event.get('userEmail')
                actor_type = event.get('actorType')
                changes = event.get('changes', [])
                
                print(f"\n--- Event at {change_time} ---")
                print(f"Actor: {user_email} ({actor_type})")
                
                for change in changes:
                    action = change.get('action')
                    resource = change.get('resource')
                    resource_after = change.get('resourceAfterChange', {})
                    
                    # Try to get a more descriptive name for the resource
                    resource_desc = resource
                    if 'displayName' in resource_after:
                        resource_desc = f"{resource_after['displayName']} ({resource})"
                    
                    print(f"  * Action: {action}")
                    print(f"    Resource: {resource_desc}")

    except Exception as e:
        print(f"Execution Error: {e}")
        print("\nTIP: Make sure the service account has sufficient permissions on the GA4 account.")

if __name__ == '__main__':
    main()
