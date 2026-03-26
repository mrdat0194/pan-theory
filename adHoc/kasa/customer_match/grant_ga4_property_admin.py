import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scopes required to manage users
SCOPES = [
    'https://www.googleapis.com/auth/analytics.edit',
    'https://www.googleapis.com/auth/analytics.manage.users'
]

def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Using the bubbly-cascade service account as in the latest grant_ga4_access.py
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
    # --- CONFIGURATION ---
    # Example: Vietnam Airlines GA4 Property ID
    PROPERTY_ID = '237200408' 
    PARENT = f"properties/{PROPERTY_ID}"
    
    USER_EMAIL = 'udemysharedcontent@gmail.com'
    
    # Predefined role for Admin
    ROLE = 'predefinedRoles/admin'
    # ---------------------

    try:
        creds = get_credentials()
        # Using GA4 Admin API v1alpha for accessBindings
        service = build('analyticsadmin', 'v1alpha', credentials=creds)

        print(f"GRANTING PROPERTY ADMIN ACCESS...")
        print(f"Target Property: {PARENT}")
        print(f"User: {USER_EMAIL}")
        
        access_binding = {
            'roles': [ROLE],
            'user': USER_EMAIL
        }
        
        # Note: Using properties() collection instead of accounts()
        result = service.properties().accessBindings().create(
            parent=PARENT,
            body=access_binding
        ).execute()
        
        print("\nSuccess! Property Admin access granted.")
        print(f"Access Binding: {result.get('name')}")
        print(f"User: {result.get('user')}")

    except Exception as e:
        print(f"Execution Error: {e}")
        print("\nTIP: Ensure the Service Account has 'Admin' level access to the PROPERTY (or its parent account) to manage users.")

if __name__ == '__main__':
    main()
