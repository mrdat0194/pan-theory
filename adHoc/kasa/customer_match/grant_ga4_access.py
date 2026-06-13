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
    # ID for "Vortex Data" account discovered in previous diagnostic
    ACCOUNT_ID = '231663578' 
    PARENT = f"accounts/{ACCOUNT_ID}"
    
    # User to grant access to
    # USER_EMAIL = 'kasatria@core-arena-291909.iam.gserviceaccount.com'
    USER_EMAIL = 'udemysharedcontent@gmail.com'
    # Predefined roles in GA4 Admin API:
    # roles/viewer, roles/editor, roles/admin, etc.
    # In GA4 Admin API v1beta/v1alpha, it's often 'predefinedRoles/admin'
    ROLE = 'predefinedRoles/admin'

    try:
        creds = get_credentials()
        # Using GA4 Admin API v1alpha as accessBindings is available there
        service = build('analyticsadmin', 'v1alpha', credentials=creds)

        print(f"Attempting to grant {ROLE} access to {USER_EMAIL} on {PARENT}...")
        
        # Create access binding
        # Reference: https://developers.google.com/analytics/devguides/config/admin/v1/rest/v1beta/accounts.accessBindings/create
        access_binding = {
            'roles': [ROLE],
            'user': USER_EMAIL
        }
        
        result = service.accounts().accessBindings().create(
            parent=PARENT,
            body=access_binding
        ).execute()
        
        print("\nSuccess! Access granted successfully.")
        print(f"Access Binding: {result.get('name')}")
        print(f"Assigned User: {result.get('user')}")
        print(f"Assigned Roles: {', '.join(result.get('roles', []))}")

    except Exception as e:
        print(f"Execution Error: {e}")
        print("\nTIP: Make sure the service account has 'Admin' level permissions on the GA4 account to manage user access.")

if __name__ == '__main__':
    main()
