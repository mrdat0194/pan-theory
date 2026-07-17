from google.oauth2.credentials import Credentials
import os

def main():
    # Path to the token file generated from a previous login
    token_path = 'token.json'
    
    if not os.path.exists(token_path):
        print(f"Error: {token_path} not found in this directory.")
        print("This file is usually created after the first successful login using client_secrets.json.")
        return

    try:
        # Load the credentials from the pickle file
        creds = Credentials.from_authorized_user_file(token_path, None)
            
        print("--- OAuth Token Diagnostic ---")
        
        # Identify the account if possible
        # credentials objects from google-auth often don't store the email directly inside the pickle 
        # unless it was specifically requested in the scopes, but we can check.
        if hasattr(creds, 'service_account_email'):
             print(f"Associated Email: {creds.service_account_email}")
        
        # List the scopes (permissions)
        scopes = getattr(creds, 'scopes', [])
        if not scopes:
            print("No scopes found in the current token.")
        else:
            print(f"\nThe existing token currently has access to These {len(scopes)} Scopes:")
            for scope in scopes:
                # Providing a friendly name if possible
                parts = scope.split('/')
                friendly = parts[-1] if parts else scope
                print(f"  * {scope} ({friendly})")
        
        # Check expiry
        if hasattr(creds, 'expired'):
            status = "EXPIRED (needs refresh)" if creds.expired else "Valid (Active)"
            print(f"\nToken Status: {status}")

        print("\nNOTE: These scopes determine what the script can do on your behalf.")

    except Exception as e:
        print(f"Error reading {token_path}: {e}")
        print("The file might be corrupted or created with a different library version.")

if __name__ == '__main__':
    main()
