import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Define all potential scopes for testing
SCOPES = [
    'https://www.googleapis.com/auth/analytics.readonly',
    'https://www.googleapis.com/auth/tagmanager.readonly',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/bigquery.readonly',
    'https://www.googleapis.com/auth/devstorage.read_only',
    'https://www.googleapis.com/auth/webmasters.readonly', # Search Console
]

def get_credentials(json_filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(BASE_DIR, json_filename)
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            creds_data = json.load(f)
        if creds_data.get('type') == 'service_account':
            private_key = creds_data.get('private_key', '')
            if private_key and '\\n' in private_key:
                creds_data['private_key'] = private_key.replace('\\n', '\n')
            return service_account.Credentials.from_service_account_info(creds_data, scopes=SCOPES)
    
    raise Exception(f"Credentials not found at {json_path}")

def test_drive(creds):
    print("\n--- Testing Google Drive Access ---")
    try:
        service = build('drive', 'v3', credentials=creds)
        results = service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print("No files found visible to this SA.")
        else:
            print(f"Success! Found {len(items)} visible files:")
            for item in items:
                print(f"- {item['name']} ({item['id']})")
    except Exception as e:
        print(f"Drive Access Failed: {e}")

def test_analytics(creds):
    print("\n--- Testing Google Analytics (GA4) Access ---")
    try:
        service = build('analyticsadmin', 'v1alpha', credentials=creds)
        accounts = service.accounts().list().execute()
        items = accounts.get('accounts', [])
        if not items:
            print("No GA4 accounts found visible to this SA.")
        else:
            print(f"Success! Found {len(items)} GA4 accounts:")
            for acc in items:
                print(f"- {acc['displayName']} ({acc['name']})")
    except Exception as e:
        print(f"Analytics Access Failed: {e}")

def test_gtm(creds):
    print("\n--- Testing Google Tag Manager Access ---")
    try:
        service = build('tagmanager', 'v2', credentials=creds)
        accounts = service.accounts().list().execute()
        items = accounts.get('account', [])
        if not items:
            print("No GTM accounts found visible to this SA.")
        else:
            print(f"Success! Found {len(items)} GTM accounts:")
            for acc in items:
                print(f"- {acc['name']} ({acc['accountId']})")
    except Exception as e:
        print(f"GTM Access Failed: {e}")

def test_bigquery(creds):
    print("\n--- Testing BigQuery Access ---")
    try:
        service = build('bigquery', 'v2', credentials=creds)
        # Note: BigQuery projects are slightly different to list
        # We check the default project the SA belongs to first
        project_id = creds.project_id
        print(f"Listing datasets in project: {project_id}")
        datasets = service.datasets().list(projectId=project_id).execute()
        items = datasets.get('datasets', [])
        if not items:
            print(f"No BigQuery datasets found in project {project_id}.")
        else:
            print(f"Success! Found {len(items)} datasets:")
            for ds in items:
                print(f"- {ds.get('datasetReference', {}).get('datasetId')}")
    except Exception as e:
        print(f"BigQuery Access Failed: {e}")

def test_gcs(creds):
    print("\n--- Testing Google Cloud Storage Access ---")
    try:
        # Using the json client for GCS usually involves a different library, 
        # but the discovery API works for listing buckets too.
        service = build('storage', 'v1', credentials=creds)
        project_id = creds.project_id
        print(f"Listing buckets in project: {project_id}")
        buckets = service.buckets().list(project=project_id).execute()
        items = buckets.get('items', [])
        if not items:
            print(f"No GCS buckets found in project {project_id}.")
        else:
            print(f"Success! Found {len(items)} buckets:")
            for b in items:
                print(f"- {b['name']}")
    except Exception as e:
        print(f"GCS Access Failed: {e}")

def test_search_console(creds):
    print("\n--- Testing Google Search Console Access ---")
    try:
        service = build('searchconsole', 'v1', credentials=creds)
        sites = service.sites().list().execute()
        items = sites.get('siteEntry', [])
        if not items:
            print("No Search Console sites found visible to this SA.")
        else:
            print(f"Success! Found {len(items)} sites:")
            for site in items:
                print(f"- {site['siteUrl']} (Permission: {site['permissionLevel']})")
    except Exception as e:
        print(f"Search Console Access Failed: {e}")

def main():
    JSON_FILE = 'bubbly-cascade-398303-5f3dd0a21703.json'
    print(f"Deep Investigation of Service Account: {JSON_FILE}")
    
    try:
        creds = get_credentials(JSON_FILE)
        print(f"Service Account Email: {creds.service_account_email}")
        print(f"Project ID: {creds.project_id}")
        
        test_drive(creds)
        test_analytics(creds)
        test_gtm(creds)
        test_bigquery(creds)
        test_gcs(creds)
        test_search_console(creds)
        
    except Exception as e:
        print(f"Initialization Error: {e}")

if __name__ == '__main__':
    main()
