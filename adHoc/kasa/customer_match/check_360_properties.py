import os
from google.analytics.admin import AnalyticsAdminServiceClient
from google.oauth2 import service_account

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_PATH = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')

properties = [
    {"id": "237200408", "name": "VNA"},
    {"id": "258003657", "name": "Vinpearl"},
    {"id": "318969518", "name": "VinWonders"},
]

creds = service_account.Credentials.from_service_account_file(
    CREDS_PATH,
    scopes=["https://www.googleapis.com/auth/analytics.readonly"]
)
client = AnalyticsAdminServiceClient(credentials=creds)

print(f"{'Property':<15} {'ID':<15} {'360?':<8} {'Type'}")
print("-" * 60)
for prop in properties:
    try:
        result = client.get_property(name=f"properties/{prop['id']}")
        is_360 = getattr(result, 'service_level', None)
        prop_type = str(is_360) if is_360 else str(result)
        print(f"{prop['name']:<15} {prop['id']:<15} {str(is_360)}")
    except Exception as e:
        print(f"{prop['name']:<15} {prop['id']:<15} ERROR: {e}")
