import requests
import re
import json
import os

# Load keys from environment variable GOOGLE_API_KEYS (comma-separated)
keys = [k.strip() for k in os.environ.get("GOOGLE_API_KEYS", "").split(",") if k.strip()]

project_ids = [
    "bubbly-cascade-398303",
    "ga4-gtm-automation-468107",
    "core-arena-291909"
]

def check_match(k, p_id):
    # This endpoint will return an error if the key project doesn't match the URL project
    url = f"https://generativelanguage.googleapis.com/v1beta/projects/{p_id}/models?key={k}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return "MATCH (API Enabled)"
        
        data = r.json()
        message = data.get('error', {}).get('message', '')
        
        # If it says "project ... does not match", it's NOT a match
        if "does not match" in message:
            return "NO MATCH"
        
        # If it says "permission denied" or "API not enabled", it might still be the right project
        if p_id in message:
            return "MATCH (But API Error: " + message[:50] + "...)"
            
        return "UNKNOWN (" + str(r.status_code) + ")"
    except Exception as e:
        return f"Error: {e}"

print(f"{'API Key (Full)':<45} | {'Project ID':<25} | {'Result':<20}")
print("-" * 100)
for k in keys:
    for p in project_ids:
        res = check_match(k, p)
        if "MATCH" in res:
             print(f"{k:<45} | {p:<25} | {res:<20}")
print("-" * 100)
