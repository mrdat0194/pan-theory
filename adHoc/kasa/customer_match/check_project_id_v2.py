import requests
import re
import json

keys = [
    "AIzaSyCZNhvHu_fXy7OfNa1jm8E8zhk7c3q2L_E",
    "AIzaSyD3HTx9W0v854m_tCjp9cfyXU_F8_wpRR0",
    "AIzaSyDVNm1D_iIemwPLfvZ5zSIfLy-J9GfuNXE",
    "AIzaSyBn9_j-O4JVQaOMmvehVKBC6QQ4SGr_Ypo",
    "AIzaSyBuD1ZDhkhiuKew9diygaLBef0kfy1DBy0",
    "AIzaSyBB6_3Yu4v3Yw98ZCg-ARXfJwkQOd-b-Q4"
]

def check_project_details(k):
    # This endpoint often provides the project ID or number in the error message when mismatched
    url = f"https://generativelanguage.googleapis.com/v1beta/projects/unknown/models?key={k}"
    try:
        r = requests.get(url, timeout=10)
        
        # 1. Header project ID
        p_id = r.headers.get('x-goog-project-id')
        if p_id: return f"ID: {p_id}"
        
        # 2. Body project details
        msg = r.text
        # Common message: "Project 'google.com:bubbly-cascade-398303' not found..."
        match = re.search(r"project '([\w.-]+:[\w.-]+|[\w.-]+)'", msg)
        if match: return f"Identifier: {match.group(1)}"
        
        # Search for project number
        match_n = re.search(r"project ([\w.-]+)", msg)
        if match_n: return f"Number/ID: {match_n.group(1)}"
        
        return f"Status {r.status_code}: {msg[:100]}"
    except Exception as e:
        return f"Error: {e}"

print(f"{'API Key':<45} | {'Project Details'}")
print("-" * 100)
for k in keys:
    print(f"{k:<45} | {check_project_details(k)}")
print("-" * 100)
