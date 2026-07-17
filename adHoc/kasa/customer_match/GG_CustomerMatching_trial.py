from __future__ import print_function
from google.oauth2.credentials import Credentials
import os.path
import sys
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
import numpy as np
import re

import hashlib

def no_accent_vietnamese(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def get_credentials():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(BASE_DIR, 'token.json')
    json_path = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')

    # Check if Gdrive_secrets.json is a service account
    if os.path.exists(json_path):
        import json
        try:
            with open(json_path, 'r') as f:
                creds_data = json.load(f)
            if creds_data.get('type') == 'service_account':
                from google.oauth2 import service_account
                print(f"Using Service Account credentials from: {json_path}")
                print(f"Service Account Email: {creds_data.get('client_email')}")
                # Ensure the private key has correct line breaks and no hidden garbage
                private_key = creds_data.get('private_key', '')
                if private_key:
                    # Remove any non-ASCII characters and weird whitespace, keeping valid PEM chars
                    import re
                    # Keep only A-Z, a-z, 0-9, +, /, =, -, and whitespace (\s includes \n)
                    # This helps if there were non-breaking spaces or other invisible chars
                    private_key = re.sub(r'[^A-Za-z0-9+/=\-\s]', '', private_key)
                    creds_data['private_key'] = private_key
                
                return service_account.Credentials.from_service_account_info(creds_data, scopes=SCOPES)
        except Exception as e:
            print(f"Warning: Could not parse credentials.json as service account: {e}")

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
    # If there are no (valid) credentials available, refresh them or start new flow.
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        except Exception as e:
            print(f"Warning: Could not refresh token: {e}")
            creds = None

    if not creds or not creds.valid:
        if os.path.exists(json_path):
            print("Token expired or missing. Starting new authentication flow using credentials.json...")
            flow = InstalledAppFlow.from_client_secrets_file(json_path, SCOPES)
            creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        else:
            print("CRITICAL ERROR: No valid token.pickle found and credentials.json is missing.")
            print(f"Please place a valid 'credentials.json' from your Google Cloud Project into: {BASE_DIR}")
            return None

    return creds


def main():
    creds = get_credentials()
    if not creds:
        return None

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME1).execute()
    values = result.get('values', [])

    return values


def get_df_from_speadsheet(gsheet_id: str, sheet_name: str):
    data = gspread_values(gsheet_id, sheet_name)
    if not data:
        return pd.DataFrame()
    column = data[0]
    check_fistrow = data[1]
    x = len(column) - len(check_fistrow)
    k = [None] * x
    check_fistrow.extend(k)
    row = data[2:]
    row.insert(0, check_fistrow)
    df = pd.DataFrame(row, columns=column).apply(lambda x: x.str.strip()).fillna(value='').astype(str)
    return df


def service():
    creds = get_credentials()
    if not creds:
        raise Exception("Authentication failed. No valid token or credentials.json.")
    return build('sheets', 'v4', credentials=creds)


def gspread_values(gsheet_id, sheet_name):
    # Call the Sheets API
    sheet = service().spreadsheets()
    result = sheet.values().get(spreadsheetId=gsheet_id,
                                range=sheet_name).execute()
    values = result.get('values', [])
    return values


if __name__ == '__main__':

    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    # The ID of the spreadsheet
    SAMPLE_SPREADSHEET_ID = '1-VkrDEcXPIGuZuBOXtuZ_IlIlmW-tPx1fVduAxG6kGU'
    SAMPLE_RANGE_NAME1 = 'RSVP'
    SAMPLE_RANGE_NAME2 = 'SĐT'
    SAMPLE_RANGE_NAME3 = 'Email 1'

    try:
        df_RSVP = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME1)
        df_SDT = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME2)
        df_EMAIL1 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME3)
    except Exception as e:
        print(f"Failed to fetch data from Sheets: {e}")
        exit(1)

    if df_RSVP.empty or df_SDT.empty or df_EMAIL1.empty:
        print("Fetched data is empty. Check your spreadsheet IDs and ranges.")
        exit(1)

    Extract_RSVP = df_RSVP[['Order Id','First Name','Last Name','Email','Phone Number']]
    Extract_SDT = df_SDT[['Order ID', 'Phone']]
    Extract_SDT.rename(columns={'Order ID': 'Order Id'}, inplace=True)

    Extract_join = pd.merge(Extract_RSVP, Extract_SDT, on= 'Order Id', how='left')
    Extract_join = Extract_join[0:1000].drop_duplicates(subset=['First Name','Last Name','Phone Number'])
    print(f"Processing {len(Extract_join)} records...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(BASE_DIR, "Customer_Match_Upload_template2500.csv")
    
    if not os.path.exists(template_path):
        print(f"Template file not found: {template_path}")
        exit(1)

    df_result = pd.read_csv(template_path)
    df_result = df_result.drop(['Email.1', 'Zip.1', 'Phone.1'], axis=1)

    Extract_join = Extract_join.reset_index()
    row_indexes = Extract_join.index
    n = 1
    for row_order  in row_indexes:
        if n == 1:
            df_result['Email'].loc[row_order] = df_EMAIL1['email'].loc[row_order].strip().lower()
            df_result['First Name'].loc[row_order] = no_accent_vietnamese(Extract_join['First Name'].loc[row_order].strip().lower())
            df_result['Last Name'].loc[row_order] = no_accent_vietnamese(Extract_join['Last Name'].loc[row_order].strip().lower())
            df_result['Country'].loc[row_order] = 'VN'.strip()
            df_result['Zip'].loc[row_order] = ''.strip()
            df_result['Phone'].loc[row_order] = ('+84'+ Extract_join['Phone'].loc[row_order].strip()[-9:])
            if row_order == 100:
                break
        else:
            df_result['Email'].loc[row_order] = df_EMAIL1['email'].loc[row_order+ n].strip().lower()
            df_result['First Name'].loc[row_order] = no_accent_vietnamese(Extract_join['First Name'].loc[row_order+n ].strip().lower())
            df_result['Last Name'].loc[row_order] = no_accent_vietnamese(Extract_join['Last Name'].loc[row_order+ n].strip().lower())
            df_result['Country'].loc[row_order] = 'VN'.strip()
            df_result['Zip'].loc[row_order] = ''.strip()
            df_result['Phone'].loc[row_order] = '+84'+ Extract_join['Phone'].loc[row_order+ n].strip()[-9:]
            if row_order == 100:
                break

    print(df_result.head())

    output_path = os.path.join(BASE_DIR, "Customer_Match_Upload_sample100.csv")
    df_result.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")