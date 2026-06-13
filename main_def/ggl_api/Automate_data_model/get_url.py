from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from main_def import MAIN_DIR, credentials, tokens, get_url, to_sheet

import pandas as pd

# If modifying these scopes, delete the file token.pickle.
SCOPES = [ 'https://www.googleapis.com/auth/presentations', 'https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/presentations.readonly', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.appdata','https://www.googleapis.com/auth/drive.file']


"""Shows basic usage of the Sheets API.
Prints values from a sample spreadsheet.80
"""
creds = None
# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists(tokens):
    with open(tokens, 'rb') as token:
        creds = pickle.load(token)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            credentials, SCOPES)
        creds = flow.run_local_server(port=8080)
    # Save the credentials for the next run
    with open(tokens, 'wb') as token:
        pickle.dump(creds, token)

service = build('sheets', 'v4', credentials=creds)

# # Call the Sheets API
# sheet = service.spreadsheets()

def gspread_values(gsheet_id, sheet_name):
    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=gsheet_id,
                                range=sheet_name).execute()
    values = result.get('values', [])
    return values

def get_df_from_speadsheet(gsheet_id: str, sheet_name: str):
    # need to optimize to read df from column_index: int = 0 (default = 0)
    data = gspread_values(gsheet_id, sheet_name)
    # data = main()
    column = data[0]
    check_fistrow = data[1]
    x = len(column) - len(check_fistrow)
    k = [None] * x
    check_fistrow.extend(k)  # if only have column name but all data of column null =>> error
    row = data[2:]
    row.insert(0, check_fistrow)
    df = pd.DataFrame(row, columns=column).apply(lambda x: x.str.strip()).fillna(value='').astype(str)
    # df.apply(lambda x: x.str.strip()).fillna(value='').astype(str)
    return df

if __name__ == '__main__':
    # Change this spreadsheet ID
    # Yapo
    SAMPLE_SPREADSHEET_ID = to_sheet[0]
    # SAMPLE_SPREADSHEET_ID = "1N2BSFl37f7MZf9CwbwL07N7_0Zmlkq4GPmOfsizMeT4"

    SAMPLE_RANGE_NAME = 'Step0'
    df3 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)
    print(df3['CaptureURL'])
    save_path = get_url
    df3['CaptureURL'].to_csv(save_path, index=False)
