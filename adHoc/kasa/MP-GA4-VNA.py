from __future__ import print_function
from ga4mp import GtagMP
from ga4mp.store import DictStore
import json

# refund

import pandas as pd
import numpy as np

import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def gspread_values(gsheet_id, sheet_name):
    # Call the Sheets API
    sheet = service().spreadsheets()
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


def createList(r1, r2):
    return list(range(r1, r2 + 1))


def get_df_from_sheet(gsheet_id: str, sheet_name: str):
    '''
    Only take
    :param gsheet_id:
    :param sheet_name:
    :return: dataframe
    '''
    SAMPLE_SPREADSHEET_ID = gsheet_id
    SAMPLE_RANGE_NAME = sheet_name
    values = main()
    row_index = values[0]
    length = len(row_index)
    values = [[None if x == '' else x for x in c] for c in values]
    values = np.array([xi + [None] * (length - len(xi)) if (length - len(xi)) >= 0 else xi[0:length] for xi in values],
                      dtype=object)
    axis = 1
    range_list = createList(0, length - 1)
    np_array = np.take(values[1:], range_list, axis)
    df2 = pd.DataFrame(np.array(np_array), columns=[row_index])
    return df2


def service():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    return service


if __name__ == '__main__':
# The ID and range of a sample spreadsheet.one

    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    SAMPLE_SPREADSHEET_ID = '15mZzXaSUn6M3gbP4FEFzGDCSox5wx8RG-1NWONo4Ao8'
    SAMPLE_RANGE_NAME = 'Testing'

    df3 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)

    row_indexes = df3.index

    for row in row_indexes:
        transaction_id = df3['transaction_id'].loc[row]
        refund = df3['purchase'].loc[row]
        print(transaction_id)
        print(refund)

    # with same transaction_id need different client_id
    with open("credentials/credentials.json", 'r') as json_file:
        client_config = json.load(json_file)
        measurement_id = client_config['MEASUREMENT_ID']
        api_secret = client_config['API_SECRET']
        client_id = client_config['CLIENT_ID']
        gtag_tracker = GtagMP(api_secret=api_secret, measurement_id=measurement_id, client_id=client_id)

        # Create a new event for purchase.
        purchase_event = gtag_tracker.create_new_event(name="purchase")

        # Set transaction_id, value, and currency.
        purchase_event.set_event_param(name="transaction_id", value="t20002")
        purchase_event.set_event_param(name="currency", value="USD")
        purchase_event.set_event_param(name="value", value=200000)

        # Send the event to GA4 immediately.
        event_list = [purchase_event]
        gtag_tracker.send(events=event_list)