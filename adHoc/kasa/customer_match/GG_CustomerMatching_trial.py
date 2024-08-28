from __future__ import print_function
import pickle
import os.path
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


# https://gspread.readthedocs.io/en/latest/user-guide.html#using-gspread-with-pandas
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


def main():
    """Shows basic usage of the Sheets API.
    Prints values from a sample spreadsheet.80
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

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
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


def gspread_values(gsheet_id, sheet_name):
    # Call the Sheets API
    sheet = service().spreadsheets()
    result = sheet.values().get(spreadsheetId=gsheet_id,
                                range=sheet_name).execute()
    values = result.get('values', [])
    return values


def update_value(list_result: list, range_to_update: str, gsheet_id: str):
    body = {
        'values': list_result  # list_result is array 2 dimensional (2D)
    }
    result = service().spreadsheets().values().update(
        spreadsheetId=gsheet_id, range=range_to_update,
        valueInputOption='RAW', body=body).execute()


def extractDigits(lst):
    return list(map(lambda el: [el], lst))


if __name__ == '__main__':

    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    SAMPLE_SPREADSHEET_ID = '1-VkrDEcXPIGuZuBOXtuZ_IlIlmW-tPx1fVduAxG6kGU'
    SAMPLE_RANGE_NAME1 = 'RSVP'
    SAMPLE_RANGE_NAME2 = 'SĐT'
    SAMPLE_RANGE_NAME3 = 'Email 1'
    SAMPLE_RANGE_NAME4 = 'Email 2'
    SAMPLE_RANGE_NAME5 = 'Email 3'
    SAMPLE_RANGE_NAME6 = 'Email 4'

    df_RSVP = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME1)
    df_SDT = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME2)
    df_EMAIL1 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME3)
    # df_EMAIL2 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME4)
    # df_EMAIL3 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME5)
    # df_EMAIL4 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME6)
    # print(df_RSVP.head())
    # print(df_RSVP.columns)
    # print(df_SDT.head())
    # print(df_SDT.columns)


    Extract_RSVP = df_RSVP[['Order Id','First Name','Last Name','Email','Phone Number']]
    Extract_SDT = df_SDT[['Order ID', 'Phone']]
    Extract_SDT.rename(columns={'Order ID': 'Order Id'}, inplace=True)

    Extract_join = pd.merge(Extract_RSVP, Extract_SDT, on= 'Order Id', how='left')

    # Extract_join = Extract_join[0:1898].drop_duplicates(subset=['First Name','Last Name','Phone Number'])
    Extract_join = Extract_join[0:1000].drop_duplicates(subset=['First Name','Last Name','Phone Number'])
    print(len(Extract_join))

    Extract_join['check_email'] = 'NA'


    # print(Extract_join)

    df_result = pd.read_csv("/Users/petern/Desktop/Mypage/pan-theory/adHoc/kasa/Customer_Match_Upload_template100.csv")
    df_result = df_result.drop(['Email.1', 'Zip.1', 'Phone.1'], axis=1)
    # print(df_result)

    # print(Extract_join)

    Extract_join = Extract_join.reset_index()
    row_indexes = Extract_join.index
    n = 1
    for row_order  in row_indexes:
        # Hashed
        if n == 1:
            # hashlib.sha256('1234').hexdigest()
            # df_result['Email'].loc[row_order] = df_EMAIL1['check_email'].loc[row_order].strip().lower()
            # print(hashlib.sha256(df_EMAIL1['email'].loc[row_order+ n].strip().lower().encode('utf-8')))
            # print(str(df_EMAIL1['email'].loc[row_order+ n].strip().lower()))
            df_result['Email'].loc[row_order] = df_EMAIL1['email'].loc[row_order].strip().lower()
            # print(row_order)
            # print(no_accent_vietnamese(Extract_join['First Name'].loc[row_order+n ].strip().lower()))
            df_result['First Name'].loc[row_order] = no_accent_vietnamese(Extract_join['First Name'].loc[row_order].strip().lower())
            # df_result['First Name'].loc[row_order] = no_accent_vietnamese(df_result['First Name'].loc[row_order])
            df_result['Last Name'].loc[row_order] = no_accent_vietnamese(Extract_join['Last Name'].loc[row_order].strip().lower())
            # df_result['Last Name'].loc[row_order] = no_accent_vietnamese(df_result['Last Name'].loc[row_order])
            df_result['Country'].loc[row_order] = 'VN'.strip()
            df_result['Zip'].loc[row_order] = ''.strip()
            df_result['Phone'].loc[row_order] = ('+84'+ Extract_join['Phone'].loc[row_order].strip()[-9:])

            # df_result['Email'].loc[row_order] = hashlib.sha256(
            #     df_EMAIL1['email'].loc[row_order].strip().lower().encode('utf-8')).hexdigest()
            # # print(row_order)
            # # print(no_accent_vietnamese(Extract_join['First Name'].loc[row_order+n ].strip().lower()))
            # df_result['First Name'].loc[row_order] = hashlib.sha256(
            #     no_accent_vietnamese(Extract_join['First Name'].loc[row_order].strip().lower()).encode(
            #         'utf-8')).hexdigest()
            # # df_result['First Name'].loc[row_order] = no_accent_vietnamese(df_result['First Name'].loc[row_order])
            # df_result['Last Name'].loc[row_order] = hashlib.sha256(
            #     no_accent_vietnamese(Extract_join['Last Name'].loc[row_order].strip().lower()).encode(
            #         'utf-8')).hexdigest()
            # # df_result['Last Name'].loc[row_order] = no_accent_vietnamese(df_result['Last Name'].loc[row_order])
            # df_result['Country'].loc[row_order] = 'VN'.strip()
            # df_result['Zip'].loc[row_order] = ''.strip()
            # df_result['Phone'].loc[row_order] = hashlib.sha256(
            #     ('+84' + Extract_join['Phone'].loc[row_order].strip()[-9:]).encode('utf-8')).hexdigest()

            # take 2 list of 100 users break as required
            if row_order == 100:
                break
        else:
            # df_result['Email'].loc[row_order] = df_EMAIL1['check_email'].loc[row_order].strip().lower()
            df_result['Email'].loc[row_order] = df_EMAIL1['email'].loc[row_order+ n].strip().lower()
            # print(row_order)
            # print(no_accent_vietnamese(Extract_join['First Name'].loc[row_order+n ].strip().lower()))
            df_result['First Name'].loc[row_order] = no_accent_vietnamese(Extract_join['First Name'].loc[row_order+n ].strip().lower())
            # df_result['First Name'].loc[row_order] = no_accent_vietnamese(df_result['First Name'].loc[row_order])
            df_result['Last Name'].loc[row_order] = no_accent_vietnamese(Extract_join['Last Name'].loc[row_order+ n].strip().lower())
            # df_result['Last Name'].loc[row_order] = no_accent_vietnamese(df_result['Last Name'].loc[row_order])
            df_result['Country'].loc[row_order] = 'VN'.strip()
            df_result['Zip'].loc[row_order] = ''.strip()
            df_result['Phone'].loc[row_order] = '+84'+ Extract_join['Phone'].loc[row_order+ n].strip()[-9:]
            # take 2 list of 100 users break as required
            if row_order == 100:
                break

    # print(len(df_result.drop_duplicates(subset=['Phone'], keep = False)))
    print(df_result)

    df_result.to_csv("/Users/petern/Desktop/Mypage/pan-theory/adHoc/kasa/Customer_Match_Upload_sample100.csv", index=False)