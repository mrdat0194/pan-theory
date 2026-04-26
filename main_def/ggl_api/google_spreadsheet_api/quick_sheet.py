from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
import numpy as np

# https://developers.google.com/sheets/api/quickstart/python
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

def readFromGspread():
    # Using gspread hoàn thành nhé:

    from gspread_dataframe import get_as_dataframe

    worksheet = service().spreadsheets()

    df = get_as_dataframe(worksheet, parse_dates=True, usecols=[0,2], skiprows=1, header=None)

    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = Credentials.from_service_account_file(
        '/Users/petern/Downloads/pythonapppredict-ecbedb2c4ce5.json',
        scopes=scopes
    )

    gc = gspread.authorize(credentials)

    sh = gc.open_by_key("1xpFtEI7AhIm5ytc6t1u8DXWOOUTpdjPCKKLND7Hamm4")

    print(sh.sheet1.get('A1'))

def extractDigits(lst):
    return list(map(lambda el: [el], lst))

def artistImageInsert():
    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    SAMPLE_SPREADSHEET_ID = '1xpFtEI7AhIm5ytc6t1u8DXWOOUTpdjPCKKLND7Hamm4'
    SAMPLE_RANGE_NAME = 'Artist_image'
    # Artist_image


    df3 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)

    df3 = df3['Memo'].str.lower()

    df3['insert'] = '0'

    artist_filter = df3[
        (df3['Artist_uuid'].notnull())
        & (df3['Artist_uuid'] != '')
        & (df3['url_to_add'].notnull())
        & (df3['url_to_add'] != '')
        & (df3['Memo'] == 'added')
        ][['url_to_add', 'Artist_uuid', 'Memo']].drop_duplicates('Artist_uuid', keep='first')

    row_indexes = artist_filter.index
    for row in row_indexes:
        artist_uuid = artist_filter['Artist_uuid'].loc[row]
        url_to_add = artist_filter['url_to_add'].loc[row]
        str_insert = f"insert into v4.crawlingtasks(Id, ActionId,objectid ,TaskDetail, Priority) values (uuid4(), 'OA9CPKSUT6PBGI1ZHPLQUPQCGVYQ71S9','{artist_uuid}',JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.url','{url_to_add}','$.object_type','artist','$.when_exists','replace','$.PIC','Copy of  Artist Page 30.11.2020'),99);"
        df3.loc[row, 'insert'] = str_insert
    # SAME AS
    # for row in artist_filter.itertuples():
    #     str_insert = f"""insert into v4.crawlingtasks(Id, ActionId,objectid ,TaskDetail, Priority) values (uuid4(), 'OA9CPKSUT6PBGI1ZHPLQUPQCGVYQ71S9','{row.Artist_uuid}',JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.url','{row.url_to_add}','$.object_type','artist','$.when_exists','replace','$.PIC','Copy of  Artist Page 30.11.2020'),99);"""
    #     df3.loc[row.Index, 'insert'] = str_insert

    return df3
    #
    # list_update = df3['insert'].tolist()
    # list_update.insert(0, 'insert')
    # list_update = extractDigits(list_update)
    # range_name = f"{SAMPLE_RANGE_NAME}!J1"
    # update_value(list_update, range_name, SAMPLE_SPREADSHEET_ID)

if __name__ == '__main__':
    # Artist_image
    # df_artist_img = artistImageInsert()
    # print(df_artist_img['insert'])

    # Artist_wiki
    pd.set_option("display.max_rows", None, "display.max_columns", 60, 'display.width', 1000)
    SAMPLE_SPREADSHEET_ID = '1xpFtEI7AhIm5ytc6t1u8DXWOOUTpdjPCKKLND7Hamm4'
    SAMPLE_RANGE_NAME = 'Artist_wiki'

    df3 = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)

    df3['Memo'] = df3['Memo'].str.lower()
    df3['insert'] = '0'
    # print(df3['content to add'].apply(lambda x: 1 if len(x) > 10 else 0))
    df3['insert'] = df3['content to add'].apply(lambda x: 1 if len(x) > 10 else 0)
    df3['content to add'] = df3['content to add'].str.replace("'", r"\'")
    df3['content to add'] = df3['content to add'].str.replace('"', r'\"')
    print(df3['content to add'][df3['insert'] == 1])



    df3['insert'] = '0'

    artist_filter = df3[
          (df3['content to add'].notnull())
        & (df3['Artist_uuid'] != '')
        & (df3['url_to_add'].notnull())
        & (df3['url_to_add'] != '')
        & (df3['Memo'] == "added")
        & (df3['content to add'].notnull())
        ][['url_to_add', 'Artist_uuid', 'Memo']].drop_duplicates('Artist_uuid', keep='first')

    row_indexes = artist_filter.index
    for row in row_indexes:
            artist_uuid = artist_filter['Artist_uuid'].loc[row]
            url_to_add = artist_filter['url_to_add'].loc[row]
            str_insert = f"insert into v4.crawlingtasks(Id, ActionId,objectid ,TaskDetail, Priority) values (uuid4(), 'OA9CPKSUT6PBGI1ZHPLQUPQCGVYQ71S9','{artist_uuid}',JSON_SET(IFNULL(crawlingtasks.TaskDetail, JSON_OBJECT()), '$.url','{url_to_add}','$.object_type','artist','$.when_exists','replace','$.PIC','Copy of  Artist Page 30.11.2020'),99);"
            df3.loc[row, 'insert'] = str_insert
    print(df3['insert'][0])

    list_update = df3['insert'].tolist()
    list_update.insert(0, 'insert')
    list_update = extractDigits(list_update)
    range_name = f"{SAMPLE_RANGE_NAME}!J1"
    print(list_update)
    print(range_name)
    # update_value(list_update, range_name, SAMPLE_SPREADSHEET_ID)

    # IFS (OR( F20 = "added"; AND( F20 = "not ok"; NOT(G20 = "none"); NOT(E20 = "none"))) ;"UPDATE artists
    # SET
    # Info = JSON_SET(Info, '$.wiki', JSON_OBJECT('brief','" & E20 & "'),'$.wiki_url','" & G20 & "')
    # WHERE
    # id = '"& B20 &"'"; AND(F20 = "not ok"; OR(G20 = "none"; E20 = "none")) ; "UPDATE artists
    # SET
    # Info = JSON_REMOVE(Info, '$.wiki','$.wiki_url')
    # WHERE
    # id = '"& B20 &"'" )


    # import gspread
    # from google.oauth2.service_account import Credentials
    #
    # scopes = [
    #     'https://www.googleapis.com/auth/spreadsheets',
    #     'https://www.googleapis.com/auth/drive'
    # ]
    #
    # credentials = Credentials.from_service_account_file(
    #     '/Users/petern/Downloads/pythonapppredict-ecbedb2c4ce5.json',
    #     scopes=scopes
    # )
    #
    # gc = gspread.authorize(credentials)
    #
    # sh = gc.open_by_key("1xpFtEI7AhIm5ytc6t1u8DXWOOUTpdjPCKKLND7Hamm4")
    #
    # print(sh.sheet1.get('A1'))
    #
    # import pandas as pd
    #
    # from gspread_dataframe import get_as_dataframe
    #
    #
    # worksheet = sh.get_worksheet(0)
    #
    # df = get_as_dataframe(worksheet, parse_dates=True, skiprows=1, header=None )
    #
    # print(df[4])
    #
    # df = pd.read_csv("mycreds1.txt")
