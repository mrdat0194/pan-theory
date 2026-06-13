functions-framework==3.*
pandas
google-auth>=1.14.1
google-api-python-client==1.8.0
google-cloud
google-cloud-bigquery
google-cloud-storage

from google.cloud import bigquery

import pandas as pd
from googleapiclient.discovery import build
import os
import functions_framework
from google.oauth2 import service_account

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# If modifying these scopes, delete the file token.pickle.

SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/drive.file']

"""Shows basic usage of the Sheets API.
Prints values from a sample spreadsheet.80
"""

credentials = service_account.Credentials.from_service_account_file(
    'credentials.json')

creds = credentials.with_scopes(
    SCOPES)

service = build('sheets', 'v4', credentials=creds)


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


@functions_framework.http
def hello_bigquery(request):
    # Change this spreadsheet ID
    SAMPLE_SPREADSHEET_ID = '1pnPvw21HM3TY9QRmK7QyBsg15nnFmO9KC1dCdxB2asg'
    SAMPLE_RANGE_NAME = 'input'
    df = get_df_from_speadsheet(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)
    # df = str(df3['event_name'][1])
    # Construct a BigQuery client object.
    client = bigquery.Client(project='shopkasatriavn')

    # TODO(developer): Set table_id to the ID of the table to create.
    table_id = "shopkasatriavn.cf_test.testing_upload"

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("event_name", "STRING"),
            bigquery.SchemaField("time", "STRING"),
        ],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        # time_partitioning=bigquery.TimePartitioning(
        #     type_=bigquery.TimePartitioningType.DAY,
        #     field="date",  # Name of the column to use for partitioning.
        #     expiration_ms=7776000000,  # 90 days.
        # ),
    )

    load_job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    # Make an API request.

    load_job.result()  # Wait for the job to complete.

    table = client.get_table(table_id)

    return "Loaded {} rows to table {}".format(table.num_rows, table_id)