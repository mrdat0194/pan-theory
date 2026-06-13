import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from main_def import MAIN_DIR, credentials, tokens, save_path_sheet, title_sheet, title_test_plan


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

def create_sheet(title):

    spreadsheet = {"properties": {"title": title}}
    spreadsheet = (
        service.spreadsheets()
        .create(body=spreadsheet, fields="spreadsheetId")
        .execute()
    )

    print(f"Spreadsheet ID: {(spreadsheet.get('spreadsheetId'))}")

    return spreadsheet.get("spreadsheetId")


def copy_sheet_tab(from_sheet,to_sheet, sheet_id, newName):

    # The ID of the sheet to copy. Everybody has access!!!

    copy_sheet_to_another_spreadsheet_request_body = {
      "destinationSpreadsheetId": to_sheet
    }

    request = service.spreadsheets().sheets().copyTo(spreadsheetId=from_sheet, sheetId=sheet_id, body=copy_sheet_to_another_spreadsheet_request_body)
    response = request.execute()

    def _batch( requests):
        body = {
            'requests': requests
        }
        return service.spreadsheets().batchUpdate(spreadsheetId=to_sheet, body=body).execute()

    def renameSheet(sheetId, newName):
        return _batch({
            "updateSheetProperties": {
                "properties": {
                    "sheetId": str(sheetId),
                    "title": newName,
                },
                "fields": "title",
            }
        })

    renameSheet(response['sheetId'], newName)

    print("Done " + str(sheet_id))

def move_sheet(file_id, folder_id):
    service = build("drive", "v3", credentials=creds)

    # pylint: disable=maybe-no-member
    # Retrieve the existing parents to remove
    file = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents"))
    # Move the file to the new folder
    file = (
        service.files()
        .update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        )
        .execute()
    )
    return file.get("parents")

if __name__ == '__main__':
    title = title_sheet
    from_sheet = "1tzwlqobwa_CWQH5wulsFd2itgywu78NXp2fIn2rSiSQ"
    to_sheet = create_sheet(title)
    copy_sheet_tab(from_sheet, to_sheet, 1644599131, "CopyStep")
    copy_sheet_tab(from_sheet, to_sheet, 19961054, "Step0")
    copy_sheet_tab(from_sheet, to_sheet, 2092577969, "Dimensions & Metrics")
    copy_sheet_tab(from_sheet, to_sheet, 1439550923, "Data Model")
    move_sheet(to_sheet,'1vYGGGvdX7LN3pi4l-5O64_g3YeCGsi6A')
    save_path = save_path_sheet

    import csv
    file = open(save_path, 'w', newline='')

    # Create a CSV writer object
    csv_writer = csv.writer(file)

    # Write strings to the file
    csv_writer.writerow([to_sheet])

    # Close the file
    file.close()

    print(to_sheet)

    # Step 5:
    if title_test_plan is not None:
        title = title_test_plan
        from_sheet = '1H-Th3XdYAZLbzNCPYHAgmu9PL4eqba51YdTfb7w1N6Q'
        to_sheet = create_sheet(title)
        copy_sheet_tab(from_sheet, to_sheet, 859547738, "Testplan - Web")
        copy_sheet_tab(from_sheet, to_sheet, 1520383332, "Testplan - App")
        copy_sheet_tab(from_sheet, to_sheet, 1505683671, "Pic")
        copy_sheet_tab(from_sheet, to_sheet, 1628567292, "Resource")
        move_sheet(to_sheet, '1vYGGGvdX7LN3pi4l-5O64_g3YeCGsi6A')
        save_path = save_path_sheet

        import csv

        file = open(save_path, 'w', newline='')

        # Create a CSV writer object
        csv_writer = csv.writer(file)

        # Write strings to the file
        csv_writer.writerow([to_sheet])

        # Close the file
        file.close()

        print(to_sheet)

