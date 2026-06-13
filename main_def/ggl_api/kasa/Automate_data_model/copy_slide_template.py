from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import os
from main_def import MAIN_DIR, credentials, tokens
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive.file']

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
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open(tokens, 'wb') as token:
        pickle.dump(creds, token)
def copy_presentation(presentation_id, copy_title):
    """
    Creates the copy Presentation the user has access to.
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    service = build('drive', 'v3', credentials=creds)

    try:
        body = {"name": copy_title}
        drive_response = (
            service.files().copy(fileId=presentation_id, body=body).execute()
        )
        presentation_copy_id = drive_response.get("id")

    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Presentations  not copied")
        return error

    return presentation_copy_id
def move_slide(file_id, folder_id):
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

    # Select option
    option = "Step 3"
    # presentation_id = "1_qBr_fa7aimyCl8ByUHi4GFlVYg6VGmLgIT3bkN3SyY" # Default template Dev Guide
    presentation_id = "1_qBr_fa7aimyCl8ByUHi4GFlVYg6VGmLgIT3bkN3SyY"

    if option == "Step 3":
        # Copy Dev Guide template
        new_slide = "[Tayara] GA4 Web Developer Guideline"
        presentation_copy_id = copy_presentation(presentation_id, new_slide)
        print(presentation_copy_id)
        move_slide(presentation_copy_id, '1vYGGGvdX7LN3pi4l-5O64_g3YeCGsi6A')
    elif option == "Step 4":
        # Copy Dev Guide to Event Summary
        presentation_id = ""
        new_slide = "[Avito] Event Summary"
        presentation_copy_id = copy_presentation(presentation_id, new_slide)
        move_slide(presentation_copy_id, '1vYGGGvdX7LN3pi4l-5O64_g3YeCGsi6A')
        print(presentation_copy_id)
