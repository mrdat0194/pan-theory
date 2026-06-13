# Cannot use
from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

from main_def import MAIN_DIR, credentials, tokens, image_path, target_folder_id
import os

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly','https://www.googleapis.com/auth/drive.file']

def uploadDrive(image_path,target_folder_id):
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
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
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(tokens, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)

    # # Call the Drive v3 API

    # Get the path to the directory you want to list files in

    # List all the files in the directory
    files = os.listdir(image_path)

    results = service.files().list(q="'" + target_folder_id + "' in parents",
                                   fields="nextPageToken, files(id, name)").execute()

    for file in results['files']:
        file_id = file['id']
        service.files().delete(fileId=file_id).execute()

    # Print the list of files and upload, it can duplicate what in the drive, need to move to trash permanently.
    for file in files:
        file_metadata = {
            'name': os.path.basename(file),
            'parents': [target_folder_id]
        }

        # Create a MediaFileUpload object to represent the image file
        image_pic = os.path.join(image_path,file)

        print(image_pic)

        media = MediaFileUpload(image_pic, mimetype='image/jpeg')

        # Create a request to upload the file
        request = service.files().create(body=file_metadata, media_body=media)

        # Execute the request
        request.execute()

    results = service.files().list(q="'" + target_folder_id + "' in parents",
                                   fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])


    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
            service.permissions().create(body={"role": "reader", "type": "anyone"}, fileId=item['id']).execute()

if __name__ == '__main__':

    uploadDrive(image_path,target_folder_id)


