# Cannot use
from __future__ import print_function
import os.path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

from main_def import credentials, tokens
import os
# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly','https://www.googleapis.com/auth/drive.file' ]

def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(tokens):
        creds = Credentials.from_authorized_user_file(tokens, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(tokens, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    # # Call the Drive v3 API
    # image_path = r"C:\Users\mrdat\PycharmProjects\pan-theory\main_def\ggl_api\kasa\Automate_data_model\Pic"
    target_folder_id = '1TTpwAAzmBTLKDOtPjETKFOQAyzHAN_wW'
    # Get the path to the directory you want to list files in

    # List all the files in the directory
    # files = os.listdir(image_path)

    # Print the list of files
    # for file in files:
    #     file_metadata = {
    #         'name': os.path.basename(file),
    #         'parents': [target_folder_id]
    #     }

        # Create a MediaFileUpload object to represent the image file
        # media = MediaFileUpload(image_path + "\\" +  file, mimetype='image/jpeg')

        # # Create a request to upload the file
        # request = service.files().create(body=file_metadata, media_body=media)
        #
        # # Execute the request
        # response = request.execute()

    results = service.files().list(q="'" + target_folder_id + "' in parents",
                                   fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
            # service.permissions().create(body={"role": "reader", "type": "anyone"}, fileId=item['id']).execute()

    # print(response)


if __name__ == '__main__':
    main()

