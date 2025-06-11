from __future__ import print_function
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from main_def import MAIN_DIR, credentials, tokens, FIX_SLIDE_ID,new_slide, target_folder_id, get_url, slide_image_path
import csv

import pickle
import os.path
import time
import pandas as pd

def copy_presentation(drive_service,presentation_id, copy_title):
    """
    Creates the copy Presentation the user has access to.
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    try:
        body = {"name": copy_title}
        drive_response = (
            drive_service.files().copy(fileId=presentation_id, body=body).execute()
        )
        presentation_copy_id = drive_response.get("id")

    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Presentations  not copied")
        return error

    return presentation_copy_id

def create_slide(service,presentation_id, slide_id):
    """
    Creates the Presentation the user has access to.
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.\n
    """

    try:
        # Add a slide at index 1 using the predefined
        # 'TITLE_AND_TWO_COLUMNS' layout and the ID page_id.
        duplicate_object_request = {
            "objectId": slide_id
        }

        batch_update_request = {
            "requests": [{
                "duplicateObject": duplicate_object_request
            }]
        }

        response = (
            service.presentations()
            .batchUpdate(presentationId=presentation_id, body=batch_update_request)
            .execute()
        )

        create_slide_response = response.get("replies")[0].get("duplicateObject")['objectId']
        print(f"Created slide with ID:{create_slide_response}")
    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Slides not created")
        return error


def drive_get_file(creds, presentation_id, target_folder_id, new_slide):
    """
    Get drive get items vs copy slide

    """

    service = build('drive', 'v3', credentials=creds)

    PRESENTATION_ID = copy_presentation(service, presentation_id, new_slide)

    permission = {
        'type': 'anyone',
        'role': 'writer',
    }

    request = service.permissions().create(fileId=PRESENTATION_ID, body=permission)
    request.execute()

    print("https://docs.google.com/presentation/d/"+str(PRESENTATION_ID))

    file = open(slide_image_path, 'w', newline='')

    # Create a CSV writer object
    csv_writer = csv.writer(file)

    # Write strings to the file
    csv_writer.writerow(["https://docs.google.com/presentation/d/"+str(PRESENTATION_ID)])

    # Close the file
    file.close()

    results = service.files().list(q="'" + target_folder_id + "' in parents",
                                   fields="nextPageToken, files(id, name)").execute()

    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))

    return PRESENTATION_ID, items

def create_image(presentation_id, new_slide, target_folder_id):
    """
    Creates images the user has access to.
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/presentations.readonly', 'https://www.googleapis.com/auth/presentations', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.appdata','https://www.googleapis.com/auth/drive.file' ]

    # The ID of a sample presentation.

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

    try:
        service = build("slides", "v1", credentials=creds)

        presentation = (
            service.presentations().get(presentationId=presentation_id).execute()
        )

        PRESENTATION_ID, items = drive_get_file(creds, presentation_id, target_folder_id, new_slide)

        slides = presentation.get("slides")

        for i in range(len(items)-1):
            # Base on Template
            create_slide(service, PRESENTATION_ID, slides[21]['objectId'])

        presentation = (
            service.presentations().get(presentationId=PRESENTATION_ID).execute()
        )

        slides = presentation.get("slides")

        # URL name import

        save_path = get_url

        linkes = pd.read_csv(save_path)
        # print(linkes)
        linkes.dropna(inplace=True)
        linkes = linkes.reset_index(drop=True)
        # print(linkes)

        n = 0
        for i, slide in enumerate(slides[21:]):
            try:
                link = linkes['CaptureURL'].loc[i]
            except:
                quit
            # print(link)
            name = str(i) + ".png"
            # print(name)
            # print(items)
            filtered_items = filter(lambda x: x['name'] == name, items)

            fileid = list(filtered_items)[0]['id']
            # print(fileid)

            # pylint: disable=invalid-name

            image_id = "MyImage_" + str(i)

            element_id = "MyText_" + str(i)

            pt350 = {"magnitude": 350, "unit": "PT"}

            emu4M = {"magnitude": 9000000, "unit": "EMU"}

            IMAGE_URL = "https://drive.google.com/uc?export=download&id=" + fileid
            print(IMAGE_URL)
            # IMAGE_URL = "https://drive.google.com/uc?export=download&id=1gLU-3Li79jR4EqBHlypCpXVG4OY_Qs8E"

            requests = []
            requests.append(
                {
                    "createImage": {
                        "objectId": image_id,
                        "url": IMAGE_URL,
                        "elementProperties": {
                            "pageObjectId": slide['objectId'],
                            "size": {"height": emu4M, "width": emu4M},
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 100000,
                                "translateY": 100000,
                                "unit": "EMU",
                            },
                        },
                    },
                })

            # Execute the request.
            for t in range(0, 1):
                # while True:
                try:
                    body = {"requests": requests}
                    time.sleep(4)
                    response = (
                        service.presentations()
                        .batchUpdate(presentationId=PRESENTATION_ID, body=body)
                        .execute()
                    )
                    time.sleep(4)
                    create_image_response = response.get("replies")[0].get("createImage")
                    print(f"Created image with ID: {(create_image_response.get('objectId'))}")
                except Exception as e:
                    print(e)
                    pass

            requests = []
            requests.append(
                {
                    "createShape": {
                        "objectId": element_id,
                        "shapeType": "TEXT_BOX",
                        "elementProperties": {
                            "pageObjectId": slide['objectId'],
                            "size": {"height": pt350, "width": pt350},
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 350,
                                "translateY": 100,
                                "unit": "PT",
                            },
                        },
                    },
                }
            )
            requests.append(
                # Insert text into the box, using the supplied element ID.
                {
                    "insertText": {
                        "objectId": element_id,
                        "insertionIndex": 0,
                        "text": link,
                    }
                },
            )
            # Execute the request.
            try:
                body = {"requests": requests}

                time.sleep(4)
                response = (
                    service.presentations()
                    .batchUpdate(presentationId=PRESENTATION_ID, body=body)
                    .execute()
                )
                time.sleep(4)
                create_text_response = response.get("replies")[0].get("createShape")
                print(f"Created Text with ID: {(create_text_response.get('objectId'))}")
            except Exception as e:
                print(e)
                pass

    except HttpError as error:
        print(f"An error occurred: {error}")
        print("Images not created")
        return error


if __name__ == "__main__":
    # Run
    create_image(FIX_SLIDE_ID,new_slide, target_folder_id )
