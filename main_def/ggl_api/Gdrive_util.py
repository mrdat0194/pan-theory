from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

from timer import print_param, timer, print_param_sheet

from Gsheet_util import *



# --- PyDrive2 GoogleAuth using just service account ---

# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = 'ga4-gtm-automation.json'  # or secret_jsons, as appropriate

# Define required scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.install'
]


# Define the settings for service account authentication
settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": SERVICE_ACCOUNT_FILE
    }
}

# Create a GoogleAuth instance with the service account settings
gauth = GoogleAuth(settings=settings)

# Authenticate using the service account
gauth.ServiceAuth()

# Create a GoogleDrive instance with the authenticated session

drive = GoogleDrive(gauth)

def safe_gsheet_request(some_operation:Any):
    def f( *arg, **kwarg):
        # research to make it arg
        max_retries=15
        base_delay=1
        max_delay=32
        retries = 0
        t = 0
        while retries < max_retries:
            try:
                # Simulate an operation that might fail, like an API call or network request
                # if callable(some_operation):
                result = some_operation(*arg, **kwarg)
                # else: result = some_operation  # Success
                return result  # If operation is successful, return the result
            except gspread.exceptions.APIError as e:  # Replace with more specific exceptions as needed
                retries += 1
                t+= 1
                random_number_milliseconds = random.randint(0, 1000) / 1000
                wait_time = min( t**(base_delay + random_number_milliseconds) , max_delay)
                print(f"Attempt {retries} failed, retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        raise gspread.exceptions.APIError(f"Failed after {max_retries} retries.")
    return f

def check_exist_folder(folder_id, folder_name):
    folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
    existing_folders = {folder1['title']: folder1['id'] for folder1 in folder_list}
    print("Existing Folders:", existing_folders)
    if folder_name in existing_folders:
        return existing_folders[folder_name]
    else:
        return False

def get_spreadsheet_files(folder_id: str) -> list:
    """ get_spreadsheet_files("16TbtgVlry3A_4_7kq-bYPNdLeVZGhPNp") """
    file_list = drive.ListFile({'q':f""" '{folder_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false """}).GetList() 
    return [file['id'] for file in file_list]

def compare_sheets(gc, folder_id1, folder_id2, sheet_name1, sheet_name2):
    """
    Compare two Google Sheets in specific cells.

    Args:
        folder_id1 (str): ID of the first folder.
        folder_id2 (str): ID of the second folder.
        sheet_name1 (str): Name of the first sheet.
        sheet_name2 (str): Name of the second sheet.
        cell_range (str): Range of cells to compare (e.g., 'A1:B2').

    Returns:
        bool: True if the cells match, False otherwise.
    """
    sheet_ids1 = get_spreadsheet_files(folder_id1)
    sheet_ids2 = get_spreadsheet_files(folder_id2)

    sheet1 = None
    sheet2 = None

    for sheet_id in sheet_ids1:
        sheet = gc.open_by_key(sheet_id)
        if sheet_name1 in sheet.title:
            sheet1 = sheet
            break

    for sheet_id in sheet_ids2:
        sheet = gc.open_by_key(sheet_id)
        if sheet_name2 in sheet.title:
            sheet2 = sheet
            break

    worksheets1 = [worksheet for worksheet in sheet1.worksheets() if '_container' in worksheet.title] 
    worksheets2 = [worksheet for worksheet in sheet2.worksheets() if '_container' in worksheet.title] 

    if sheet1 and sheet2:
        values1 = sheet1.worksheet(worksheets1[0].title).get_all_values()
        values2 = sheet2.worksheet(worksheets2[0].title).get_all_values()
        result = values1 == values2
        if result is False:
            from datetime import datetime
            file_name = worksheets2[0].title +"_"+ str(datetime.now().strftime("%Y%m%d%H%M%S"))
            to_sheet_id = copy_sheet(sheet2.id, folder_id2, file_name)
            print(to_sheet_id)
            return result
        else: 
            return result
    else:
        return 'Not match'

def listFolder(parent_folder_id:str) -> list:
    """ listFolder("16TbtgVlry3A_4_7kq-bYPNdLeVZGhPNp") """
    forder_list = drive.ListFile({'q': "'%s' in parents and trashed=false and mimeType='application/vnd.google-apps.folder' and trashed=false"  % parent_folder_id }).GetList()
    return [[file['id'],file['title']  ] for file in forder_list]

def create_folder(parent_folder_id:str, subfolder_name:str) -> object:
    """ create_folder("1pCksj1OMm4pDnLmW-2XTcEjGAm60GfE8", "Manufacture_workspace") """
    newFolder = drive.CreateFile({"title": subfolder_name, 
                                  "parents": [{"kind": "drive#fileLink", "id": parent_folder_id}],
                                  "mimeType": "application/vnd.google-apps.folder"})
    return newFolder.Upload()

def create_spreadsheet(folder_id : str, spreadsheet_name:str ="Untiled File") -> object:
    """ create_spreadsheet("1pCksj1OMm4pDnLmW-2XTcEjGAm60GfE8", "Avito_worker") """
    file = drive.CreateFile({"title": spreadsheet_name,
                             "parents": [{"kind": "drive#fileLink", "id": folder_id}],
                             "mimeType": "application/vnd.google-apps.spreadsheet"})
    return file.Upload()

def create_doc(folder_id : str, docs_name:str ="Untiled File", content:str ="") -> object:
    """Create a new Google Doc in the specified folder with given name and content.
    
    Args:
        folder_id: ID of the folder to create doc in
        docs_name: Name of the new doc
        content: Initial content for the doc
        
    Returns:
        The created file metadata
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.install'
    ]
    
    SERVICE_ACCOUNT_FILE = SERVICE_ACCOUNT_FILE

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    docs_service = build('docs', 'v1', credentials=credentials)

    file_metadata = {
        'name': docs_name,
        'mimeType': 'application/vnd.google-apps.document',
        'parents': [{"kind": "drive#fileLink", "id": folder_id}]
    }
    file = service.files().create(
        body=file_metadata,
        fields='id'
    ).execute()

    print(file)

def sharePermission(file_id: str, email: str = None, role: str = 'reader', type: str = 'user'):
    """
    Share a file with specific permissions using Google Drive API.
    
    Args:
        file_id (str): The ID of the file to share
        email (str, optional): Email address of the user to share with. Required if type is 'user'
        role (str): The role to grant. Options: 'reader', 'writer', 'commenter', 'owner'
        type (str): The type of permission. Options: 'user', 'group', 'domain', 'anyone'
        
    Returns:
        dict: The permission resource if successful, None otherwise
    """
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        SCOPES = [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/drive.install'
        ]
        SERVICE_ACCOUNT_FILE = SERVICE_ACCOUNT_FILE

        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)

        permission_metadata = {
            'role': role,
            'type': type
        }
        
        if type in ['user', 'group'] and email:
            permission_metadata['emailAddress'] = email

        permission = service.permissions().create(
            fileId=file_id,
            body=permission_metadata,
            fields='id,emailAddress,role,type'
        ).execute()

        print(f"Permission granted: {permission}")
        return permission

    except Exception as e:
        print(f"Error sharing file: {e}")
        return None

# sharePermission("15Ji2ZwQk4NvBBvjsP_XnwvvjaC74FI8l0gAZ0BO9KD8", role="reader", type="anyone") 

def get_folder_and_files(folder_id):
    """
    Retrieve the parent folder metadata and all files within a specified Google Drive folder.

    Args:
        folder_id (str): The ID of the folder to inspect.

    Returns:
        dict: {
            'folder': {folder metadata},
            'files': [list of file metadata dicts]
        }
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.install'
    ]
    SERVICE_ACCOUNT_FILE = SERVICE_ACCOUNT_FILE

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)

    try:
        folder_metadata = service.files().get(
            fileId=folder_id,
            fields='id, name, mimeType, parents'
        ).execute()
    except Exception as e:
        print(f"Error retrieving folder metadata: {e}")
        return None

    files = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed=false"
    while True:
        try:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, parents)',
                pageToken=page_token
            ).execute()
            items = response.get('files', [])
            files.extend(items)
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
        except Exception as e:
            print(f"Error retrieving files in folder: {e}")
            break

    print(f"Folder: {folder_metadata.get('name')} (ID: {folder_metadata.get('id')})")
    print(f"Total files found in folder: {len(files)}")
    for item in files:
        print(f"File Name: {item.get('name')}, File ID: {item.get('id')}, MimeType: {item.get('mimeType')}")

    return {
        'folder': folder_metadata,
        'files': files
    }

def append_text_to_new_page(file_id: str, text_to_append: str, folder_id: str):
    """
    Appends text to a new page at the end of a Google Doc.

    Args:
        file_id: The ID of the Google Doc to append to.
        text_to_append: The text to append.
    """
    try:
        file = drive.CreateFile({'id': file_id})
        print('title: %s, id: %s' % (file['title'], file['id']))
        content = file.GetContentString(mimetype='text/plain')
        page_break = '-' * 60
        update = content + "\n" + page_break + "\n" + text_to_append
        file.SetContentString(update)
        return file.Upload() 
    except Exception as e:
        print(f"Error appending to new page: {e}")
        return False

def append_text_to_new_page_v2(file_id: str, text_to_append: str) -> bool:
    """
    Appends text to a new page at the end of a Google Doc using Google Docs API.
    This version uses proper page break requests and service account authentication.

    Args:
        file_id: The ID of the Google Doc to append to
        text_to_append: The text to append on the new page
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        SCOPES = [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/docs'
        ]

        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )

        docs_service = build('docs', 'v1', credentials=credentials)
        document = docs_service.documents().get(documentId=file_id).execute()
        end_index = document.get('body').get('content')[-1].get('endIndex')

        requests = [
            {
                'insertPageBreak': {
                    'location': {
                        'index': end_index - 1
                    }
                }
            },
            {
                'insertText': {
                    'location': {
                        'index': end_index
                    },
                    'text': text_to_append
                }
            }
        ]

        docs_service.documents().batchUpdate(
            documentId=file_id,
            body={'requests': requests}
        ).execute()

        return True

    except Exception as e:
        print(f"Error appending to new page: {e}")
        return False

def move_sheet(file_id, folder_id):
    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata()
    previous_parents = file['parents'][0]['id']
    file['parents'] = [{"kind": "drive#fileLink", "id": folder_id}]
    file.Upload()
    file['parents'] = [{"kind": "drive#fileLink", "id": previous_parents}]
    file.Upload()
    return file['parents']

@safe_gsheet_request
@print_param_sheet("1-itYOjEbQs_T7e5fnzJCBxJ0amiJJmQUdZPEDHOH-tw", "SheetId")
def copy_sheet(file_id, folder_id, title):
    """
    Copies a Google Sheet to a new location in Google Drive using the service account.

    Args:
        file_id (str): The ID of the spreadsheet to copy.
        folder_id (str): The ID of the destination folder.
        title (str): The new title for the copied spreadsheet.

    Returns:
        str: The ID of the copied spreadsheet, or None if failed.
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    
    IMPERSONATE_USER = "service@vn-vortexdata.com"

    # Scopes for Drive
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    # Authenticate with domain-wide delegation
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    delegated_credentials = credentials.with_subject(IMPERSONATE_USER)

    drive_service = build('drive', 'v3', credentials=delegated_credentials)

    # Copy the file
    try:  
        # Copy a file into a specific folder by setting the 'parents' field to the destination folder ID
        file_metadata = {
            "name": title,
            "parents": [folder_id]
        }
        copied_file = drive_service.files().copy(
            fileId=file_id,
            body=file_metadata
        ).execute()
        time.sleep(3)
        print(copied_file['id'])
        return title, copied_file['id']
    except Exception as e:
        import traceback
        print("An error occurred while copying the spreadsheet:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None


# copy_sheet('12xNwx3zl7kPRzKuxKv8sPipTujb54stvvln-HaaPid8', '1_bwYVDZ6VYcp2GMYxvLllnwCw-xjE2jy', title = 'vortex')

@print_param_sheet()
def copy_sheet_package(file_id, folder_id, title):
    """
    Copies a Google Sheet to a new location in Google Drive using the service account.

    Args:
        file_id (str): The ID of the spreadsheet to copy.
        folder_id (str): The ID of the destination folder.
        title (str): The new title for the copied spreadsheet.

    Returns:
        str: The ID of the copied spreadsheet, or None if failed.
    """
    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    
    IMPERSONATE_USER = "service@vn-vortexdata.com"

    # Scopes for Drive
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    # Authenticate with domain-wide delegation
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    delegated_credentials = credentials.with_subject(IMPERSONATE_USER)

    drive_service = build('drive', 'v3', credentials=delegated_credentials)

    # Copy the file
    try:  
        # Copy a file into a specific folder by setting the 'parents' field to the destination folder ID
        file_metadata = {
            "name": title,
            "parents": [folder_id]
        }
        copied_file = drive_service.files().copy(
            fileId=file_id,
            body=file_metadata
        ).execute()
        print(copied_file['id'])
        return copied_file['id']
    except Exception as e:
        import traceback
        print("An error occurred while copying the spreadsheet:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

# copy_sheet_package('1v_4ogSXDazUcOe_UeLrMxBIer0PH-6HHYdqVIUy2NF0', '1888l5wBHYMXZHRbLYIQu_Zw0mFdBY-vX', title = '[JVS][GTM Package]')


# @print_param("folder_id.csv", save_path)
@safe_gsheet_request
@print_param_sheet("1-itYOjEbQs_T7e5fnzJCBxJ0amiJJmQUdZPEDHOH-tw", "FolderId")
def create_folder_in_drive(folder_name, parent_folder_id=None, client= None):
    """
    Creates a folder in Google Drive using PyDrive2 and returns its ID.

    Args:
        folder_name (str): Name of created folder
        parent_folder_id (str, optional): ID of the parent folder in Drive. Defaults to the root.

    Returns:
        str: The folder ID if successful, None otherwise.
    """
    try:

        # # First, check if a folder with the same name exists in the parent folder (or root if not specified)
        # # Corrected query construction for PyDrive2 (Google Drive API v2)
        # # The 'name' field is not valid in v2; use 'title' instead.
  
        data_frame_source = spreadsheet_to_df('1-itYOjEbQs_T7e5fnzJCBxJ0amiJJmQUdZPEDHOH-tw', name_worksheet='FolderId', idx_cell="A1").to_pandas()
        # Find the row where 'Client' matches the given client
        match = data_frame_source[data_frame_source['Client'] == client]
        if not match.empty:
            # If found, return "ok" and the existing folder id from the 'FolderId' column
            existing_folder_id = match.iloc[0]['FolderId']
            return "ok", existing_folder_id
        # If not found, create the folder as before
        folder_metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_folder_id:
            folder_metadata['parents'] = [{'id': parent_folder_id}]
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        return client, folder['id']
    except Exception as e:
        print(f"Error creating folder in Drive: {e}")
        return None