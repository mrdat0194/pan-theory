from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
# Try to load saved client credentials
# gauth.LoadCredentialsFile("client_secrets.json")

if gauth.credentials is None:
   # Authenticate if they're not there
   gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
   # Refresh them if expired
   gauth.Refresh()
else:
   # Initialize the saved creds
   gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds1.txt")

#
drive = GoogleDrive(gauth)
#
# import glob, os
# import shutil
#
# dates_str = cend_start1.date().strftime("%d-%m-%Y")
# os.chdir("/Users/petern/Desktop/GonJoy/Right Time/Sankey_report")
# mImageLoc = "/Users/petern/Desktop/GonJoy/Right Time/Sankey_report"
# desLoc = "/Users/petern/Desktop/GonJoy/report"
# tgt_folder_id = "1DoTcL5fL3HM_nAfmZdVlMOOOUWeaFBZA"
#
# for file in glob.glob("*.html"):
#     with open(file, "r") as f:
#         fn = os.path.basename(f.name)
#         current_name = mImageLoc + "/" + fn
#         change_name = mImageLoc + "/" + "SankeyChart_Userflow_" + dates_str + ".html"
#         Des_name = desLoc + "/" + "SankeyChart_Userflow_" + dates_str + ".html"
#         os.rename(current_name, change_name)
#
# for file in glob.glob("*.html"):
#     with open(file, "r") as f:
#         fn = os.path.basename(f.name)
#         mImageLoc1 = mImageLoc + "/" + fn
#         file_drive = drive.CreateFile({'title': fn, 'mimeType': 'text/html',
#                                        'parents': [{"kind": "drive#childList", "id": tgt_folder_id}]})
#         file_drive.SetContentString(f.read())
#         file_drive.Upload()
#
# shutil.move(change_name, Des_name)
#
# os.chdir("/Users/petern")


#
# file_list = drive.ListFile({'q': 'trashed=false', 'maxResults': 100}).GetList()
# for file1 in file_list:
#  print('title: %s, id: %s, mine: %s' % (file1['title'], file1['id'], file1['mimeType'] ))
#

# def service():
#     """Shows basic usage of the Sheets API.
#     Prints values from a sample spreadsheet.
#     """
#     creds = None
#     # The file token.pickle stores the user's access and refresh tokens, and is
#     # created automatically when the authorization flow completes for the first
#     # time.
#     if os.path.exists(token_path):
#         with open(token_path, 'rb') as token:
#             creds = pickle.load(token)


