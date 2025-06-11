# pip install --upgrade google-cloud-storage
from google.cloud import storage
from google.oauth2.credentials import Credentials
# from googleapiclient.discovery import build
from main_def import MAIN_DIR, credentials, tokens
import os
from main_def import MAIN_DIR, client_secret_store, tokens

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = client_secret_store
# Set up authentication
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
# credential = Credentials.from_authorized_user_file(client_secret_store, SCOPES)
storage_client = storage.Client()

image_path = os.path.join(MAIN_DIR, "ggl_api", "Automate_data_model", "Pic")
files = os.listdir(image_path)

# for file in files:
    # Specify the bucket and file name
    # bucket_name = 'consent-mode/picture-web'
    # file_name = file
    # Upload the file to the bucket
    # local_file_path = os.path.join(image_path,file)

bucket_name = 'consent-mode'
blobs = storage_client.list_blobs(bucket_name,max_results= 10)

for blob in blobs:
    print(blob)

    # blob = storage_client.bucket(bucket_name, user_project='shopkasatriavn').blob(file_name)
    # blob.upload_from_filename(local_file_path, content_type='image/jpeg')
    #
    # # Generate the public URL
    # public_url = blob.public_url
    #
    # print(public_url)
    # break

