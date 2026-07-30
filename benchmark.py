import time
import sys
from unittest.mock import MagicMock, patch

# Mock all dependencies
mock_google = MagicMock()
sys.modules['google'] = mock_google
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()
sys.modules['google.oauth2'] = MagicMock()
sys.modules['google.oauth2.credentials'] = MagicMock()
sys.modules['google_auth_oauthlib'] = MagicMock()
sys.modules['google_auth_oauthlib.flow'] = MagicMock()

# create mock HttpError before importing
class MockHttpError(Exception):
    pass
mock_googleapiclient = MagicMock()
mock_googleapiclient_errors = MagicMock()
mock_googleapiclient_errors.HttpError = MockHttpError
sys.modules['googleapiclient'] = mock_googleapiclient
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.errors'] = mock_googleapiclient_errors

sys.modules['pandas'] = MagicMock()

import pandas as pd
mock_df = MagicMock()
mock_df['CaptureURL'].loc.__getitem__.return_value = 'http://example.com'
pd.read_csv.return_value = mock_df

# Patch sys.path so we can import easily or just import the file directly
import os
sys.path.append(os.path.abspath('main_def/ggl_api/Automate_data_model/'))

# Because the script imports variables from the global scope of main_def...
import types
main_def_mock = types.ModuleType("main_def")
main_def_mock.MAIN_DIR = 'dummy'
main_def_mock.N_DIR = 'dummy'
main_def_mock.credentials = 'dummy'
main_def_mock.tokens = 'dummy'
main_def_mock.FIX_SLIDE_ID = 'dummy'
main_def_mock.new_slide = 'dummy'
main_def_mock.target_folder_id = 'dummy'
main_def_mock.get_url = 'dummy'
main_def_mock.slide_image_path = 'dummy.csv'
sys.modules['main_def'] = main_def_mock

import create_slide_image as csi

csi.HttpError = MockHttpError

csi.credentials = 'dummy'
csi.tokens = 'dummy'
csi.FIX_SLIDE_ID = 'dummy'
csi.new_slide = 'dummy'
csi.target_folder_id = 'dummy'
csi.get_url = 'dummy'
csi.slide_image_path = 'dummy.csv'
csi.os.path.exists = MagicMock(return_value=False)
csi.pickle = MagicMock()

class MockFlow:
    @classmethod
    def from_client_secrets_file(cls, *args, **kwargs):
        flow = MagicMock()
        flow.run_local_server.return_value = MagicMock()
        return flow
sys.modules['google_auth_oauthlib.flow'].InstalledAppFlow = MockFlow

# Mock builtins open to avoid FileNotFoundError
import builtins
real_open = builtins.open
def mock_open(*args, **kwargs):
    return MagicMock()

# Mock Google API responses
mock_service = MagicMock()
mock_presentations = MagicMock()
mock_service.presentations.return_value = mock_presentations
mock_presentations.get.return_value.execute.return_value = {
    "slides": [{"objectId": f"slide_{i}"} for i in range(25)] # we need at least 22 slides because it slices [21:]
}
mock_presentations.batchUpdate.return_value.execute.return_value = {
    "replies": [
        {"duplicateObject": {"objectId": "new_slide_123"}},
        {"createImage": {"objectId": "img_123"}},
        {"createShape": {"objectId": "shape_123"}}
    ]
}

# Mock drive files
mock_drive_service = MagicMock()
mock_drive_service.files.return_value.list.return_value.execute.return_value = {
    "files": [{"id": f"file_{i}", "name": f"{i}.png"} for i in range(5)]
}
mock_drive_service.files.return_value.copy.return_value.execute.return_value = {"id": "new_presentation"}

def mock_build(service_name, version, credentials=None):
    if service_name == 'slides':
        return mock_service
    elif service_name == 'drive':
        return mock_drive_service

csi.build = mock_build

# Overwrite time.sleep so we can track wait time without waiting
class MockSleep:
    total_sleep = 0
    def sleep(self, amount):
        self.total_sleep += amount

mock_sleeper = MockSleep()
csi.time.sleep = mock_sleeper.sleep

start_time = time.time()
with patch('builtins.open', mock_open):
    csi.create_image("dummy_presentation", "dummy_new_slide", "dummy_folder")
end_time = time.time()

print(f"Execution time (real): {end_time - start_time:.4f} seconds")
print(f"Execution time (simulated sleep): {mock_sleeper.total_sleep} seconds")
print(f"Total time (real + sleep): {(end_time - start_time) + mock_sleeper.total_sleep:.4f} seconds")
print(f"batchUpdate call count (images & text inside loop): {mock_presentations.batchUpdate.call_count}")
