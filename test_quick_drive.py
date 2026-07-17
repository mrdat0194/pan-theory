import unittest
from unittest.mock import patch, MagicMock, mock_open
from main_def.ggl_api.google_spreadsheet_api.quick_drive import main

class TestQuickDrive(unittest.TestCase):
    @patch('main_def.ggl_api.google_spreadsheet_api.quick_drive.os.path.exists')
    @patch('main_def.ggl_api.google_spreadsheet_api.quick_drive.Credentials.from_authorized_user_file')
    @patch('main_def.ggl_api.google_spreadsheet_api.quick_drive.build')
    def test_main_with_valid_creds(self, mock_build, mock_load, mock_exists):
        mock_exists.return_value = True
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_load.return_value = mock_creds

        mock_service = MagicMock()
        mock_build.return_value = mock_service

        mock_service.files().list().execute.return_value = {'files': [{'name': 'test', 'id': '123'}]}

        with patch('builtins.open', mock_open(read_data=b'data')):
            main()

        mock_build.assert_called_once()
        self.assertTrue(mock_service.files().list.called)

if __name__ == '__main__':
    unittest.main()
