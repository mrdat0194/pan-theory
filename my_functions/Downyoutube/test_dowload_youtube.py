import unittest  # noqa: E402
from unittest.mock import patch, MagicMock  # noqa: E402
import sys  # noqa: E402
import os  # noqa: E402

# Add the directory containing the module to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We need to mock youtube_dl BEFORE importing download_youtube if it's imported at top level
# But in our case it's imported at top level in dowload_youtube.py
# So we mock it in sys.modules

mock_youtube_dl = MagicMock()
sys.modules['youtube_dl'] = mock_youtube_dl

from dowload_youtube import download_youtube  # noqa: E402

class TestDownloadYoutube(unittest.TestCase):

    @patch('argparse.ArgumentParser.parse_args')
    @patch('os.chdir')
    def test_download_youtube(self, mock_chdir, mock_parse_args):
        # Setup mock arguments
        mock_args = MagicMock()
        mock_args.link = 'https://www.youtube.com/watch?v=test'
        mock_parse_args.return_value = mock_args

        # Setup mock YoutubeDL context manager
        mock_ydl_instance = mock_youtube_dl.YoutubeDL.return_value.__enter__.return_value

        # Call the function
        download_youtube()

        # Verify YoutubeDL was instantiated with empty opts
        mock_youtube_dl.YoutubeDL.assert_called_once_with({})

        # Verify download was called with the correct link
        mock_ydl_instance.download.assert_called_once_with(['https://www.youtube.com/watch?v=test'])

        # Verify os.chdir('.') was called
        mock_chdir.assert_called_once_with('.')

if __name__ == '__main__':
    unittest.main()
