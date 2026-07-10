import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
from my_functions.Music.Mp4_to_Wav import convert_mp4_to_wav, is_safe_filename
from my_functions.Music.Avi_to_mp4 import convert_avi_to_mp4

class TestSecuritySubprocess(unittest.TestCase):

    def test_is_safe_filename(self):
        self.assertTrue(is_safe_filename("normal.mp4"))
        self.assertTrue(is_safe_filename("file with spaces.mp4"))
        self.assertFalse(is_safe_filename("-option.mp4"))
        self.assertFalse(is_safe_filename("protocol:injection.mp4"))

    @patch('subprocess.run')
    @patch('os.listdir')
    @patch('pathlib.Path.exists')
    def test_convert_mp4_to_wav_security(self, mock_exists, mock_listdir, mock_run):
        mock_exists.return_value = True
        mock_listdir.return_value = [
            'normal.mp4',
            '-option.mp4',
            'file with spaces.mp4',
            'protocol:injection.mp4',
            'semi;colon.mp4',
            'back`tick.mp4'
        ]
        mock_run.return_value = MagicMock(returncode=0)

        input_dir = Path('/tmp/Desktop')
        output_dir = Path('/tmp/Desktop')

        convert_mp4_to_wav(input_dir, output_dir)

        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 4, f"Expected 4 calls, got {len(calls)}")

        expected_filenames = ['normal.mp4', 'file with spaces.mp4', 'semi;colon.mp4', 'back`tick.mp4']

        for call in calls:
            args = call[0][0]
            self.assertEqual(args[0], 'ffmpeg')
            self.assertEqual(args[1], '-i')
            input_path = args[2]
            self.assertTrue(any(fname in input_path for fname in expected_filenames))

    @patch('subprocess.run')
    @patch('os.listdir')
    @patch('pathlib.Path.exists')
    def test_convert_avi_to_mp4_security(self, mock_exists, mock_listdir, mock_run):
        mock_exists.return_value = True
        mock_listdir.return_value = ['dangerous -option.avi', 'safe.avi']
        mock_run.return_value = MagicMock(returncode=0)

        input_dir = Path('/tmp/Desktop')
        output_dir = Path('/tmp/Desktop')

        convert_avi_to_mp4(input_dir, output_dir)

        calls = mock_run.call_args_list
        self.assertEqual(len(calls), 2)

        args = calls[0][0][0]
        self.assertEqual(args[0], 'ffmpeg')
        self.assertEqual(args[2], '/tmp/Desktop/dangerous -option.avi')

if __name__ == '__main__':
    unittest.main()
