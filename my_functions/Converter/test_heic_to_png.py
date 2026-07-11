import unittest
from unittest.mock import patch, call
import os
from my_functions.Converter.heic_to_png import delete_all_heic_files

class TestDeleteHeicFiles(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.remove')
    def test_delete_all_heic_files_success(self, mock_remove, mock_listdir):
        """Test that only .heic files are deleted, case-insensitively."""
        mock_listdir.return_value = ['image1.HEIC', 'image2.heic', 'document.pdf', 'photo.jpg']
        folder = '/mock/folder'

        delete_all_heic_files(folder)

        # Check if os.remove was called for .heic files
        expected_calls = [
            call(os.path.join(folder, 'image1.HEIC')),
            call(os.path.join(folder, 'image2.heic'))
        ]
        mock_remove.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_remove.call_count, 2)

    @patch('os.listdir')
    @patch('os.remove')
    def test_delete_all_heic_files_empty_folder(self, mock_remove, mock_listdir):
        """Test that it handles an empty folder gracefully."""
        mock_listdir.return_value = []
        folder = '/mock/folder'

        delete_all_heic_files(folder)

        mock_remove.assert_not_called()

    @patch('os.listdir')
    @patch('os.remove')
    def test_delete_all_heic_files_no_heic(self, mock_remove, mock_listdir):
        """Test that it does nothing when no .heic files are present."""
        mock_listdir.return_value = ['image1.png', 'image2.jpg', 'notes.txt']
        folder = '/mock/folder'

        delete_all_heic_files(folder)

        mock_remove.assert_not_called()

    @patch('os.listdir')
    @patch('os.remove')
    @patch('builtins.print')
    def test_delete_all_heic_files_exception(self, mock_print, mock_remove, mock_listdir):
        """Test that it handles exceptions during os.remove and continues."""
        mock_listdir.return_value = ['fail.heic', 'success.heic']
        folder = '/mock/folder'

        # Mock os.remove to raise an exception for the first file
        mock_remove.side_effect = [Exception("Delete failed"), None]

        delete_all_heic_files(folder)

        self.assertEqual(mock_remove.call_count, 2)
        # Verify that error message was printed
        mock_print.assert_any_call("Failed to delete fail.heic: Delete failed")
        mock_print.assert_any_call("Deleted: success.heic")

if __name__ == '__main__':
    unittest.main()
