import unittest
from unittest.mock import patch, MagicMock
import sys
import os

from my_functions.collage.collage import main

class TestCollageMain(unittest.TestCase):

    @patch('my_functions.collage.collage.sys.exit')
    @patch('builtins.print')
    def test_main_help(self, mock_print, mock_exit):
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            main(['-h'])
        mock_print.assert_any_call('collage.py -i <inputfile> -d <directory> -o <outputfile> -t <tilesize> -n <numtiles> -b <blend_amt> -B <brightness> -C <contrast>')

    @patch('my_functions.collage.collage.sys.exit')
    @patch('builtins.print')
    def test_main_invalid_option(self, mock_print, mock_exit):
        mock_exit.side_effect = SystemExit
        with self.assertRaises(SystemExit):
            main(['--invalid'])
        mock_print.assert_any_call('collage.py -i <inputfile> -d <directory> -o <outputfile> -t <tilesize> -n <numtiles> -b <blend_amt> -B <brightness> -C <contrast>')

    @patch('my_functions.collage.collage.Image')
    @patch('my_functions.collage.collage.os')
    @patch('my_functions.collage.collage.glob')
    @patch('builtins.print')
    def test_main_no_thumbnails(self, mock_print, mock_glob, mock_os, mock_image):
        # Setup mocks
        mock_os.path.dirname.return_value = '/test/dir'
        mock_os.path.abspath.return_value = '/test/dir/input.jpg'
        mock_os.path.join.return_value = '/test/dir/source'
        mock_os.path.exists.return_value = True
        mock_glob.glob.return_value = [] # no thumbnails

        main(['-i', 'input.jpg'])

        mock_print.assert_any_call('There are no thumbnails in this directory!\n')

    @patch('my_functions.collage.collage.Image.blend')
    @patch('my_functions.collage.collage.Image.new')
    @patch('my_functions.collage.collage.ImageEnhance')
    @patch('my_functions.collage.collage.Image.open')
    @patch('my_functions.collage.collage.os')
    @patch('my_functions.collage.collage.glob')
    @patch('builtins.print')
    def test_main_success(self, mock_print, mock_glob, mock_os, mock_image_open, mock_image_enhance, mock_image_new, mock_image_blend):
        # Setup mocks
        mock_os.path.dirname.return_value = '/test/dir'
        mock_os.path.abspath.return_value = '/test/dir/input.jpg'

        def side_effect_join(a, b):
            return f"{a}/{b}"
        mock_os.path.join.side_effect = side_effect_join
        mock_os.path.exists.return_value = True
        mock_os.path.isfile.return_value = False
        mock_os.path.splitext.return_value = ('test_file', '.jpg')

        def mock_glob_side_effect(path):
            if path.endswith("/*"):
                return ['/test/dir/source/test_file.jpg']
            elif path.endswith("source/*_thumb.jpg"):
                return ['/test/dir/source/test_file_thumb.jpg']
            return []

        mock_glob.glob.side_effect = mock_glob_side_effect

        # Mock image
        mock_img_instance = MagicMock()
        mock_img_instance.size = [100, 100]  # Return a list with 2 elements for unpacking
        mock_img_instance.getpixel.return_value = (10, 20, 30)
        mock_img_instance.copy.return_value = mock_img_instance
        mock_image_open.return_value = mock_img_instance

        mock_enhanced_img = MagicMock()
        mock_enhanced_img.size = [100, 100]
        mock_enhanced_img.getpixel.return_value = (10, 20, 30)
        mock_enhanced_img.copy.return_value = mock_enhanced_img

        mock_enhancer_instance = MagicMock()
        mock_enhancer_instance.enhance.return_value = mock_enhanced_img
        mock_image_enhance.Brightness.return_value = mock_enhancer_instance
        mock_image_enhance.Contrast.return_value = mock_enhancer_instance

        mock_final_img = MagicMock()
        mock_final_img.size = (100, 100)
        mock_image_new.return_value = mock_final_img

        mock_blended_img = MagicMock()
        mock_image_blend.return_value = mock_blended_img

        main(['-i', 'input.jpg', '-B', '1.5', '-C', '1.5'])

        # Assertions
        mock_print.assert_any_call("Making thumbnails...")
        mock_print.assert_any_call("Collecting data...")
        mock_print.assert_any_call("Resizing target...")
        mock_print.assert_any_call("Making collage...")
        mock_print.assert_any_call("Saving...")
        mock_blended_img.save.assert_called_once_with('outfile.jpeg', 'JPEG')

if __name__ == '__main__':
    unittest.main()
