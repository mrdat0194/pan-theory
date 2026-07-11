import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock cv2 because the environment may not have it installed
sys.modules['cv2'] = MagicMock()

# Ensure the root directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from my_functions.open_cv_function.openCV_simple import decode_fourcc

class TestDecodeFourcc(unittest.TestCase):
    def test_decode_fourcc_avc1(self):
        # 828601953 -> 'avc1'
        self.assertEqual(decode_fourcc(828601953), 'avc1')

    def test_decode_fourcc_mp4v(self):
        # 1983148141 -> 'mp4v'
        self.assertEqual(decode_fourcc(1983148141), 'mp4v')

    def test_decode_fourcc_av01(self):
        # 825259617 -> 'av01'
        self.assertEqual(decode_fourcc(825259617), 'av01')

    def test_decode_fourcc_string_input(self):
        # The function converts input to int, so a string should work if it contains a valid int.
        self.assertEqual(decode_fourcc("828601953"), 'avc1')

    def test_decode_fourcc_float_input(self):
        # The function converts input to int
        self.assertEqual(decode_fourcc(828601953.0), 'avc1')

    def test_decode_fourcc_invalid_input(self):
        # Should raise ValueError for string that cannot be cast to int
        with self.assertRaises(ValueError):
            decode_fourcc("invalid")

if __name__ == '__main__':
    unittest.main()
