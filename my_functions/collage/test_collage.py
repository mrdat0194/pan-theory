import unittest
from PIL import Image
import sys
import os

# Ensure the module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collage import getAvgColor

class TestCollage(unittest.TestCase):
    def test_getAvgColor_solid_color(self):
        # Create a solid color image
        img = Image.new("RGB", (10, 10), color=(100, 150, 200))

        r, g, b = getAvgColor(img)

        self.assertEqual(r, 100.0)
        self.assertEqual(g, 150.0)
        self.assertEqual(b, 200.0)

    def test_getAvgColor_mixed_colors(self):
        # Create a small image (2x2) with mixed colors
        img = Image.new("RGB", (2, 2))
        img.putpixel((0, 0), (10, 20, 30))
        img.putpixel((1, 0), (20, 30, 40))
        img.putpixel((0, 1), (30, 40, 50))
        img.putpixel((1, 1), (40, 50, 60))

        r, g, b = getAvgColor(img)

        self.assertEqual(r, 25.0)
        self.assertEqual(g, 35.0)
        self.assertEqual(b, 45.0)

    def test_getAvgColor_1x1_image(self):
        # Create a 1x1 image
        img = Image.new("RGB", (1, 1), color=(11, 22, 33))

        r, g, b = getAvgColor(img)

        self.assertEqual(r, 11.0)
        self.assertEqual(g, 22.0)
        self.assertEqual(b, 33.0)

if __name__ == '__main__':
    unittest.main()
