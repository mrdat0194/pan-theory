import unittest
from PIL import Image
from my_functions.collage.collage import getAvgColor

class TestCollage(unittest.TestCase):

    def test_getAvgColor_solid_color(self):
        # Create a 10x10 solid red image
        im = Image.new("RGB", (10, 10), (255, 0, 0))
        avg_color = getAvgColor(im)
        self.assertEqual(avg_color, (255.0, 0.0, 0.0))

    def test_getAvgColor_1x1(self):
        # Create a 1x1 blue image
        im = Image.new("RGB", (1, 1), (0, 0, 255))
        avg_color = getAvgColor(im)
        self.assertEqual(avg_color, (0.0, 0.0, 255.0))

    def test_getAvgColor_mixed_pixels(self):
        # Create a 2x1 image with two different pixels
        im = Image.new("RGB", (2, 1))
        im.putpixel((0, 0), (100, 50, 200))
        im.putpixel((1, 0), (200, 150, 100))
        # Average: ((100+200)/2, (50+150)/2, (200+100)/2) = (150, 100, 150)
        avg_color = getAvgColor(im)
        self.assertEqual(avg_color, (150.0, 100.0, 150.0))

    def test_getAvgColor_all_black(self):
        im = Image.new("RGB", (5, 5), (0, 0, 0))
        avg_color = getAvgColor(im)
        self.assertEqual(avg_color, (0.0, 0.0, 0.0))

    def test_getAvgColor_all_white(self):
        im = Image.new("RGB", (5, 5), (255, 255, 255))
        avg_color = getAvgColor(im)
        self.assertEqual(avg_color, (255.0, 255.0, 255.0))

if __name__ == '__main__':
    unittest.main()
