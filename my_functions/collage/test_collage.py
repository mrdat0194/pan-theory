import pytest
from PIL import Image
from my_functions.collage.collage import getFrequent, quantize

def test_getFrequent_majority_color():
    im = Image.new("RGB", (3, 3), "white")
    # Set most to white, one to black
    im.putpixel((0, 0), (0, 0, 0))
    # 255 quantized by 3 (>> 3 << 3)
    # 255 >> 3 = 31, 31 << 3 = 248
    expected = (248, 248, 248)
    assert getFrequent(im) == expected

def test_getFrequent_identical_colors():
    im = Image.new("RGB", (2, 2), "black")
    expected = (0, 0, 0)
    assert getFrequent(im) == expected

def test_getFrequent_multiple_colors():
    im = Image.new("RGB", (4, 4), "blue") # blue is (0, 0, 255) -> quantized to (0, 0, 248)
    # Add multiple reds
    for i in range(3):
        im.putpixel((i, 0), (255, 0, 0)) # red -> quantized to (248, 0, 0)
    # Add multiple greens, making it the majority
    for i in range(4):
        im.putpixel((i, 1), (0, 255, 0)) # green -> quantized to (0, 248, 0)
    for i in range(2):
        im.putpixel((i, 2), (0, 255, 0))

    # Total: 6 green, 3 red, 7 blue. Blue is still majority!
    # Wait, 16 total pixels.
    # Row 0: 3 red, 1 blue
    # Row 1: 4 green
    # Row 2: 2 green, 2 blue
    # Row 3: 4 blue
    # Total: 3 red, 6 green, 7 blue. Majority is blue.
    expected_blue = (0, 0, 248)
    assert getFrequent(im) == expected_blue

    # Add more greens to make it majority
    im.putpixel((3, 2), (0, 255, 0))
    im.putpixel((0, 3), (0, 255, 0))
    im.putpixel((1, 3), (0, 255, 0))
    # Total: 3 red, 9 green, 4 blue. Majority is green.
    expected_green = (0, 248, 0)
    assert getFrequent(im) == expected_green

def test_getFrequent_empty_image():
    im = Image.new("RGB", (0, 0), "white")
    with pytest.raises(ValueError):
        getFrequent(im)
