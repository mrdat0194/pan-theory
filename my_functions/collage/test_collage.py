import pytest
from my_functions.collage.collage import matchQuality

def test_matchQuality_exact_match_after_quantization():
    match = (255, 255, 255)
    target = (248, 248, 248)
    assert matchQuality(match, target) == 0

def test_matchQuality_max_difference():
    match = (0, 0, 0)
    target = (255, 255, 255)
    assert matchQuality(match, target) == 765

def test_matchQuality_partial_match():
    # 100 >> 3 = 12, 12 << 3 = 96. target=100. diff = 4
    # 150 >> 3 = 18, 18 << 3 = 144. target=150. diff = 6
    # 195 >> 3 = 24, 24 << 3 = 192. target=195. diff = 3
    match = (100, 150, 195)
    target = (100, 150, 195)
    # sum of differences = 4 + 6 + 3 = 13
    assert matchQuality(match, target) == 13

def test_matchQuality_with_negative_differences():
    # match quantized to (248, 248, 248)
    # target is (200, 200, 200)
    # dR = 200 - 248 = -48, abs(-48) = 48
    # total diff = 48 * 3 = 144
    match = (255, 255, 255)
    target = (200, 200, 200)
    assert matchQuality(match, target) == 144
