import pytest
from my_functions.collage.collage import matchQuality

def test_match_quality_success():
    # match = (10, 20, 30), quantized (8, 16, 24)
    # target = (10, 20, 30)
    # dR = 10 - 8 = 2
    # dG = 20 - 16 = 4
    # dB = 30 - 24 = 6
    # return 2 + 4 + 6 = 12
    assert matchQuality((10, 20, 30), (10, 20, 30)) == 12

def test_match_quality_invalid_match_type():
    with pytest.raises(TypeError):
        matchQuality(None, (255, 255, 255))

def test_match_quality_invalid_target_type():
    with pytest.raises(TypeError):
        matchQuality((255, 255, 255), None)

def test_match_quality_short_match():
    with pytest.raises(IndexError):
        matchQuality((255, 255), (255, 255, 255))

def test_match_quality_short_target():
    with pytest.raises(IndexError):
        matchQuality((255, 255, 255), (255, 255))

def test_match_quality_non_numeric_match():
    # quantize will fail on int('a')
    with pytest.raises(ValueError):
        matchQuality(('a', 'b', 'c'), (255, 255, 255))
