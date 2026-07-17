import pytest
from my_functions.collage.collage import quantize

def test_quantize_basic():
    assert quantize((255, 255, 255), 3) == (248, 248, 248)

def test_quantize_zero_shift():
    assert quantize((100, 150, 200), 0) == (100, 150, 200)

def test_quantize_large_shift():
    assert quantize((255, 255, 255), 8) == (0, 0, 0)

def test_quantize_type_coercion():
    assert quantize(("100", 150.5, "200"), 4) == (96, 144, 192)

def test_quantize_negative_shift():
    with pytest.raises(ValueError):
        quantize((255, 255, 255), -1)

def test_quantize_invalid_input_length():
    with pytest.raises(IndexError):
        quantize((255, 255), 3)

def test_quantize_invalid_type():
    with pytest.raises(ValueError):
        quantize(("abc", 255, 255), 3)
