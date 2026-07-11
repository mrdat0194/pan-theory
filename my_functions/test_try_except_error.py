import pytest
from my_functions.try_except_error import elements_check

def test_elements_check_all_same():
    # Test with iterator where all elements are the same
    array_iter = iter([5, 5, 5, 5, 5])
    assert elements_check(array_iter) is True

def test_elements_check_different():
    # Test with iterator where elements are different
    array_iter = iter([5, 5, 4, 5, 5])
    assert elements_check(array_iter) is False

def test_elements_check_empty():
    # Test with empty iterator
    array_iter = iter([])
    assert elements_check(array_iter) is True

def test_elements_check_single_element():
    # Test with iterator containing a single element
    array_iter = iter([5])
    assert elements_check(array_iter) is True

def test_elements_check_strings_same():
    # Test with iterator of identical strings
    array_iter = iter(["a", "a", "a", "a", "a"])
    assert elements_check(array_iter) is True

def test_elements_check_strings_different():
    # Test with iterator of different strings
    array_iter = iter(["a", "a", "b", "a", "a"])
    assert elements_check(array_iter) is False

def test_elements_check_generator():
    # Test with a generator expression
    gen = (x for x in [2, 2, 2, 2])
    assert elements_check(gen) is True

    gen_diff = (x for x in [2, 2, 3, 2])
    assert elements_check(gen_diff) is False

def test_elements_check_list_raises_typeerror():
    # Test with a regular list, which should raise a TypeError since it's not an iterator
    with pytest.raises(TypeError):
        elements_check([1, 1, 1])

def test_elements_check_none():
    # Test with None, which should raise a TypeError since it's not an iterator
    with pytest.raises(TypeError):
        elements_check(None)

def test_elements_check_custom_iterator():
    # Test with mixed types iterator
    array_iter = iter([1, "1"])
    assert elements_check(array_iter) is False
