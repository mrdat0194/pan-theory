import pytest
from unittest.mock import patch
from my_functions.timer import add, timer

def test_add_basic():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-2, -3) == -5

def test_add_floats():
    assert add(2.5, 3.1) == 5.6

def test_add_strings():
    assert add("a", "b") == "ab"

@patch('time.time', side_effect=[100.0, 105.5])
def test_add_timing_output(mock_time, capsys):
    result = add(10, 20)
    assert result == 30
    captured = capsys.readouterr()
    assert "--- 5.5 seconds ---" in captured.out

def test_timer_preserves_metadata():
    assert add.__name__ == 'add'
