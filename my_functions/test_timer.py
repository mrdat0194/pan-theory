import time
from unittest.mock import patch
import pytest
from my_functions.timer import timer

def test_timer_returns_result():
    @timer
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

def test_timer_prints_time(capsys):
    @timer
    def some_function():
        return "done"

    with patch('time.time') as mock_time:
        mock_time.side_effect = [100.0, 100.5]
        result = some_function()
        assert result == "done"

    captured = capsys.readouterr()
    assert "--- 0.5 seconds ---" in captured.out

def test_timer_with_args_and_kwargs():
    @timer
    def complex_func(*args, **kwargs):
        return len(args) + len(kwargs)

    assert complex_func(1, 2, a=3, b=4) == 4

def test_timer_metadata():
    @timer
    def sample_function():
        """This is a docstring."""
        pass

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "This is a docstring."
