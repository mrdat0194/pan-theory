import unittest
from unittest.mock import patch
import pytest

from my_functions.timer import timer

def test_timer_preserves_metadata():
    @timer
    def dummy_func():
        """This is a dummy docstring."""
        pass

    assert dummy_func.__name__ == 'dummy_func'
    assert dummy_func.__doc__ == 'This is a dummy docstring.'

def test_timer_output_and_return_value(capsys):
    @timer
    def add(a, b):
        return a + b

    with patch('time.time') as mock_time:
        mock_time.side_effect = [10.0, 15.0]
        result = add(2, 3)

    assert result == 5
    captured = capsys.readouterr()
    assert "--- 5.0 seconds ---" in captured.out
