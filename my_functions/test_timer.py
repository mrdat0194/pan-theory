import os
import time
import pytest
from unittest.mock import patch
from my_functions.timer import timer, print_param

def test_timer_stdout(capsys):
    with patch('time.time') as mock_time:
        # mock_time.side_effect = [100.0, 101.5]
        # Using a simpler way to control time
        mock_time.return_value = 100.0

        @timer
        def mock_func():
            mock_time.return_value = 101.5
            return "done"

        result = mock_func()

        assert result == "done"
        captured = capsys.readouterr()
        assert "--- 1.5 seconds ---" in captured.out

def test_timer_arguments():
    @timer
    def add(a, b, c=0):
        return a + b + c

    assert add(1, 2) == 3
    assert add(1, 2, c=3) == 6

def test_print_param_basic(tmp_path, capsys):
    test_file = "test_output.txt"
    base_dir = str(tmp_path)

    @print_param(name_output=test_file, BASE_DIR=base_dir)
    def greet(name):
        return f"Hello {name}"

    result = greet("World")
    assert result == "Hello World"

    # Verify file content
    output_path = tmp_path / test_file
    assert output_path.exists()
    assert output_path.read_text() == "Hello World"

    # Verify stdout (it prints myfile.readlines())
    captured = capsys.readouterr()
    assert "['Hello World']" in captured.out

def test_print_param_generator(tmp_path, capsys):
    test_file = "gen_output.txt"
    base_dir = str(tmp_path)

    @print_param(name_output=test_file, BASE_DIR=base_dir)
    def my_gen():
        yield 1
        yield 2
        yield 3

    result = my_gen()
    # Note: result is a generator, but print_param exhausts it
    # and writes to file.

    # Verify file content
    output_path = tmp_path / test_file
    assert output_path.exists()
    # It joins with \n and adds a trailing \n for generators
    assert output_path.read_text() == "1\n2\n3\n"

    # Verify stdout
    captured = capsys.readouterr()
    # myfile.readlines() will return ['1\n', '2\n', '3\n']
    assert "['1\\n', '2\\n', '3\\n']" in captured.out

def test_print_param_existing_file(tmp_path):
    test_file = "existing.txt"
    base_dir = str(tmp_path)
    output_path = tmp_path / test_file
    output_path.write_text("initial")

    @print_param(name_output=test_file, BASE_DIR=base_dir)
    def overwrite():
        return "new"

    overwrite()
    # print_param opens with "r+" if exists, but then writes.
    # r+ doesn't truncate.
    # "new" is 3 chars, "initial" is 7 chars.
    # it writes "new" at the beginning, so it might become "newitial"
    # Let's see what it actually does.
    # f.write(str(result))

    # Actually, it's safer to check what the code does.
    # if os.path.exists(output): f = open(output, "r+")
    # else: f = open(output, "w")
    # f.write(str(result))

    # In my test above, "new" will overwrite "ini" of "initial" -> "newtial"
    # This might be a bug in print_param if truncation was intended.
    # But for now, I'm testing existing behavior.
    assert output_path.read_text() == "newtial"
