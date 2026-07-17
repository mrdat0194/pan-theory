import os
import types
from unittest.mock import patch
from my_functions.timer import print_param, timer

def test_print_param_new_file(tmp_path, capsys):
    output_name = "test_output.txt"
    base_dir = str(tmp_path)

    @print_param(name_output=output_name, BASE_DIR=base_dir)
    def my_func(a, b):
        """This is my_func docstring."""
        return a + b

    result = my_func(2, 3)

    assert result == 5
    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "This is my_func docstring."

    output_path = os.path.join(base_dir, output_name)
    assert os.path.exists(output_path)

    with open(output_path, "r") as f:
        content = f.read()

    assert content == "5"

    captured = capsys.readouterr()
    assert "['5']" in captured.out

def test_print_param_existing_file(tmp_path, capsys):
    output_name = "test_output_existing.txt"
    base_dir = str(tmp_path)
    output_path = os.path.join(base_dir, output_name)

    # Create existing file
    with open(output_path, "w") as f:
        f.write("old data")

    @print_param(name_output=output_name, BASE_DIR=base_dir)
    def my_func():
        return "new data"

    my_func()

    with open(output_path, "r") as f:
        content = f.read()

    # r+ overwrites from beginning without truncating
    assert content == "new data"

    captured = capsys.readouterr()
    assert "['new data']" in captured.out

def test_print_param_generator(tmp_path, capsys):
    output_name = "test_gen.txt"
    base_dir = str(tmp_path)

    @print_param(name_output=output_name, BASE_DIR=base_dir)
    def my_gen():
        yield 1
        yield 2
        yield 3

    result = my_gen()

    assert isinstance(result, types.GeneratorType)

    output_path = os.path.join(base_dir, output_name)
    with open(output_path, "r") as f:
        content = f.read()

    assert content == "1\n2\n3\n"

    captured = capsys.readouterr()
    assert "['1\\n', '2\\n', '3\\n']" in captured.out

@patch("time.time")
def test_timer(mock_time, capsys):
    mock_time.side_effect = [10.0, 12.0]

    @timer
    def slow_func():
        return 42

    result = slow_func()

    assert result == 42

    captured = capsys.readouterr()
    assert "--- 2.0 seconds ---" in captured.out
