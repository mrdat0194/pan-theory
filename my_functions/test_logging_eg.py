from my_functions.logging_eg import add, subtract, multiply, divide

def test_add():
    assert add(10, 5) == 15
    assert add(-1, 1) == 0
    assert add(-1, -1) == -2
    assert add(0, 0) == 0

def test_subtract():
    assert subtract(10, 5) == 5
    assert subtract(-1, 1) == -2
    assert subtract(-1, -1) == 0
    assert subtract(0, 0) == 0

def test_multiply():
    assert multiply(10, 5) == 50
    assert multiply(-1, 1) == -1
    assert multiply(-1, -1) == 1
    assert multiply(10, 0) == 0

def test_divide():
    assert divide(10, 5) == 2
    assert divide(-1, 1) == -1
    assert divide(-1, -1) == 1
    assert divide(5, 2) == 2.5
    assert divide(0, 10) == 0

def test_divide_by_zero(caplog):
    assert divide(10, 0) is None
    assert 'Tried to divide by zero' in caplog.text
