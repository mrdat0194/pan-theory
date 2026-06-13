import traceback

def func():
    print("this is a func")

if __name__ == "__main__":
    try:
        name = "joy"
        a = [1, 2, 0, 3]
        for i in a:
            func()
            z = i + 5
            k = z * 10
            print(i)
            print(1 / i)
        x = 1 / 0
        # connect db die
        with open("joy.txt", "r") as f:
            print(f.readline())
    except ZeroDivisionError:
        print("Divide by 0")
    except FileNotFoundError:
        print("File not found")
    except:  # noqa
        print("Unknown error")
        print(traceback.format_exc())

####
def elements_check(array):
    try:
        first = next(array)
    except StopIteration:
        return True
    return all(first == x for x in array)

if __name__ == '__main__':
    array1 = [5, 5, 5, 5, 5]
    array2 = ["a", "a", "a", "a", "a"]
    array_iter = iter(array2)
    print(elements_check(array_iter))