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