def cubes_of_multiples_of_5(start, end):
  """
  Finds the cubes of integers between start and end that are multiples of 5.

  Args:
    start: The starting integer.
    end: The ending integer.

  Returns:
    A list of cubes of integers that are multiples of 5.
  """

  return list(map(lambda x: x**3, filter(lambda x: x % 5 == 0, range(start, end + 1))))

# Example usage:
result = cubes_of_multiples_of_5(1, 100)
print(result)