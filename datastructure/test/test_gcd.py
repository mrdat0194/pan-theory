import unittest
import sys
import os

# Add the directory containing GCD.py to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Data_Structures_Algorithms_In_Python-master', 'Mathematics'))

from GCD import gcd

class TestGCD(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(gcd(21, 6), 3)
        self.assertEqual(gcd(6, 21), 3)
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(18, 48), 6)

    def test_prime_numbers(self):
        self.assertEqual(gcd(13, 17), 1)
        self.assertEqual(gcd(17, 13), 1)

    def test_with_zero(self):
        self.assertEqual(gcd(0, 5), 5)
        self.assertEqual(gcd(5, 0), 5)
        self.assertEqual(gcd(0, 0), 0)

if __name__ == '__main__':
    unittest.main()
