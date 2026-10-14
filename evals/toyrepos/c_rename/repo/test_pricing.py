import unittest

from cart import checkout
from pricing import compute_total
from report import summary


class Tests(unittest.TestCase):
    def test_all(self):
        self.assertEqual(compute_total([1, 2]), 3)
        self.assertEqual(checkout([{"price": 10}]), 11.0)
        self.assertEqual(summary([2, 2]), "Total: 4")


if __name__ == "__main__":
    unittest.main()
