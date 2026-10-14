import unittest

from mathx import add


class T(unittest.TestCase):
    def test_negatives(self):
        self.assertEqual(add(-2, -3), -5)
        self.assertEqual(add(-2, 3), 1)


if __name__ == "__main__":
    unittest.main()
