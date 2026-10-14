import unittest

from rng import sum_to


class SumTests(unittest.TestCase):
    def test_sum_to_five(self):
        self.assertEqual(sum_to(5), 15)


if __name__ == "__main__":
    unittest.main()
