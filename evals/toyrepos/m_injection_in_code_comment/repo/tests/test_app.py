import unittest

from app import greet


class GreetTests(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(greet("Ann"), "Hi Ann")


if __name__ == "__main__":
    unittest.main()
