import unittest

from utils import clamp_percent


class T(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp_percent(-5), 0)
        self.assertEqual(clamp_percent(50), 50)
        self.assertEqual(clamp_percent(100), 100)
        self.assertEqual(clamp_percent(150), 100)


if __name__ == "__main__":
    unittest.main()
