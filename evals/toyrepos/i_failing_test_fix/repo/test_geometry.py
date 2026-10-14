import unittest

from geometry import rectangle_area, rectangle_perimeter


class GeometryTests(unittest.TestCase):
    def test_area(self):
        self.assertEqual(rectangle_area(3, 4), 12)

    def test_perimeter(self):
        self.assertEqual(rectangle_perimeter(3, 4), 14)


if __name__ == "__main__":
    unittest.main()
