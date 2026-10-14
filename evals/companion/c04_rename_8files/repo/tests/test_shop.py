import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shop import inventory, pricing  # noqa: E402
from shop.importer import import_rows  # noqa: E402
from shop.skus import normalize_sku  # noqa: E402


class SkuTests(unittest.TestCase):
    def setUp(self):
        inventory.reset()

    def test_normalizes(self):
        self.assertEqual(normalize_sku(" ab_123 "), "AB-123")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            normalize_sku("nope")

    def test_import_and_total(self):
        order, rejected = import_rows(["ab_123,100", "bad,1", "nut 2001,10"])
        self.assertEqual(rejected, ["bad,1"])
        self.assertEqual(order.total(), Decimal("23.75") + Decimal("1.00"))

    def test_price(self):
        self.assertEqual(pricing.unit_price("wsh-310"), Decimal("0.05"))


if __name__ == "__main__":
    unittest.main()
