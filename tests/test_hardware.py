import unittest

from ironclad.hardware import choose_profile
from ironclad.types import HardwareSnapshot


def snapshot(vram: float, ram: float = 32, unified: bool = False) -> HardwareSnapshot:
    return HardwareSnapshot("test", 8, ram, ram, "gpu", vram, vram, unified)


class HardwareProfileTests(unittest.TestCase):
    def test_discrete_vram_tiers(self):
        self.assertEqual(choose_profile(snapshot(8)).name, "edge-8g")
        self.assertEqual(choose_profile(snapshot(16)).name, "forge-16g")
        self.assertEqual(choose_profile(snapshot(32)).name, "titan-32g")

    def test_unified_memory_uses_safe_fraction(self):
        self.assertEqual(choose_profile(snapshot(0, ram=48, unified=True)).name, "titan-32g")


if __name__ == "__main__":
    unittest.main()
