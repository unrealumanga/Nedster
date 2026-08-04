import unittest

from ironclad.models import parameter_count


class ModelParsingTests(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(parameter_count("qwen3.5:9b-q4_K_M"), 9.0)
        self.assertEqual(parameter_count("nemotron-3-nano-30b-a3b"), 3.0)
        self.assertEqual(parameter_count("model-without-size"), 0.0)


if __name__ == "__main__":
    unittest.main()
