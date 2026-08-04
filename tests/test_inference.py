import unittest

from ironclad.inference import llama_cpp_plan
from ironclad.types import HardwareProfile


class LlamaCppPlanTests(unittest.TestCase):
    def setUp(self):
        self.profile = HardwareProfile("test", (1.5, 3), (7, 9), 2, 32768, "Q4", "q4_0")

    def test_command_is_loopback_only_and_quantizes_kv(self):
        plan = llama_cpp_plan(self.profile, "worker.gguf")
        command = list(plan.command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--cache-type-k") + 1], "q4_0")
        self.assertEqual(command[command.index("--spec-type") + 1], "ngram-mod")

    def test_draft_model_enables_draft_speculation(self):
        plan = llama_cpp_plan(self.profile, "worker.gguf", draft_model="router.gguf")
        self.assertEqual(plan.speculative_mode, "draft-model")
        self.assertIn("--model-draft", plan.command)

    def test_rejects_invalid_port(self):
        with self.assertRaises(ValueError):
            llama_cpp_plan(self.profile, "worker.gguf", port=70000)


if __name__ == "__main__":
    unittest.main()
