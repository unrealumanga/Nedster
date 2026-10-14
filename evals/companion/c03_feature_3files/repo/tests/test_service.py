import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import TaskService  # noqa: E402


class TaskServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tasks.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_assigns_increasing_ids(self):
        svc = TaskService(self.path)
        self.assertEqual(svc.add("a").id, 1)
        self.assertEqual(svc.add("b").id, 2)

    def test_complete_marks_done(self):
        svc = TaskService(self.path)
        task = svc.add("a")
        svc.complete(task.id)
        self.assertTrue(TaskService(self.path).list()[0].done)

    def test_empty_title_rejected(self):
        with self.assertRaises(ValueError):
            TaskService(self.path).add("   ")


if __name__ == "__main__":
    unittest.main()
