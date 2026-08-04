import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from ironclad.ledger import EventLedger


class LedgerTests(unittest.TestCase):
    def test_chain_verifies_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            with EventLedger(path, b"k" * 32) as ledger:
                ledger.append("run.started", "test", {"objective": "ship"})
                ledger.append("run.finished", "test", {"passed": True})
                valid, count, _ = ledger.verify()
                self.assertTrue(valid)
                self.assertEqual(count, 2)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("UPDATE events SET payload = ? WHERE seq = 1", ('{"objective":"changed"}',))
                connection.commit()
            with EventLedger(path, b"k" * 32) as ledger:
                valid, _, message = ledger.verify()
                self.assertFalse(valid)
                self.assertIn("chain break", message)


if __name__ == "__main__":
    unittest.main()
