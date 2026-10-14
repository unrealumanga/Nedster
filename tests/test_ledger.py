import hashlib
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from ironclad.ledger import EventLedger, load_or_create_ledger_key


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


class LedgerInvariantTests(unittest.TestCase):
    def test_first_event_prev_hash_is_genesis(self):
        with tempfile.TemporaryDirectory() as directory, EventLedger(
            Path(directory) / "e.sqlite3", b"k" * 32
        ) as ledger:
            ledger.append("a", "test")
            first = next(iter(ledger.iter_events()))
            self.assertEqual(first["prev_hash"], EventLedger.GENESIS)
            self.assertEqual(len(first["prev_hash"]), 64)

    def test_append_returns_monotonic_seq_and_iter_orders_by_seq(self):
        with tempfile.TemporaryDirectory() as directory, EventLedger(
            Path(directory) / "e.sqlite3", b"k" * 32
        ) as ledger:
            seqs = [ledger.append("a", "test", {"i": i}) for i in range(5)]
            self.assertEqual(seqs, [1, 2, 3, 4, 5])
            rows = list(ledger.iter_events())
            self.assertEqual([row["seq"] for row in rows], seqs)
            self.assertEqual([row["payload"]["i"] for row in rows], list(range(5)))

    def test_verify_succeeds_with_identical_ts_ns(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            EventLedger(Path(directory) / "e.sqlite3", b"k" * 32) as ledger,
            mock.patch("ironclad.ledger.time.time_ns", return_value=42),
        ):
            for index in range(10):
                ledger.append("a", "test", {"i": index})
            valid, count, _ = ledger.verify()
            self.assertTrue(valid)
            self.assertEqual(count, 10)

    def test_payload_is_stored_in_canonical_form(self):
        with tempfile.TemporaryDirectory() as directory, EventLedger(
            Path(directory) / "e.sqlite3", b"k" * 32
        ) as ledger:
            ledger.append("a", "test", {"b": 1, "a": {"z": True, "y": None}})
            row = next(iter(ledger.iter_events()))
            self.assertEqual(row["payload"], {"a": {"y": None, "z": True}, "b": 1})
            valid, _, _ = ledger.verify()
            self.assertTrue(valid)

    def test_tampered_ts_ns_breaks_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e.sqlite3"
            with EventLedger(path, b"k" * 32) as ledger:
                ledger.append("a", "test")
                ledger.append("b", "test")
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("UPDATE events SET ts_ns = ts_ns + 1 WHERE seq = 1")
                connection.commit()
            with EventLedger(path, b"k" * 32) as ledger:
                valid, _, message = ledger.verify()
                self.assertFalse(valid)
                self.assertIn("chain break", message)


class LedgerConcurrencyTests(unittest.TestCase):
    def test_two_instances_two_threads_appends_never_fork_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            ledgers = [EventLedger(path, b"k" * 32) for _ in range(2)]
            errors: list[BaseException] = []

            def pump() -> None:
                try:
                    for ledger in ledgers:
                        for index in range(200):
                            ledger.append("concurrent", "test", {"i": index})
                except BaseException as exc:  # pragma: no cover - failure reporting
                    errors.append(exc)

            threads = [threading.Thread(target=pump) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            try:
                self.assertEqual(errors, [])
                for ledger in ledgers:
                    valid, count, _ = ledger.verify()
                    self.assertTrue(valid)
                    self.assertEqual(count, 800)
                seqs = [row["seq"] for row in ledgers[0].iter_events()]
                self.assertEqual(seqs, list(range(1, 801)))
            finally:
                for ledger in ledgers:
                    ledger.close()

    def test_key_race_threads_share_one_key(self):
        with tempfile.TemporaryDirectory() as directory:
            results: list[bytes] = []
            barrier = threading.Barrier(8)

            def load() -> None:
                barrier.wait()
                results.append(load_or_create_ledger_key(directory))

            threads = [threading.Thread(target=load) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(results), 8)
            self.assertEqual(len(set(results)), 1)
            key_path = Path(directory) / "ledger.key"
            self.assertTrue(key_path.exists())
            self.assertEqual(key_path.read_bytes(), results[0])
            self.assertEqual(len(results[0]), 32)

    @unittest.skipIf(sys.platform == "win32", "POSIX file mode semantics")
    def test_key_file_created_private(self):
        with tempfile.TemporaryDirectory() as directory:
            load_or_create_ledger_key(directory)
            mode = (Path(directory) / "ledger.key").stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    @unittest.skipIf(sys.platform == "win32", "POSIX file mode semantics")
    def test_existing_world_readable_key_is_tightened(self):
        with tempfile.TemporaryDirectory() as directory:
            key_path = Path(directory) / "ledger.key"
            key_path.write_bytes(b"k" * 32)
            key_path.chmod(0o644)
            self.assertEqual(load_or_create_ledger_key(directory), b"k" * 32)
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)

    def test_env_key_override_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict("os.environ", {"NEDSTER_LEDGER_KEY": "secret"}):
                key = load_or_create_ledger_key(directory)
            self.assertFalse((Path(directory) / "ledger.key").exists())
            self.assertEqual(key, hashlib.sha256(b"secret").digest())


if __name__ == "__main__":
    unittest.main()
