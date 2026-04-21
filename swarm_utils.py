import sqlite3
import time
from pathlib import Path

DB_PATH = "swarm.db"


def acquire_lock(project_path: str, filepath: str, agent_id: int, timeout: int = 10):
    """Tries to acquire a lock on a file for a specific agent."""
    db = Path(project_path) / DB_PATH
    absolute_filepath = str((Path(project_path) / filepath).resolve())

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            conn = sqlite3.connect(db)
            cursor = conn.cursor()

            # Check for existing lock
            cursor.execute(
                "SELECT agent_id FROM file_locks WHERE filepath = ?",
                (absolute_filepath,),
            )
            result = cursor.fetchone()

            if result is None:
                # No lock, acquire it
                cursor.execute(
                    "INSERT INTO file_locks (filepath, agent_id, timestamp) VALUES (?, ?, ?)",
                    (absolute_filepath, agent_id, time.time()),
                )
                conn.commit()
                conn.close()
                print(f"[Agent {agent_id}] Acquired lock for {filepath}")
                return True
            elif result[0] == agent_id:
                # We already have the lock
                conn.close()
                return True
            else:
                # Locked by another agent
                conn.close()
                print(
                    f"[Agent {agent_id}] Waiting for lock on {filepath} (held by Agent {result[0]})"
                )
                time.sleep(0.5)

        except sqlite3.OperationalError as e:
            # Database might be locked, wait
            print(f"[Agent {agent_id}] Database locked, retrying...")
            time.sleep(0.2)

    return False


def release_lock(project_path: str, filepath: str, agent_id: int):
    """Releases a lock on a file."""
    db = Path(project_path) / DB_PATH
    absolute_filepath = str((Path(project_path) / filepath).resolve())

    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM file_locks WHERE filepath = ? AND agent_id = ?",
            (absolute_filepath, agent_id),
        )
        conn.commit()
        conn.close()
        print(f"[Agent {agent_id}] Released lock for {filepath}")
    except Exception as e:
        print(f"Error releasing lock for {filepath}: {e}")


def update_job_status(project_path: str, job_id: int, status: str):
    """Updates the status of a job."""
    db = Path(project_path) / DB_PATH
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating job {job_id} status: {e}")
