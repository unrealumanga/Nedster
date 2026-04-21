import sqlite3
import subprocess
import json
import asyncio
import ollama
from pathlib import Path
import time

DB_PATH = "swarm.db"
OLLAMA_MODEL_PRIMARY = "aria-local"
OLLAMA_MODEL_FALLBACK = "qwen3.5:9b"
PYTHON_EXECUTABLE = "python3"


class OllamaQueue:
    """A simple async queue to serialize Ollama requests."""

    def __init__(self):
        self._lock = asyncio.Lock()

    async def chat(self, *args, **kwargs):
        async with self._lock:
            print("[Coordinator] Ollama lock acquired...")
            response = await ollama.AsyncClient().chat(*args, **kwargs)
            print("[Coordinator] Ollama lock released.")
            return response


class SwarmCoordinator:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.db_path = self.project_path / DB_PATH
        self.ollama_queue = OllamaQueue()
        self._setup_database()

    def _setup_database(self):
        """Initializes the SQLite database for jobs and locks."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, task_description TEXT NOT NULL, directories TEXT NOT NULL, status TEXT NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS file_locks (filepath TEXT PRIMARY KEY, agent_id INTEGER NOT NULL, timestamp REAL NOT NULL)"
        )
        conn.commit()
        conn.close()

    async def decompose_task(self, main_prompt):
        """Uses Ollama to decompose the main prompt into sub-tasks."""
        print("[Coordinator] Decomposing task...")
        decomposition_prompt = f'You are a swarm coordinator... Main Task: "{main_prompt}"... Respond ONLY with the JSON array.'

        current_model = OLLAMA_MODEL_PRIMARY
        try:
            response = await self.ollama_queue.chat(
                model=current_model,
                messages=[{"role": "user", "content": decomposition_prompt}],
                options={"temperature": 0.1},
            )
        except ollama.ResponseError as e:
            if e.status_code == 404:
                print(
                    f"[Coordinator] WARN: Model '{OLLAMA_MODEL_PRIMARY}' not found. Falling back to '{OLLAMA_MODEL_FALLBACK}'."
                )
                current_model = OLLAMA_MODEL_FALLBACK
                try:
                    response = await self.ollama_queue.chat(
                        model=current_model,
                        messages=[{"role": "user", "content": decomposition_prompt}],
                        options={"temperature": 0.1},
                    )
                except Exception as fallback_e:
                    print(f"Error during fallback decomposition: {fallback_e}")
                    return None
            else:
                print(f"Error during decomposition: {e}")
                return None

        content = response["message"]["content"].strip("`json \n")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(
                f"Error decoding task decomposition JSON: {e}\nRaw response: {content}"
            )
            return None

    async def run_swarm(self, main_prompt):
        """Decomposes the task and runs the agent swarm."""
        jobs = await self.decompose_task(main_prompt)
        if not jobs:
            print("Could not decompose task. Aborting.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        job_ids = []
        for job in jobs:
            cursor.execute(
                "INSERT INTO jobs (task_description, directories, status) VALUES (?, ?, ?)",
                (job["task_description"], json.dumps(job["directories"]), "pending"),
            )
            job_ids.append(cursor.lastrowid)
        conn.commit()
        conn.close()

        print(f"[Coordinator] Created {len(jobs)} jobs. Spawning agents...")
        agent_processes = []
        for i, job in enumerate(jobs):
            job_id = job_ids[i]
            cmd = [
                PYTHON_EXECUTABLE,
                "nedster.py",
                "work",
                "--project-dir",
                str(self.project_path),
                "--task",
                job["task_description"],
                "--job-id",
                str(job_id),
                "--scoped-dirs",
                ",".join(job["directories"]),
            ]
            proc = await asyncio.create_subprocess_exec(*cmd)
            agent_processes.append(proc)

        await asyncio.gather(*[proc.wait() for proc in agent_processes])
        print("[Coordinator] All agent processes have completed.")


def main_swarm_entry(prompt, project_dir):
    """Entry point for the swarm command."""
    coordinator = SwarmCoordinator(project_dir)
    asyncio.run(coordinator.run_swarm(prompt))
