from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
import psutil
import subprocess

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Path to the main project directory, which is one level up
PROJECT_ROOT = Path(__file__).parent.parent


def get_nedster_state():
    """Reads the nedster_state.json file."""
    try:
        with open(PROJECT_ROOT / "nedster_state.json") as f:
            return json.load(f)
    except Exception:
        return {}


def get_session_log():
    """Reads the latest session log."""
    try:
        log_dir = Path.home() / ".aria" / "sessions"
        latest_log = sorted(log_dir.glob("*.log"), key=os.path.getmtime, reverse=True)[
            0
        ]
        with open(latest_log) as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def get_vram_usage():
    """Gets VRAM usage from nvidia-smi."""
    try:
        vram = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if vram.returncode == 0:
            used, total = vram.stdout.strip().split(",")
            return f"{float(used.strip()) / 1024:.1f} / {float(total.strip()) / 1024:.1f} GB"
    except Exception:
        return "N/A"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    state = get_nedster_state()
    log = get_session_log()

    context = {
        "request": request,
        "project_name": state.get("project", "N/A"),
        "model": state.get("model", "N/A"),
        "last_updated": state.get("updated", "N/A"),
        "vram": get_vram_usage(),
        "cpu": f"{psutil.cpu_percent()}%",
        "ram": f"{psutil.virtual_memory().percent}%",
        "log_events": log[-20:],  # Last 20 events
    }
    return templates.TemplateResponse("index.html", context)
