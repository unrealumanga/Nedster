import json
import os
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class NedsterAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/api/models":
            try:
                r = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                models = [line.split()[0] for line in r.stdout.strip().split("\n")[1:] if line]
                self._send_json({"models": models})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        elif self.path == "/api/sessions":
            path = os.path.expanduser("~/.aria/sessions_index.json")
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        self._send_json(json.load(f))
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
            else:
                self._send_json([])
        elif self.path == "/api/vram":
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                parts = r.stdout.strip().split(',')
                if len(parts) == 2:
                    self._send_json({
                        "used_mb": int(parts[0]),
                        "total_mb": int(parts[1])
                    })
                else:
                    self._send_json({"error": "Parse error"}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/api/switch":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                model = data.get("model")
                if model:
                    import nedster
                    if hasattr(nedster, "_ACTIVE_MODEL"):
                        subprocess.run(["ollama", "stop", nedster._ACTIVE_MODEL], capture_output=True)
                        nedster._ACTIVE_MODEL = model
                        subprocess.run(["ollama", "run", model, ""], capture_output=True)
                        self._send_json({"status": "success", "model": model})
                    else:
                        self._send_json({"error": "Nedster not initialized"}, 500)
                else:
                    self._send_json({"error": "Missing model"}, 400)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"error": "Not found"}, 404)

def start_api_server():
    server = HTTPServer(("0.0.0.0", 7702), NedsterAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
