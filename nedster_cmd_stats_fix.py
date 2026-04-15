def print_stats(project_dir: str = "."):
    """Show VRAM, token budget, tool status, and pending todos.

    FIX: The original split this function — the todo-reading block was prepended
    before the docstring/imports but the function signature took no arguments.
    cmd_stats() called print_stats() with no args, causing a NameError for
    `project_dir` inside the todo block. Merged into a single function with a
    default argument.
    """
    import os, json, torch, psutil, subprocess

    # --- Pending todos ---
    todo_path = os.path.join(str(project_dir), ".nedster_todos.json")
    if os.path.exists(todo_path):
        try:
            with open(todo_path) as f:
                todos = json.load(f)
            pending = [t for t in todos if t.get("status") != "completed"]
            if pending:
                print(f"\nPending tasks ({len(pending)}):")
                for t in pending[:5]:
                    print(f"  [{t.get('status','todo')}] {t.get('content','')[:60]}")
        except Exception:
            pass

    tui = NedsterTUI()

    # CPU RAM
    cpu_ram = psutil.virtual_memory()
    used_gb = cpu_ram.used / (1024**3)
    total_gb = cpu_ram.total / (1024**3)
    print(f"CPU RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total")

    # GPU VRAM
    try:
        vram = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if vram.returncode == 0:
            for line in vram.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) == 2:
                    used = float(parts[0].strip()) / 1024
                    total = float(parts[1].strip()) / 1024
                    print(f"GPU VRAM: {used:.1f} GB used / {total:.1f} GB total")
    except Exception:
        print("GPU VRAM: nvidia-smi not available")

    # ChromaDB vectors
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="rag_docs")
        print(f"ChromaDB: {collection.count()} vectors stored")
    except Exception:
        print("ChromaDB: 0 vectors stored (or not initialized)")

    # Token budget
    print(f"\nToken Budget: 4096 (aria-qwen)")
    print(f"Model: aria-qwen (Qwen3.5:9b)")


def cmd_stats(project_dir: str = "."):
    """Show stats — pass project_dir so print_stats can find todos."""
    print_stats(project_dir)
