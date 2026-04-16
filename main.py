# File: main.py
import argparse
import os
import sys
import shutil
import psutil
import subprocess
import chromadb


def print_stats():
    import torch

    # CPU RAM: X.X GB used / 64.0 GB total
    cpu_ram = psutil.virtual_memory()
    used_gb = cpu_ram.used / (1024**3)
    total_gb = cpu_ram.total / (1024**3)
    print(f"CPU RAM: {used_gb:.1f} GB used / {total_gb:.1f} GB total")

    # GPU VRAM: X.X GB used / 8.0 GB total
    try:
        vram = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if vram.returncode == 0:
            lines = vram.stdout.strip().split("\n")
            for line in lines:
                parts = line.split(",")
                if len(parts) == 2:
                    used = float(parts[0].replace("MiB", "").strip()) / 1024
                    total = float(parts[1].replace("MiB", "").strip()) / 1024
                    print(f"GPU VRAM: {used:.1f} GB used / {total:.1f} GB total")

                    if used > (total * 0.95):
                        print("Warning: VRAM OOM risk detected!")
                        print("Suggest: ollama stop && ollama pull qwen3.5:9b-q3_k_m")
    except FileNotFoundError:
        print("GPU VRAM: nvidia-smi not found")

    # ChromaDB vectors
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="rag_docs")
        print(f"ChromaDB: {collection.count()} vectors stored")
    except Exception:
        print("ChromaDB: 0 vectors stored (or not initialized)")

    print("\n[Speed Tips]")
    print(
        "  Flash Attention:",
        os.environ.get("OLLAMA_FLASH_ATTENTION", "NOT SET (run start.sh)"),
    )
    print(
        "  KV Cache Type: ",
        os.environ.get("OLLAMA_KV_CACHE_TYPE", "NOT SET (run start.sh)"),
    )
    print("  CPU Threads:   ", torch.get_num_threads())
    print("  Model:          aria-qwen (Modelfile-optimized)")


def check_ollama():
    try:
        import ollama

        ollama.list()
    except Exception:
        print("Error: Ollama not running. Run: ollama serve")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Local RAG Stack for RTX 3060 Ti")
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument(
        "--folder", type=str, required=True, help="Folder to ingest"
    )

    # chat
    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument(
        "--think", action="store_true", help="Enable thinking mode"
    )

    # query
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("text", type=str, help="Question text")
    query_parser.add_argument(
        "--think", action="store_true", help="Enable thinking mode"
    )

    # plan
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("text", type=str)
    plan_parser.add_argument("--think", action="store_true")

    # stats
    subparsers.add_parser("stats")

    # reset
    subparsers.add_parser("reset")

    args = parser.parse_args()

    if args.command == "ingest":
        from rag_engine.ingestion import ingest_folder

        ingest_folder(args.folder)

    elif args.command == "chat":
        check_ollama()
        from rag_engine.rag import RAGPipeline

        pipeline = RAGPipeline()
        import os
        milestone_path = os.path.expanduser("~/.aria/milestones.md")
        milestone_count = 0
        if os.path.exists(milestone_path):
            with open(milestone_path) as f:
                milestone_count = len([l for l in f if l.startswith("##")])

        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            time_greeting = "Morning"
        elif hour < 17:
            time_greeting = "Afternoon"
        else:
            time_greeting = "Evening"

        if milestone_count > 0:
            print(f"Aria. {time_greeting}, H2. "
                  f"({milestone_count} past sessions loaded)")
        else:
            print("Aria. Ready.")
        print("Type 'exit' to quit | '/clear' to reset memory | '/plan [task]' for multi-step\n")
        while True:
            try:
                user_input = input("\nYou: ")
                if not user_input.strip():
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input.lower() in ["/clear", "/reset-memory"]:
                    pipeline.memory.clear()
                    print("[Short-term memory cleared. Long-term memory preserved.]")
                    continue
                if user_input.strip() == "/stats":
                    print_stats()
                    continue
                if user_input.strip() == "/memory":
                    summary = pipeline.memory.session_summary
                    print(f"Session summary:\n{summary if summary else '(empty)'}\n")
                    continue

                lower_input = user_input.lower()
                if lower_input.startswith("/plan ") or any(
                    kw in lower_input
                    for kw in ["step by step", "plan:", "architect", "design a system"]
                ):
                    if lower_input.startswith("/plan "):
                        user_input = user_input[6:]
                    pipeline.plan_and_execute(user_input, think=args.think)
                else:
                    pipeline.generate(user_input, think=args.think)
            except KeyboardInterrupt:
                break

    elif args.command == "query":
        check_ollama()
        from rag_engine.rag import RAGPipeline

        pipeline = RAGPipeline()
        lower_input = args.text.lower()
        if lower_input.startswith("/plan ") or any(
            kw in lower_input
            for kw in ["step by step", "plan:", "architect", "design a system"]
        ):
            if lower_input.startswith("/plan "):
                args.text = args.text[6:]
            pipeline.plan_and_execute(args.text, think=args.think)
        else:
            pipeline.generate(args.text, think=args.think)

    elif args.command == "plan":
        check_ollama()
        from rag_engine.rag import RAGPipeline

        pipeline = RAGPipeline()
        pipeline.plan_and_execute(args.text, think=args.think)

    elif args.command == "stats":
        print_stats()
        # To print tool stats if pipeline exists we would need to instantiate or pass it
        # The prompt says "Add to stats output: print tool_stats if available."
        # If we just print it in print_stats, we need an instance of RAGPipeline
        # So maybe we should instantiate pipeline and print its tool_stats?
        # Or print it in print_stats if passed?
        try:
            from rag_engine.rag import RAGPipeline

            p = RAGPipeline()
            print(f"Tool Stats: {p.tool_stats}")
        except Exception as e:
            pass

    elif args.command == "reset":
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("ChromaDB wiped. Re-run ingest to populate.")
        else:
            print("No ChromaDB found to reset.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
