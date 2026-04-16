# File: ingestion.py
import os
import fitz  # PyMuPDF
import tiktoken
import chromadb
from tqdm import tqdm
from rag_engine.embedder import Embedder


def get_text_from_file(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    text = ""
    if ext == ".pdf":
        try:
            doc = fitz.open(filepath)
            for page in doc:
                extracted = page.get_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}")
    elif ext in [
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".java",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".html",
        ".css",
        ".sh",
    ]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return text


def chunk_text(text, filename, chunk_size=1200, overlap=200):
    enc = tiktoken.get_encoding("cl100k_base")
    # Split by double newline for semantic paragraph boundaries
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        tokens = enc.encode(p)

        # If adding this paragraph exceeds limit, finalize current chunk
        if current_length + len(tokens) > chunk_size and current_chunk:
            # Prepend the structural context! (Contextual Chunking)
            chunk_str = f"[Document: {filename}]\n" + "\n\n".join(current_chunk)
            chunks.append(chunk_str)
            current_chunk = [p]
            current_length = len(tokens)
        else:
            current_chunk.append(p)
            current_length += len(tokens)

    if current_chunk:
        chunk_str = f"[Document: {filename}]\n" + "\n\n".join(current_chunk)
        chunks.append(chunk_str)

    # Second pass: token-level overlap stitching
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            prev_tokens = enc.encode(chunks[i - 1])
            overlap_text = enc.decode(prev_tokens[-overlap:])
            # Prepend overlap text (without the [Document:] header)
            body = chunk.split("\n", 1)[1] if "\n" in chunk else chunk
            header = f"[Document: {filename}]\n"
            overlapped_chunks.append(header + overlap_text + "\n" + body)

    return overlapped_chunks


def embed_file_chunks(chunks, embedder, collection, filename):
    for i in tqdm(
        range(0, len(chunks), embedder.batch_size),
        desc=f"Embedding from {filename}",
    ):
        batch = chunks[i : i + embedder.batch_size]
        try:
            embeddings = embedder.embed_documents(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocate" in str(e).lower():
                print(f"OOM on batch {i}, retrying with batch_size=4...")
                embedder.batch_size = 4
                embeddings = embedder.embed_documents(batch)
            else:
                raise

        ids = [f"{filename}_chunk_{i + j}" for j in range(len(batch))]
        metadatas = [
            {
                "source_file": filename,
                "chunk_index": int(i + j),
                "total_chunks": int(len(chunks)),
                "char_count": int(len(batch[j])),
            }
            for j in range(len(batch))
        ]

        collection.add(
            documents=batch,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,  # type: ignore
        )


def ingest_folder(folder_path):
    print(f"Scanning folder: {folder_path}")
    files_to_process = []
    supported_exts = (
        ".txt",
        ".md",
        ".pdf",
        ".py",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".java",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".html",
        ".css",
        ".sh",
    )
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_exts):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print("No valid files found.")
        return

    embedder = Embedder()

    # Store in ChromaDB local persistent ./chroma_db
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection(name="rag_docs")
    except Exception as e:
        print(
            f"ChromaDB initialization failed. If corrupt, run 'python main.py reset'. Error: {e}"
        )
        return

    total_chunks_ingested = 0
    total_files = len(files_to_process)

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        text = get_text_from_file(filepath)
        if not text.strip():
            continue

        chunks = chunk_text(text, filename=filename, chunk_size=1200, overlap=200)
        total_chunks_ingested += len(chunks)

        embed_file_chunks(chunks, embedder, collection, filename)

    print(
        f"\nIngested {total_chunks_ingested} context-aware chunks from {total_files} files -> ChromaDB"
    )
