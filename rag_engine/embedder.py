# File: embedder.py
import torch
import psutil
from sentence_transformers import SentenceTransformer

_EMBEDDER_INSTANCE = None


def get_embedder_model(model_name, device):
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"dtype": torch.bfloat16},
        )
    return _EMBEDDER_INSTANCE


class Embedder:
    def __init__(self):
        self.device = "cpu"
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"

        # CPU embedding tip: pre-embed ALL docs at ingestion, never at query time
        self.model = get_embedder_model(self.model_name, self.device)

        # Pin CPU threads to i7-11700k P-cores for embedding
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)

        self.batch_size = 16  # smaller batch for CPU
        self.embedding_dim = 512  # MRL truncation, balances speed vs quality on CPU

        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Embedder loaded on CPU — RAM usage: {ram_mb:.2f} MB")

    def embed_queries(self, queries):
        instruction = (
            "Instruct: Given a user question, retrieve relevant passages\nQuery: "
        )
        texts = [instruction + q for q in queries]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            truncate_dim=self.embedding_dim,
        ).tolist()

    def embed_documents(self, documents):
        instruction = "Instruct: Represent this document for retrieval\nPassage: "
        texts = [instruction + d for d in documents]

        # CPU RAM OOM during embedding: reduce batch_size to 4 automatically and retry
        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                truncate_dim=self.embedding_dim,
            ).tolist()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocate" in str(e).lower():
                print("RAM OOM detected. Retrying with batch_size=4...")
                return self.model.encode(
                    texts,
                    batch_size=4,
                    normalize_embeddings=True,
                    truncate_dim=self.embedding_dim,
                ).tolist()
            raise e
