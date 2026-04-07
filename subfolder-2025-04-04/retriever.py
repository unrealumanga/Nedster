# File: retriever.py
import chromadb
import uuid
from datetime import datetime
from rank_bm25 import BM25Okapi
from embedder import Embedder
from reranker import Reranker


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.reranker = Reranker()

        self.doc_collection = None
        self.mem_collection = None
        self.bm25 = None
        self.corpus_docs = []
        self.corpus_metas = []

        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            # Collection for static ingested documents
            try:
                self.doc_collection = self.client.get_collection(name="rag_docs")
                self._init_bm25()
            except:
                self.doc_collection = None

            # Collection for long-term chat memory
            self.mem_collection = self.client.get_or_create_collection(
                name="chat_memory"
            )

        except Exception as e:
            print(f"ChromaDB connection failed: {e}")

    def _init_bm25(self):
        """Builds an in-memory BM25 index on startup from ChromaDB docs."""
        if not self.doc_collection:
            return

        print("Building BM25 Keyword index...")
        all_data = self.doc_collection.get(include=["documents", "metadatas"])
        if all_data and all_data.get("documents"):
            docs = all_data.get("documents")
            metas = all_data.get("metadatas")
            if docs:
                self.corpus_docs = list(docs)
                self.corpus_metas = list(metas) if metas else []

                # Simple tokenization for BM25
                tokenized_corpus = [
                    str(doc).lower().split() for doc in self.corpus_docs
                ]
                if tokenized_corpus:
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    print(f"BM25 index built with {len(tokenized_corpus)} chunks.")

    def add_to_memory(
        self, user_query: str, ai_response: str, session_id: str = "unknown"
    ):
        """
        Only saves to long-term memory if the exchange is IMPORTANT.
        Importance scoring (0.0 to 1.0):
        - Base: 0.0
        - Response longer than 100 chars: +0.3
        - Response longer than 300 chars: +0.2 more
        - Query contains question words (what/how/why/when/who/which): +0.2
        - Response does NOT contain "don't have that information": +0.1
        - Query longer than 30 chars (not trivial): +0.1
        - Threshold to save: importance >= 0.4
        """
        if not self.mem_collection:
            return

        # Score importance
        score = 0.0
        resp_lower = ai_response.lower()
        query_lower = user_query.lower()

        if len(ai_response) > 100:
            score += 0.3
        if len(ai_response) > 300:
            score += 0.2
        if any(
            w in query_lower
            for w in [
                "what",
                "how",
                "why",
                "when",
                "who",
                "which",
                "explain",
                "describe",
            ]
        ):
            score += 0.2
        if (
            "don't have that information" not in resp_lower
            and "i don't know" not in resp_lower
        ):
            score += 0.1
        if len(user_query) > 30:
            score += 0.1

        if score < 0.4:
            print(f"[\033[90mMemory skipped (importance={score:.1f})\033[0m]")
            return

        text = f"Past Conversation -> User: {user_query}\nAssistant: {ai_response}"
        embedding = self.embedder.embed_documents([text])[0]
        doc_id = f"mem_{uuid.uuid4().hex[:8]}"

        self.mem_collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[
                {
                    "source_file": "Long-Term Memory",
                    "type": "memory",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "importance": round(score, 2),
                }
            ],
        )
        print(
            f"  • [\x1b[38;5;245mMemory saved (importance={score:.1f}, session={session_id})\x1b[0m]"
        )

        # Auto-prune if collection exceeds 600 memories
        self._prune_old_memories(max_count=600)

    def _prune_old_memories(self, max_count: int = 600):
        """
        If memory collection exceeds max_count, delete the oldest 50 entries.
        Oldest = lowest importance score. Uses ChromaDB get() then delete().
        """
        if not self.mem_collection:
            return
        count = self.mem_collection.count()
        if count <= max_count:
            return

        # Get all memories with metadata
        all_data = self.mem_collection.get(include=["metadatas"])
        if not all_data or not all_data.get("ids"):
            return

        ids = all_data["ids"]
        metas = all_data.get("metadatas") or []

        # Sort by importance ASC (lowest importance deleted first)
        # Fallback sort key if importance missing: 0.0
        paired = list(zip(ids, metas))
        paired.sort(key=lambda x: float((x[1] or {}).get("importance", 0.0)))

        # Delete the 50 lowest importance entries
        ids_to_delete = [p[0] for p in paired[:50]]
        self.mem_collection.delete(ids=ids_to_delete)
        print(
            f"[\033[90mMemory pruned: deleted {len(ids_to_delete)} low-importance entries\033[0m]"
        )

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed_queries([query])[0]
        unique_docs = {}  # map text -> meta

        # 1. Vector Search (Semantic)
        if self.doc_collection:
            doc_results = self.doc_collection.query(
                query_embeddings=[query_embedding], n_results=15
            )
            if doc_results.get("documents") and doc_results["documents"]:
                docs = doc_results["documents"][0]
                m_list = doc_results.get("metadatas")
                metas = m_list[0] if m_list else []
                for d, m in zip(docs, metas):
                    unique_docs[d] = m

        # 2. BM25 Search (Keyword/Exact match)
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_idx = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:15]
            for idx in top_n_idx:
                if bm25_scores[idx] > 0:
                    d = self.corpus_docs[idx]
                    m = self.corpus_metas[idx]
                    if d not in unique_docs:
                        unique_docs[d] = m

        # 3. Retrieve from long-term memory
        if self.mem_collection and self.mem_collection.count() > 0:
            # Retrieve more, then re-score with recency weight before picking top 3
            mem_results = self.mem_collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["documents", "metadatas", "distances"],
            )
            # Recency + importance weighting:
            # final_score = (1 - cosine_distance) * 0.6 + importance * 0.2 + recency * 0.2
            # recency: memories from this session get 1.0, older get decayed by days
            if (
                mem_results
                and mem_results.get("documents")
                and mem_results["documents"]
            ):
                m_docs = mem_results.get("documents")
                if m_docs:
                    docs = m_docs[0]
                    m_list = mem_results.get("metadatas")
                    metas = m_list[0] if m_list else []
                    distances = (mem_results.get("distances") or [[]])[0]
                    scored_mems = []
                    now = datetime.now()
                    for d, m, dist in zip(docs, metas, distances):
                        m = m or {}
                        semantic = max(0.0, 1.0 - dist)
                        importance = float(m.get("importance", 0.3))
                        # Recency: parse timestamp, decay by days (max 1.0 = today)
                        try:
                            ts = datetime.fromisoformat(
                                str(m.get("timestamp", now.isoformat()))
                            )
                            days_old = max(0, (now - ts).days)
                            recency = 1.0 / (1.0 + days_old * 0.1)
                        except Exception:
                            recency = 0.5
                        final = semantic * 0.6 + importance * 0.2 + recency * 0.2
                        scored_mems.append((d, m, final))
                    scored_mems.sort(key=lambda x: x[2], reverse=True)
                    for d, m, _ in scored_mems[:3]:
                        if d not in unique_docs:
                            unique_docs[d] = m

        all_docs = list(unique_docs.keys())
        all_metas = list(unique_docs.values())

        if not all_docs:
            return []

        # 4. Rerank combined unique results (Vector + BM25 + Memory) -> return top 4
        ranked_results = self.reranker.rerank(query, all_docs, all_metas)
        top_4 = ranked_results[:4]

        # Print retrieved chunks with filenames + reranker scores
        print("\n--- Retrieved & Reranked Context ---")
        for idx, (doc, score, meta) in enumerate(top_4):
            source = (
                meta.get("source_file", "Unknown")
                if meta and isinstance(meta, dict)
                else "Unknown"
            )
            print(f"[{idx + 1}] Source: {source} | Score: {score:.2f}")
        print("------------------------------------\n")

        return top_4
