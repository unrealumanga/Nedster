# ===----------------------------------------------------------------------=== #
# Nedster RAG Engine & Grounding Evidence Retrieval Module
# ===----------------------------------------------------------------------=== #

from math import sqrt
from .inference import simd_dot_product

struct GroundingSource:
    var source_id: String
    var label: String
    var content: String

    fn __init__(inout self, source_id: String, label: String, content: String):
        self.source_id = source_id
        self.label = label
        self.content = content


fn cosine_similarity(vec_a: List[Float32], vec_b: List[Float32], dim: Int) -> Float32:
    """Computes SIMD accelerated cosine similarity between vector embeddings."""
    var dot = simd_dot_product(vec_a, vec_b, dim)
    var norm_a = sqrt(simd_dot_product(vec_a, vec_a, dim))
    var norm_b = sqrt(simd_dot_product(vec_b, vec_b, dim))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


struct VectorIndex:
    var sources: List[GroundingSource]
    var embeddings: List[List[Float32]]
    var dimension: Int

    fn __init__(inout self, dimension: Int = 384):
        self.sources = List[GroundingSource]()
        self.embeddings = List[List[Float32]]()
        self.dimension = dimension

    fn add_document(inout self, source_id: String, label: String, content: String, embedding: List[Float32]):
        self.sources.append(GroundingSource(source_id, label, content))
        self.embeddings.append(embedding)

    fn search_top_k(self, query_vec: List[Float32], k: Int = 3) -> List[GroundingSource]:
        """Performs fast SIMD vector search over index."""
        var results = List[GroundingSource]()
        if len(self.sources) == 0:
            return results

        var scores = List[Float32]()
        for i in range(len(self.embeddings)):
            var sim = cosine_similarity(query_vec, self.embeddings[i], self.dimension)
            scores.append(sim)

        # Basic top-k extraction
        var count = k
        if len(self.sources) < count:
            count = len(self.sources)

        for _ in range(count):
            var best_idx = 0
            var best_score: Float32 = -1.0
            for j in range(len(scores)):
                if scores[j] > best_score:
                    best_score = scores[j]
                    best_idx = j
            if best_score > -1.0:
                results.append(self.sources[best_idx])
                scores[best_idx] = -10.0 # mark used

        return results
