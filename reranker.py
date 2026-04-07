# File: reranker.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Reranker:
    def __init__(self):
        self.device = "cpu"
        self.model_name = "Qwen/Qwen3-Reranker-0.6B"
        self._cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        # Get yes/no tokens
        self.yes_token = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def _score_pair(self, query: str, document: str) -> float:
        cache_key = (query, document[:200])  # truncate doc for key size
        if cache_key in self._cache:
            return self._cache[cache_key]

        text = f"Query: {query}\nDocument: {document}\nRelevant:"
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]

            # Extract yes/no token logits -> softmax -> relevance score
            if logits.dim() > 1:  # if sequence length > 1
                yes_logit = logits[-1, self.yes_token]
                no_logit = logits[-1, self.no_token]
            else:
                yes_logit = logits[self.yes_token]
                no_logit = logits[self.no_token]

            scores = torch.tensor([no_logit, yes_logit], dtype=torch.float32)
            probs = torch.nn.functional.softmax(scores, dim=0)
            result = probs[1].item()  # return 'Yes' probability

            if len(self._cache) > 512:  # cap cache size
                self._cache.clear()
            self._cache[cache_key] = result
            return result

    def rerank(self, query: str, documents: list, metadatas: list) -> list:
        # Reranker tip: only run reranker on top 15, not all results
        results = []
        for doc, meta in zip(documents, metadatas):
            score = self._score_pair(query, doc)
            results.append((doc, score, meta))

        # Return sorted [(doc_text, score, metadata), ...] list
        results.sort(key=lambda x: x[1], reverse=True)
        return results
