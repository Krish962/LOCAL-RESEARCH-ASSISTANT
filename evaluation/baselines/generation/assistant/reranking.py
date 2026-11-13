from __future__ import annotations

from .config import Config, configure_cache
from .schemas import Hit


class Reranker:
    def __init__(self, config: Config):
        configure_cache()
        import torch
        from sentence_transformers import CrossEncoder
        torch.set_num_threads(config.cpu_threads)
        self.config = config
        self.model = CrossEncoder(config.reranker_model, device="cpu", max_length=512)
        self.tokenizer = self.model.tokenizer
        self.model.model.eval()
        self.revision = getattr(self.model.model.config, "_commit_hash", None)

    def scores(self, question: str, texts: list[str]) -> tuple[list[float], list[int]]:
        import torch
        import numpy as np
        pairs = [(question, t) for t in texts]
        lengths = [len(self.tokenizer(q, t, truncation=False)["input_ids"]) for q, t in pairs]
        if any(n > 512 for n in lengths):
            raise ValueError("A question-passage pair exceeds reranker limit; shorten the question explicitly")
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.config.reranker_batch_size):
                batch = pairs[start:start + self.config.reranker_batch_size]
                inputs = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                        padding=True, truncation=False, return_tensors="pt")
                logits = self.model.model(**inputs).logits.reshape(-1).cpu().numpy()
                if not np.isfinite(logits).all():
                    raise RuntimeError("Reranker returned nonfinite values")
                scores.extend(float(x) for x in logits)
        return scores, lengths

    def rerank(self, question: str, hits: list[Hit]) -> tuple[list[Hit], dict]:
        # The exact passage remains unchanged; avoid title prefixes exceeding budgets.
        scores, lengths = self.scores(question, [h.chunk.text for h in hits])
        for hit, score in zip(hits, scores):
            hit.scores["cross_encoder"] = score
        ordered = sorted(hits, key=lambda h: (-h.scores["cross_encoder"], h.chunk.chunk_id))
        for rank, hit in enumerate(ordered, 1):
            hit.ranks["cross_encoder"] = rank
        return ordered, {"model": self.config.reranker_model, "revision": self.revision,
                         "pair_lengths": lengths, "pairs": len(hits), "executed": True}
