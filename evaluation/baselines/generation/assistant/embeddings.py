from __future__ import annotations

from pathlib import Path
from .config import Config, PROJECT_ROOT, configure_cache, fingerprint


class Encoder:
    def __init__(self, config: Config, root: Path = PROJECT_ROOT):
        configure_cache(root)
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(config.cpu_threads)
        self.config = config
        self.model = SentenceTransformer(config.embedding_model, device="cpu")
        self.tokenizer = self.model.tokenizer
        if not self.tokenizer.is_fast:
            raise RuntimeError("Chunk offsets require the model's fast tokenizer")
        self.model.max_seq_length = min(256, self.model.max_seq_length)
        dimension = self.model.get_embedding_dimension() if hasattr(self.model, "get_embedding_dimension") else self.model.get_sentence_embedding_dimension()
        if dimension != config.embedding_dimension:
            raise ValueError(f"Expected {config.embedding_dimension} dimensions, received {dimension}")
        transformer = self.model[0].auto_model
        self.revision = getattr(transformer.config, "_commit_hash", None)
        self.identity = fingerprint({"model": config.embedding_model, "revision": self.revision,
                                     "max_length": self.model.max_seq_length, "normalize": True})

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def encode(self, texts: list[str]):
        import numpy as np
        for text in texts:
            if self.count(text) > self.model.max_seq_length:
                raise ValueError("Embedding text exceeds the tokenizer limit; segment it explicitly")
        if not texts:
            return np.empty((0, self.config.embedding_dimension), dtype=np.float32)
        values = self.model.encode(texts, batch_size=self.config.embedding_batch_size,
                                   normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(texts), self.config.embedding_dimension) or not np.isfinite(values).all():
            raise ValueError("Embedding output has invalid dimensions or nonfinite values")
        return values

    def segments(self, text: str, limit=180) -> list[str]:
        data = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
        offsets = data["offset_mapping"]
        return [text[offsets[i][0]:offsets[min(i + limit, len(offsets)) - 1][1]]
                for i in range(0, len(offsets), limit)] or [""]

    def encode_documents(self, texts: list[str]):
        """Metadata abstracts may be longer than passages; use all their segments."""
        import numpy as np
        result = []
        for text in texts:
            segments = self.segments(text)
            values = self.encode(segments)
            weights = [max(1, self.count(s) - 2) for s in segments]
            mean = np.average(values, axis=0, weights=weights)
            result.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
        return np.asarray(result, dtype=np.float32)
