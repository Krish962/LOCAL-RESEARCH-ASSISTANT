from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from .artifacts import atomic_json, read_jsonl, write_jsonl
from .config import Config, fingerprint
from .schemas import Chunk, Hit, Paper


class Snapshot:
    def __init__(self, path: Path, config: Config):
        import numpy as np
        from qdrant_client import QdrantClient
        self.path, self.config = path, config
        self.chunks = [Chunk(**row) for row in read_jsonl(path / "chunks.jsonl")]
        self.papers = [Paper.from_dict(row) for row in read_jsonl(path / "papers.jsonl")]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.vectors = np.load(path / "vectors.npy", allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), config.embedding_dimension):
            raise ValueError("Snapshot vector/chunk dimension mismatch")
        if len(self.by_id) != len(self.chunks):
            raise ValueError("Snapshot has duplicate chunk IDs")
        self.client = QdrantClient(path=str(path / "qdrant"))
        count = self.client.count("chunks", exact=True).count
        if count != len(self.chunks):
            self.close()
            raise ValueError("Qdrant count disagrees with saved chunk records")

    def search(self, vector, eligible: set[str], limit: int) -> list[Hit]:
        from qdrant_client import models
        if not eligible:
            return []
        condition = models.Filter(must=[
            models.FieldCondition(key="paper_id", match=models.MatchAny(any=sorted(eligible))),
            models.FieldCondition(key="corpus_id", match=models.MatchValue(value=self.config.corpus_id)),
        ])
        rows = self.client.query_points("chunks", query=vector.tolist(), query_filter=condition,
                                        limit=limit, with_payload=True).points
        hits = []
        for rank, row in enumerate(rows, 1):
            chunk = self.by_id.get(str(row.id))
            if chunk is None or chunk.paper_id not in eligible:
                raise RuntimeError("Dense result violated snapshot/metadata constraints")
            hits.append(Hit(chunk, {"dense": float(row.score)}, {"dense": rank}))
        return hits

    def close(self):
        self.client.close()


def build_snapshot(path: Path, config: Config, papers: list[Paper], chunks: list[Chunk], vectors) -> dict:
    import numpy as np
    from qdrant_client import QdrantClient, models
    if not chunks or vectors.shape != (len(chunks), config.embedding_dimension):
        raise ValueError("Cannot activate empty/inconsistent snapshot")
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path / "qdrant"))
    try:
        if not client.collection_exists("chunks"):
            client.create_collection("chunks", vectors_config=models.VectorParams(
                size=config.embedding_dimension, distance=models.Distance.COSINE))
        for start in range(0, len(chunks), 128):
            points = [models.PointStruct(id=c.chunk_id, vector=v.tolist(), payload=c.to_dict())
                      for c, v in zip(chunks[start:start + 128], vectors[start:start + 128])]
            client.upsert("chunks", points=points, wait=True)
        if client.count("chunks", exact=True).count != len(chunks):
            raise ValueError("Snapshot contains unexpected/orphan points")
    finally:
        client.close()
    write_jsonl(path / "papers.jsonl", (p.to_dict() for p in papers))
    write_jsonl(path / "chunks.jsonl", (c.to_dict() for c in chunks))
    np.save(path / "vectors.npy", vectors.astype(np.float32), allow_pickle=False)
    write_jsonl(path / "lexical.jsonl", ({"chunk_id": c.chunk_id, "text_hash": fingerprint(c.text)} for c in chunks))
    state = {"status": "ready", "papers": len(papers), "chunks": len(chunks),
             "config": asdict(config), "chunk_ids_hash": fingerprint([c.chunk_id for c in chunks]),
             "vectors_sha256": hashlib.sha256((path / "vectors.npy").read_bytes()).hexdigest()}
    atomic_json(path / "ready.json", state)
    return state
