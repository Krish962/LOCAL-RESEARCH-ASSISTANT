from __future__ import annotations

from .schemas import Hit


def reciprocal_rank_fusion(branches: dict[str, list[Hit]], constant=60) -> list[Hit]:
    merged = {}
    for name, hits in branches.items():
        seen = set()
        for rank, hit in enumerate(hits, 1):
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            item = merged.setdefault(key, Hit(hit.chunk))
            item.scores.update(hit.scores)
            item.ranks[name] = rank
            item.scores["rrf"] = item.scores.get("rrf", 0.) + 1 / (constant + rank)
    ordered = sorted(merged.values(), key=lambda h: (-h.scores["rrf"], h.chunk.chunk_id))
    for rank, hit in enumerate(ordered, 1):
        hit.ranks["rrf"] = rank
    return ordered
