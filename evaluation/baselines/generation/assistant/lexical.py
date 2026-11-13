from __future__ import annotations

import re
import unicodedata

from .schemas import Chunk, Hit


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+(?:[.+-]\w+)*", unicodedata.normalize("NFKC", text).casefold())


class LexicalIndex:
    def __init__(self, chunks: list[Chunk], k1=1.5, b=.75):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self.token_sets = [set(row) for row in self.tokens]
        self.index = BM25Okapi(self.tokens, k1=k1, b=b) if chunks else None

    def search(self, question: str, eligible_ids: set[str], limit: int) -> list[Hit]:
        query = tokenize(question)
        if self.index is None or not query or not eligible_ids:
            return []
        scores = self.index.get_scores(query)
        terms = set(query)
        rows = [i for i, c in enumerate(self.chunks)
                if c.paper_id in eligible_ids and terms.intersection(self.token_sets[i])]
        rows.sort(key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        return [Hit(self.chunks[i], {"bm25": float(scores[i])}, {"bm25": rank})
                for rank, i in enumerate(rows[:limit], 1)]
