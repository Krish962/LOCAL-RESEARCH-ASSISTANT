from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .artifacts import atomic_json
from .config import Config, fingerprint
from .filters import eligible_papers
from .lexical import tokenize
from .schemas import Paper, RetrievalFilters


def canonical_id(value: str) -> str:
    return re.sub(r"v\d+$", "", value.rsplit("/abs/", 1)[-1])


def from_arxiv(result) -> Paper:
    identifier = result.get_short_id()
    return Paper(canonical_id(identifier), identifier, " ".join(result.title.split()),
                 [a.name for a in result.authors], result.summary, result.categories,
                 result.primary_category, result.published.isoformat(), result.updated.isoformat(),
                 result.published.year, result.entry_id.replace("http://", "https://"),
                 result.pdf_url.replace("http://", "https://"), result.doi, result.journal_ref)


class Discovery:
    def __init__(self, root: Path, config: Config, encoder, reranker):
        import arxiv
        self.root, self.config = root, config
        self.encoder, self.reranker = encoder, reranker
        self.client = arxiv.Client(page_size=40, delay_seconds=max(3., config.arxiv_delay), num_retries=2)

    def fetch_ids(self, ids: list[str]) -> list[Paper]:
        from .arxiv_api import fetch
        return fetch(ids=ids,limit=len(ids))

    def search(self, query: str, filters: RetrievalFilters, limit=40, refresh=False, ranking_text: str | None = None) -> dict:
        import arxiv
        query = query.strip()
        if not query or len(query) > 1500:
            raise ValueError("Search query must contain 1-1500 characters")
        if not 1 <= limit <= 150:
            raise ValueError("Candidate count must be between 1 and 150")
        if filters.categories:
            query = f"({query}) AND (" + " OR ".join("cat:"+c for c in filters.categories) + ")"
        if filters.year_min is not None or filters.year_max is not None:
            first, last = filters.year_min or 1900, filters.year_max or 2100
            query = f"({query}) AND submittedDate:[{first}01010000 TO {last}12312359]"
        cache = self.root / "data/discovery" / (fingerprint({"query":query,"limit":limit}) + ".json")
        if cache.exists() and not refresh:
            record = json.loads(cache.read_text(encoding="utf-8"))
            papers = [Paper.from_dict(row) for row in record["papers"]]
            cached = True
        else:
            from .arxiv_api import fetch
            rows = fetch(query=query,limit=limit)
            deduplicated = {}
            for result in rows:
                paper = result
                deduplicated.setdefault(paper.paper_id, paper)
            papers = list(deduplicated.values())
            atomic_json(cache, {"query":query,"fetched_at":time.time(),"papers":[p.to_dict() for p in papers]})
            cached = False
        eligible, stats = eligible_papers(papers, filters)
        rank_question = ranking_text or re.sub(r'\b(?:ti|abs|cat|au|all):|\b(?:AND|OR|ANDNOT)\b|[()\[\]"]',' ',query)
        ranked, ranking = self.rank(rank_question.strip(), eligible)
        return {"query":query,"ranking_text":rank_question,"cache_hit":cached,"stats":stats,
                "arxiv_order":[p.paper_id for p in papers], "ranking":ranking,
                "papers":[p.to_dict() for p in ranked]}

    def rank(self, question: str, papers: list[Paper]) -> tuple[list[Paper], dict]:
        import numpy as np
        from rank_bm25 import BM25Okapi
        if not papers:
            return [], {"reranker_executed":False,"reason":"no_eligible_papers"}
        if self.encoder.count(question) > 128:
            raise ValueError("Shorten the ranking question to at most 126 encoder wordpieces")
        query_terms = tokenize(question)
        lexical = [tokenize(p.title + " " + p.title + " " + p.abstract) for p in papers]
        bm25_scores = BM25Okapi(lexical, k1=self.config.bm25_k1, b=self.config.bm25_b).get_scores(query_terms)
        bm25_order = sorted((i for i, row in enumerate(lexical) if set(row).intersection(query_terms)),
                            key=lambda i:(-float(bm25_scores[i]), papers[i].paper_id))
        semantic_texts = [p.title + "\n" + p.abstract for p in papers]
        vectors = self.encoder.encode_documents(semantic_texts)
        question_vector = self.encoder.encode_documents([question])[0]
        dense_scores = vectors @ question_vector
        dense_order = sorted(range(len(papers)), key=lambda i:(-float(dense_scores[i]), papers[i].paper_id))
        scores = {}
        for order in (bm25_order, dense_order):
            for rank, i in enumerate(order, 1):
                scores[i] = scores.get(i, 0.) + 1 / (self.config.rrf_constant + rank)
        fused = sorted(scores, key=lambda i:(-scores[i], papers[i].paper_id))
        shortlist = fused[:30]
        pair_texts, owners = [], []
        for i in shortlist:
            for text in self.encoder.segments(semantic_texts[i], 180):
                pair_texts.append(text)
                owners.append(i)
        # Retrieval rank query can contain arXiv syntax; it remains literal text, never code.
        pair_scores, pair_lengths = self.reranker.scores(question, pair_texts)
        reranked_scores = {}
        for i, score in zip(owners, pair_scores):
            reranked_scores[i] = max(reranked_scores.get(i, -float("inf")), score)
        final = sorted(shortlist, key=lambda i:(-reranked_scores[i], papers[i].paper_id))
        details = {"bm25_order":[papers[i].paper_id for i in bm25_order],
                   "dense_order":[papers[i].paper_id for i in dense_order],
                   "fused_order":[papers[i].paper_id for i in fused],
                   "reranked_order":[papers[i].paper_id for i in final],
                   "reranker_executed":True,"pair_lengths":pair_lengths,
                   "scores":{papers[i].paper_id:{"bm25":float(bm25_scores[i]),
                              "dense":float(dense_scores[i]),"rrf":scores.get(i,0.),
                              "cross_encoder":reranked_scores.get(i)} for i in range(len(papers))}}
        return [papers[i] for i in final], details
