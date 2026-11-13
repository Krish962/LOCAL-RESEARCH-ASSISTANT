from __future__ import annotations

from .config import Config
from .generation import messages
from .schemas import Hit


def source(hit: Hit, index: int) -> dict:
    chunk = hit.chunk
    return {"label":f"S{index}","chunk_id":chunk.chunk_id,"paper_id":chunk.paper_id,
            "arxiv_id":chunk.arxiv_id,"title":chunk.title,"page_number":chunk.page_number,
            "arxiv_url":chunk.arxiv_url,"pdf_url":chunk.pdf_url,"text":chunk.text,
            "start_char":chunk.start_char,"end_char":chunk.end_char,
            "scores":hit.scores,"ranks":hit.ranks}


def overlap(a: Hit, b: Hit) -> float:
    x, y = a.chunk, b.chunk
    if (x.paper_id, x.page_number) != (y.paper_id, y.page_number):
        return 0.
    intersection = max(0, min(x.end_char, y.end_char) - max(x.start_char, y.start_char))
    return intersection / max(1, min(x.end_char - x.start_char, y.end_char - y.start_char))


def pack_context(question: str, hits: list[Hit], counter, config: Config, required_papers=()) -> tuple[list[dict], dict]:
    ordered = []
    for paper_id in required_papers:
        match = next((h for h in hits if h.chunk.paper_id == paper_id), None)
        if match is not None and match not in ordered:
            ordered.append(match)
    ordered.extend(h for h in hits if h not in ordered)
    chosen, dropped = [], []
    for hit in ordered:
        if len(chosen) >= config.context_chunks:
            dropped.append({"chunk_id":hit.chunk.chunk_id,"reason":"chunk_limit"})
            continue
        if any(hit.chunk.chunk_id == old.chunk.chunk_id or overlap(hit, old) >= .8 for old in chosen):
            dropped.append({"chunk_id":hit.chunk.chunk_id,"reason":"duplicate_span"})
            continue
        tentative = [source(h, i) for i, h in enumerate(chosen + [hit], 1)]
        tokens = counter(messages(question, tentative))
        # Leave room inside prompt ceiling for one concise repair instruction.
        if tokens > config.prompt_tokens - 96:
            dropped.append({"chunk_id":hit.chunk.chunk_id,"reason":"token_budget"})
            continue
        chosen.append(hit)
    sources = [source(h, i) for i, h in enumerate(chosen, 1)]
    present = {s["paper_id"] for s in sources}
    return sources, {"prompt_tokens":counter(messages(question, sources)),"dropped":dropped,
                     "missing_required_papers":[p for p in required_papers if p not in present],
                     "selected_chunks":len(sources),"required_papers":list(required_papers)}
