from __future__ import annotations

import uuid
from .config import Config
from .schemas import Chunk, Page, Paper


def chunk_page(page: Page, paper: Paper, tokenizer, config: Config, index_id: str) -> list[Chunk]:
    if page.status != "ready" or not page.text.strip():
        return []
    data = tokenizer(page.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
    offsets = data["offset_mapping"]
    result = []
    start = 0
    while start < len(offsets):
        end = min(start + config.chunk_tokens, len(offsets))
        # Prefer a nearby natural boundary, while retaining enough new content.
        if end < len(offsets):
            lower = start + max(config.chunk_overlap + 1, config.chunk_tokens * 2 // 3)
            for candidate in range(end, lower, -1):
                gap = page.text[offsets[candidate - 1][1]:offsets[candidate][0]]
                before = page.text[max(0, offsets[candidate - 1][1] - 2):offsets[candidate - 1][1]]
                if "\n" in gap or (before.endswith((".", "?", "!")) and gap):
                    end = candidate
                    break
        a, b = offsets[start][0], offsets[end - 1][1]
        text = page.text[a:b]
        key = f"{paper.pdf_sha256}:{page.page_number}:{a}:{b}:{index_id}"
        result.append(Chunk(str(uuid.uuid5(uuid.NAMESPACE_URL, key)), paper.paper_id, paper.arxiv_id,
                            paper.title, paper.authors, paper.categories, paper.first_submitted_year,
                            paper.arxiv_url, paper.pdf_url, paper.pdf_sha256, page.page_number,
                            a, b, text, end - start, index_id, config.corpus_id))
        if end == len(offsets):
            break
        start = max(start + 1, end - config.chunk_overlap)
    return result
