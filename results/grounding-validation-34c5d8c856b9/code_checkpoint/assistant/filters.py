from __future__ import annotations

import unicodedata
from .schemas import Paper, RetrievalFilters


def normalize_name(name: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def eligible_papers(papers: list[Paper], filters: RetrievalFilters) -> tuple[list[Paper], dict]:
    known = {p.paper_id for p in papers}
    unknown_ids = set(filters.paper_ids) - known
    if unknown_ids:
        raise ValueError(f"Selected paper IDs are not in this corpus: {sorted(unknown_ids)}")
    selected, unknown_metadata = [], 0
    wanted_authors = {normalize_name(a) for a in filters.authors}
    for paper in papers:
        if filters.paper_ids and paper.paper_id not in filters.paper_ids:
            continue
        year = paper.first_submitted_year
        if filters.year_min is not None or filters.year_max is not None:
            if year is None:
                unknown_metadata += 1
                continue
            if filters.year_min is not None and year < filters.year_min:
                continue
            if filters.year_max is not None and year > filters.year_max:
                continue
        if filters.categories:
            if not paper.categories:
                unknown_metadata += 1
                continue
            if not set(filters.categories).intersection(paper.categories):
                continue
        if wanted_authors:
            if not paper.authors:
                unknown_metadata += 1
                continue
            if not wanted_authors.intersection(normalize_name(a) for a in paper.authors):
                continue
        selected.append(paper)
    return selected, {"total_papers": len(papers), "eligible_papers": len(selected),
                      "excluded_unknown_metadata": unknown_metadata,
                      "eligible_paper_ids": [p.paper_id for p in selected]}
