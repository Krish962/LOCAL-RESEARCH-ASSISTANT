from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

SAFE_CORPUS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


@dataclass
class Paper:
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    first_submitted_year: int | None = None
    arxiv_url: str = ""
    pdf_url: str = ""
    doi: str | None = None
    journal_ref: str | None = None
    pdf_path: str = ""
    pdf_sha256: str = ""
    metadata_source: str = "arxiv_api"

    @classmethod
    def from_dict(cls, value: dict) -> "Paper":
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Page:
    paper_id: str
    page_number: int
    text: str
    raw_text: str
    extractor: str
    status: str = "ready"
    error: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    first_submitted_year: int | None
    arxiv_url: str
    pdf_url: str
    pdf_sha256: str
    page_number: int
    start_char: int
    end_char: int
    text: str
    token_count: int
    index_config_id: str
    corpus_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalFilters:
    paper_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    year_min: int | None = None
    year_max: int | None = None

    def __post_init__(self):
        for name in ("paper_ids", "categories", "authors"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in value):
                raise ValueError(f"{name} must contain nonempty strings")
            object.__setattr__(self, name, tuple(dict.fromkeys(v.strip() for v in value)))
        for year in (self.year_min, self.year_max):
            if year is not None and (type(year) is not int or not 1900 <= year <= 2100):
                raise ValueError("Year bounds must be integer years between 1900 and 2100")
        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError("Minimum year exceeds maximum year")

    @classmethod
    def from_dict(cls, data: dict | None) -> "RetrievalFilters":
        return cls(**(data or {}))


@dataclass
class Hit:
    chunk: Chunk
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"chunk": self.chunk.to_dict(), "scores": dict(self.scores), "ranks": dict(self.ranks)}


def validate_corpus_id(value: str) -> str:
    if not SAFE_CORPUS.fullmatch(value):
        raise ValueError("Corpus ID must use letters, digits, underscore or hyphen")
    return value
