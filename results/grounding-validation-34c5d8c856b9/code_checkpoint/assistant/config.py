from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    corpus_id: str = "course_rag_v1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    grounding_enabled: bool = True
    grounding_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    grounding_revision: str = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
    grounding_entailment_threshold: float = .70
    grounding_contradiction_threshold: float = .50
    generator_model: str = "qwen2.5:3b"
    embedding_dimension: int = 384
    embedding_batch_size: int = 16
    reranker_batch_size: int = 8
    chunk_tokens: int = 180
    chunk_overlap: int = 30
    bm25_k1: float = 1.5
    bm25_b: float = .75
    branch_top_k: int = 20
    rrf_constant: int = 60
    rerank_top_n: int = 20
    context_chunks: int = 4
    context_window: int = 4096
    prompt_tokens: int = 3072
    output_tokens: int = 512
    safety_tokens: int = 128
    temperature: float = 0.0
    seed: int = 42
    cpu_threads: int = 4
    extractor: str = "pypdf"
    ollama_url: str = "http://127.0.0.1:11434"
    download_attempts: int = 3
    arxiv_delay: float = 3.0

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        data = json.loads((path or PROJECT_ROOT / "configs/default.json").read_text(encoding="utf-8"))
        unknown = data.keys() - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unsupported configuration keys: {sorted(unknown)}")
        result = cls(**data)
        if not 0 <= result.chunk_overlap < result.chunk_tokens <= 254:
            raise ValueError("Chunk token/overlap configuration exceeds the encoder budget")
        if result.prompt_tokens + result.output_tokens + result.safety_tokens > result.context_window:
            raise ValueError("Prompt/output/reserve exceeds the configured model window")
        if result.branch_top_k < 1 or result.rerank_top_n < 1 or result.context_chunks < 1:
            raise ValueError("Retrieval/context limits must be positive")
        if type(result.grounding_enabled) is not bool or not .5 < result.grounding_entailment_threshold < 1:
            raise ValueError("Grounding requires a boolean switch and an entailment threshold in (.5,1)")
        if not .5 <= result.grounding_contradiction_threshold < 1:
            raise ValueError('Grounding contradiction threshold must be in [.5,1)')
        return result

    def identity(self) -> str:
        return fingerprint(asdict(self))


def configure_cache(root: Path = PROJECT_ROOT) -> None:
    # Project-local caches avoid altering an unrelated global model installation.
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
