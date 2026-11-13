# Implementation contributions

The contribution is the complete Local Research Assistant application and its experimental evaluation. It spans the acquisition, indexing, retrieval, answer and interface layers. Individual functions are implementation examples within that system, rather than the full contribution.

## Module coverage

| Area | Source modules | Implemented behavior |
|---|---|---|
| Paper acquisition | `arxiv_api.py`, `discovery.py`, `downloads.py` | Official metadata/PDF acquisition, ranked discovery, bounded requests and download verification. |
| Source-aware indexing | `extraction.py`, `chunking.py`, `storage.py`, `service.py` | Pagewise extraction, page-local chunks, provenance, versioned snapshots and embedding reuse. |
| Constrained retrieval | `filters.py`, `lexical.py`, `embeddings.py`, `retrieval.py`, `reranking.py`, `context.py` | Metadata eligibility, BM25/dense candidates, RRF fusion, cross-encoder reranking and context budgets. |
| Source-based answers | `generation.py`, `evidence_selection.py`, `grounding.py`, `citations.py` | Structured evidence IDs, quotation matching, support checks, pruning, refusals and citation rendering. |
| Application integration | `service.py`, `ui.py`, `cli.py`, `app.py`, `enhanced_app.py` | Shared backend, persistent library, interactive interface and command-line access. |
| Configuration and observability | `config.py`, `schemas.py`, `artifacts.py`, `telemetry.py` | Typed records, consistent settings, saved artifacts and stage-level traces. |
| Experimental evaluation | `evaluation/`, `tests/`, `scripts/` | Frozen questions, evidence/reference checks, retrieval comparisons, metadata ablation, answer-policy comparison, metrics and local evaluator integration. |

Core modules above are in `assistant/` unless another directory or root entry point is named.

## Representative integration points

- `AssistantService.ingest` connects extraction, chunking, embeddings, reusable snapshots and index activation.
- `AssistantService.retrieve` applies metadata eligibility before retrieval, combines branch ranks and invokes reranking.
- `AssistantService._extractive_answer` connects evidence-unit ranking, model selection, verification, pruning and source rendering.
- `AssistantService.ask` connects retrieval, context packing, answering, citations and trace recording.
- The interface and CLI share this service; evaluation operates on the same pipeline components.

The presentation's code slide contains selected statements from the ingestion, retrieval and answer orchestration methods, with surrounding control flow omitted. The repository contains their complete definitions and the remaining modules. The report describes the implementation as a complete application.

## Adopted components

BM25 and RRF are established retrieval methods. Qdrant, Gradio, pypdf, pretrained MiniLM models, Qwen2.5 and Ragas are adopted tools or models. The project contribution is the implemented system, its source/provenance handling, component integration and evaluation; it does not include inventing those algorithms or training the pretrained models.
