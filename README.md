# Local Research Assistant

A course project by **Group 29, IIT Kharagpur**, for Foundations of Large Language Models (AI60213).

| Team member | Roll number |
|---|---|
| Krish | 22AG3AI07 |
| Prajojeet Pradhan | 22CE3AI15 |
| Aaditya Saraf | 22NA3AI40 |
| Pranjul Shukla | 22NA3AI42 |

The application searches research papers and returns selected source excerpts with paper and PDF-page citations. It runs locally using pretrained models. The indexed course collection contains **20 arXiv papers, 312 pages and 2,467 text chunks**.

## Pipeline

1. Paper acquisition from the official arXiv API.
2. Pagewise pypdf extraction and page-local chunking.
3. Persistent Qdrant storage with paper metadata, source spans and embeddings.
4. Paper, author, category and first-submission-year filtering before retrieval.
5. BM25 and MiniLM dense retrieval combined through reciprocal rank fusion.
6. MiniLM cross-encoder reranking and context packing of up to four passages.
7. Qwen2.5 3B evidence selection, source verification and deterministic citation rendering.

Adding papers and answering questions are separate operations. Repeated questions reuse the stored index. Empty scopes and insufficient evidence produce explicit refusal messages.

## Application interface

The application entry point uses the local environment, indexed corpus and downloaded models. Commands below are relative to the repository root:

```cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File scripts\start_ollama.ps1
.venv311\Scripts\python.exe -X utf8 app.py
```

The Gradio interface is served at http://127.0.0.1:7860. Both `app.py` and `enhanced_app.py` use the same backend.

Individual commands:

```cmd
.venv311\Scripts\python.exe -X utf8 -m assistant.cli library
.venv311\Scripts\python.exe -X utf8 -m assistant.cli ask "How does HyDE construct a retrieval query?" --paper 2212.10496 --mode hybrid_rerank
.venv311\Scripts\python.exe -X utf8 -m assistant.cli search "ti:retrieval" --limit 20
```

Paper discovery requires internet access. Question answering uses the indexed documents and cached local models. Interface requests and evaluations share the local index and model server, so their execution is sequential.

## Models and settings

| Component | Setting |
|---|---|
| Runtime | Python 3.11.18, CPU PyTorch, local Ollama |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Evidence selector | `qwen2.5:3b` |
| Support checker | `cross-encoder/nli-MiniLM2-L6-H768` |
| Extraction | pypdf by default; MarkItDown optional |
| Chunks | 180 encoder tokens with 30-token overlap; page-local |
| Retrieval | 20 candidates per branch; RRF constant 60; rerank top 20 |
| Context | At most four passages; 4,096-token context window |
| Supplementary evaluator | Local `llama3.2:3b`, Ragas 0.4.3 |

The configuration is in `configs/default.json`. Runtime requirements are pinned with hashes. The setup entry point is `scripts/clean_setup.ps1`. Model weights and corpus files are separate runtime dependencies.

## Saved evaluation

The final set has 40 questions: 20 single-paper questions, eight comparisons, eight missing-evidence cases and four empty scopes. Results are in `results/grounding-validation-abad545bfc7a`.

| Measure | Source-based system |
|---|---:|
| Answerable cases answered | 22/28 |
| Single-paper cases answered | 19/20 |
| Comparisons answered | 3/8 |
| Missing-evidence cases correctly refused | 8/8 |
| Empty scopes correctly refused | 4/4 |
| Runtime failures | 0 |

Answer coverage is **78.6%, not accuracy**. Automated source-reference review classified the 22 answers as seven complete, eight partial and seven wrong or unresponsive. These judgments have not received independent human validation. All 27 copied quotation spans matched their source pages; this establishes literal provenance, not answer relevance.

The earlier retrieval diagnostic covers six graded questions. Its nDCG@4 values are 0.613 for BM25, 0.613 for dense retrieval, 0.733 for hybrid retrieval and 0.866 for hybrid retrieval with reranking. These results have a different scope from the final answering comparison.

Evaluation entry points:

```cmd
scripts\validate_40.cmd -CheckOnly
scripts\validate_40.cmd
```

`-CheckOnly` validates frozen data without inference. The full command runs or resumes the CPU comparison. Supplementary Ragas scores concern an earlier baseline sample and do not establish the final system's accuracy.

## Project files

- `assistant/`: ingestion, storage, retrieval, evidence selection, verification and interface.
- `evaluation/`: frozen questions, references, metrics and evaluation tools.
- `scripts/`: setup and command-line launchers.
- `tests/`: code and data-integrity tests.
- `configs/`: model and pipeline settings.
- `data/`: corpus manifests, downloads and persistent indexes.
- `results/`: saved measurements, answers and experiment checkpoints.
- `docs/baseline/`: fixed development snapshots used for reproducibility and comparisons.

The documentation includes [implementation contributions](docs/CONTRIBUTIONS.md), [evaluation protocol](docs/VALIDATION_40.md), [findings](docs/RUN_FINDINGS.md) and [project summary](docs/FINAL_STATUS.md).

## Submission files

The named [report](deliverables/Group_29_Report.pdf), [presentation](deliverables/Group_29_Presentation.pptx) and [editable LaTeX source](deliverables/Group_29_LaTeX_Source.zip) are in `deliverables/`.

Model weights, Python environments, downloaded PDFs, generated indexes and per-question runtime traces stay local and are excluded from Git. A new machine needs the dependencies, models and paper collection before running the application or full evaluation. Saved benchmark answers and measurements remain included.


## Limits and attribution

The collection and diagnostic samples are small. Comparisons, requested-paper attribution, tables and equations remain difficult. A genuine quotation can still be unrelated to the question. Independent human validation is not available for the reported review labels.

The project uses existing methods and pretrained models. References include [RAG](https://arxiv.org/abs/2005.11401), [reciprocal rank fusion](https://doi.org/10.1145/1571941.1572114), [MiniLM](https://arxiv.org/abs/2002.10957), [Qwen2.5](https://arxiv.org/abs/2412.15115) and [Ragas](https://arxiv.org/abs/2309.15217). Their methods, models and licenses retain their original attribution.
