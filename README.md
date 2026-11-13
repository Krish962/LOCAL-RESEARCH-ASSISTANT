# Local Research Assistant

A Foundations of Large Language Models (AI60213) course project by **Group 29, IIT Kharagpur**.

| Team member | Roll number |
|---|---|
| Krish | 22AG3AI07 |
| Prajojeet Pradhan | 22CE3AI15 |
| Aaditya Saraf | 22NA3AI40 |
| Pranjul Shukla | 22NA3AI42 |

The application finds research papers, stores their contents and returns selected source excerpts with paper and PDF-page citations. The reported collection contains **20 arXiv papers, 312 pages and 2,467 chunks**.

## Pipeline and contribution

The contribution is the complete Python application and experimental evaluation. Paper acquisition, pagewise extraction, persistent indexing, metadata filtering, hybrid retrieval, reranking, evidence selection, support checks and citation rendering share one backend.

| Component | Implementation |
|---|---|
| Paper discovery and download | Official arXiv API, ranked titles/abstracts, verified PDF downloads |
| Extraction and storage | Pagewise pypdf, 180-token chunks with 30-token overlap, page/span metadata, reusable Qdrant snapshots |
| Metadata filtering | Paper ID, author, category and first-submission year, applied before both search branches |
| Hybrid retrieval | BM25 and 384-dimensional MiniLM embeddings, 20 candidates per branch, RRF constant 60 |
| Reranking and context | MiniLM cross-encoder, top 20 candidates, at most four passages |
| Answering | Local Qwen2.5 3B selects excerpt IDs; Python verifies and renders source quotations |
| Evidence checks | Identifier/unit checks, exact source spans, local MiniLM NLI, pruning and bounded rechecks |
| Interfaces and evaluation | Gradio page, CLI, retrieval comparisons, metadata ablation, final answering comparison and supplementary Ragas |

Adding papers and asking questions are separate operations. Questions reuse the saved index. Empty scopes and insufficient evidence produce explicit refusal responses.

## Setup and application

Commands are relative to the repository root and use Windows Command Prompt.

Runtime setup requires an existing Python installation to bootstrap the project-local tools:

~~~cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File scripts\clean_setup.ps1
~~~

[Ollama](https://ollama.com/) supplies the local model server. The startup script accepts either an installed Ollama executable or the project-local portable runtime.

~~~cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File scripts\start_ollama.ps1
ollama pull qwen2.5:3b
.venv311\Scripts\python.exe -B -X utf8 -m scripts.prepare_grounding
~~~

For a portable-only Ollama installation, the model-download command uses .runtime\ollama\ollama.exe in place of ollama.

The embedding, reranking and generator-tokenizer files download on their first use and are cached under the project. Paper discovery and initial downloads require internet access.

The fixed course collection can be downloaded from its versioned source manifest and indexed with:

~~~cmd
.venv311\Scripts\python.exe -B -X utf8 -m assistant.cli ingest data\corpora\course_rag_v1.source.json --download
.venv311\Scripts\python.exe -B -X utf8 app.py
~~~

The page is served at **http://127.0.0.1:7860**. Additional papers can be searched and indexed through the page. Application requests and evaluations run sequentially against the same local index and model server.

Other interface commands:

~~~cmd
.venv311\Scripts\python.exe -B -X utf8 -m assistant.cli library
.venv311\Scripts\python.exe -B -X utf8 -m assistant.cli ask "How does HyDE construct a retrieval query?" --paper 2212.10496 --mode hybrid_rerank
.venv311\Scripts\python.exe -B -X utf8 -m assistant.cli search "ti:retrieval" --limit 20
~~~

## Models and configuration

Configuration is stored in configs/default.json. Inference runs on CPU without model training.

| Role | Model |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Evidence selection | qwen2.5:3b |
| Support checking | cross-encoder/nli-MiniLM2-L6-H768 |
| Optional evaluation judge | llama3.2:3b |

The generator context window is 4,096 tokens, with a 3,072-token prompt limit, up to 512 output tokens and a safety reserve. The support model uses an entailment threshold of 0.70 and a contradiction threshold of 0.50. These thresholds are uncalibrated.

## Evaluation and reported results

Recorded inputs, outputs and measurements are retained for the experiments in the report and presentation. Recalculation from saved records requires no new model answers:

~~~cmd
.venv311\Scripts\python.exe -B -X utf8 -m evaluation.summarize
~~~

The final set has 40 questions: 20 single-paper cases, eight comparisons, eight missing-evidence cases and four empty scopes. The saved run is results/grounding-validation-abad545bfc7a.

| Measure | Generation baseline | Source-based system |
|---|---:|---:|
| Answerable cases answered | 27/28 | 22/28 |
| Single-paper cases answered | 20/20 | 19/20 |
| Comparisons answered | 7/8 | 3/8 |
| Missing-evidence cases refused | 0/8 | 8/8 |
| Empty scopes correctly refused | 4/4 | 4/4 |
| Runtime / invalid-output failures | 1 | 0 |
| Mean seconds per question | 27.10 | 33.79 |

**Answer coverage is separate from accuracy.** Automated source-reference review classified the 22 source-based answers as seven complete, eight partial and seven wrong or unresponsive. These labels have no independent human validation. All 27 copied quotation spans matched source pages; literal matching does not establish answer relevance.

The generation baseline backend is retained under evaluation/baselines/generation because it is required for the reported comparison. Its source inventory verifies the retained modules. It is separate from the application backend.

Source-data verification and a fresh/resumed 40-question comparison:

~~~cmd
scripts\validate_40.cmd -CheckOnly
scripts\validate_40.cmd
~~~

The check-only command verifies the frozen dataset and indexed PDF spans without inference. The full command runs 80 question/arm pairs using the same hybrid retrieval and reranking. Configuration and source fingerprints distinguish new runs from the recorded experiment.

Earlier retrieval and filtering experiments:

~~~cmd
.venv311\Scripts\python.exe -B -X utf8 -m evaluation.run_retrieval --split all --output results\retrieval-reproduction
.venv311\Scripts\python.exe -B -X utf8 -m evaluation.run_filter_ablation
~~~

The earlier nDCG@4 diagnostic uses six graded questions: BM25 0.613, dense 0.613, hybrid 0.733 and hybrid with reranking 0.866. Annotated-span recall uses a different earlier answerable set. Disabling metadata filters returned 64 out-of-scope passages among 144; enabling filters returned zero among 96 across 72 records.

Optional Ragas setup uses a separate environment:

~~~cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File scripts\clean_setup.ps1 -IncludeEvaluation
ollama pull llama3.2:3b
.venv-eval311\Scripts\python.exe -B -X utf8 -m evaluation.run_ragas --answers results\ragas-857109279246\inputs.jsonl --limit 10 --max-seconds 1800 --output results\ragas-reproduction
~~~

The reported supplementary Ragas run scored earlier generation-baseline answers. Of 40 planned metric calls, 25 completed, four failed and 11 were not attempted within 1,800 seconds. These scores do not establish the final system's accuracy.

Optional extraction comparison:

~~~cmd
powershell.exe -NoLogo -NoProfile -ExecutionPolicy RemoteSigned -File scripts\clean_setup.ps1 -IncludeExtraction
.venv311\Scripts\python.exe -B -X utf8 -m scripts.extraction_pilot
~~~

This six-page comparison requires Poppler's pdftoppm on PATH. MarkItDown is optional; the sampled two-column text favored pypdf. The extracted text, page previews, measurements and source-review record are in results/extraction-pilot.

Reproduction commands write separate retrieval, filtering, Ragas and extraction output folders. They preserve the recorded experiment files.

## Repository and deliverables

| Location | Contents |
|---|---|
| app.py and assistant/ | Application entry point, shared backend and CLI |
| configs/ | Pipeline settings |
| scripts/ | Setup, model preparation, extraction comparison and evaluation launcher |
| evaluation/ | Experiment code, frozen questions, relevance labels and generation baseline |
| data/ | Course-paper source and corpus manifests |
| results/ | Recorded evidence behind reported experiments |
| deliverables/ | Report, presentation and editable LaTeX source |

Model weights, Python environments, downloaded PDFs, generated indexes, runtime traces and evaluator caches remain local and are excluded from Git. A fresh installation downloads the models and paper collection before running inference or source-span checks. Rebuilding with different PDF/model revisions can change source spans and outcomes.

The submission artifacts are the [report](deliverables/Group_29_Report.pdf), [presentation](deliverables/Group_29_Presentation.pptx) and [LaTeX source](deliverables/Group_29_LaTeX_Source.zip).

The report source uses the official ICLR 2025 style. Its main file is report.tex, compiled with pdfLaTeX and BibTeX or imported into Overleaf.

## Limitations and attribution

The corpus and diagnostic samples are small. Comparisons, requested-paper attribution, answer completeness, tables, equations and scanned pages remain difficult. A genuine quotation can be unrelated to a question. The evaluation lacks independent human validation, and the answer-policy comparison changes several components together.

The project uses published methods and pretrained models, including [RAG](https://arxiv.org/abs/2005.11401), [RRF](https://doi.org/10.1145/1571941.1572114), [MiniLM](https://arxiv.org/abs/2002.10957), [Qwen2.5](https://arxiv.org/abs/2412.15115) and [Ragas](https://arxiv.org/abs/2309.15217). Their methods, weights and licenses retain their original attribution.
