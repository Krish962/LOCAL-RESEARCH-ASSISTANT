# Project summary

Local Research Assistant is the Group 29 course project for AI60213 at IIT Kharagpur.

## Implementation scope

The implementation covers paper discovery and ingestion, page/span provenance, persistent indexing, metadata filtering, hybrid retrieval, reranking, evidence selection and verification, citation rendering, Gradio/CLI access, and experimental evaluation. The [contribution map](CONTRIBUTIONS.md) connects these capabilities to source files and entry points.

The project contains 20 indexed papers, 312 pages and 2,467 chunks. Pretrained MiniLM models and local Qwen2.5 3B provide text representations, ranking, evidence selection and support checks. No model training is part of the project.

## Experimental evidence

The final comparison is `results/grounding-validation-abad545bfc7a`. The source-based arm answered 22/28 answerable cases, refused all eight missing-evidence cases and four empty scopes, and had zero runtime failures.

Automated source-reference review classified the 22 answers as seven complete, eight partial and seven wrong or unresponsive. All 27 selected quotation spans matched their source pages. Answer coverage, literal provenance and semantic correctness are separate measures. Independent human validation is unavailable.

## Deliverables

The `deliverables/` directory contains the four-page ICLR-format report, six-slide presentation and editable LaTeX source. Both documents identify Group 29, all four team members and IIT Kharagpur.

The repository includes application code, tests, frozen evaluation data, baseline dependencies and recorded measurements. Environments, model weights, downloaded PDFs, generated indexes and per-question runtime traces are local runtime artifacts excluded from Git.
