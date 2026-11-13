# Frozen 40-question validation

The validation corpus contains 20 papers, 312 extracted pages and 2,467 indexed chunks.

| Case type | Count |
|---|---:|
| Single-paper answerable | 20 |
| Two-paper comparison | 8 |
| Missing evidence | 8 |
| Empty metadata scope | 4 |
| Total | 40 |

All 20 indexed papers have one single-paper question. Six of these search the whole library, four use valid metadata-only restrictions, and ten explicitly select a paper. The metadata cases cover category, year and author filters. RAGTruth correctly uses its first arXiv submission year, 2023, rather than inferring a year from its 2401 identifier.

Each answerable case has a paraphrased reference and supporting quotations tied to real PDF-page offsets and frozen chunk IDs. The dataset has 39 evidence groups and 49 known supporting alternatives. Quotations are not exhaustive semantic relevance judgments. Labels were checked through automated source-reference review; independent human validation is pending.

Negative controls include four absent-model requests, two unreported measurements and two unreported statistical results. One deliberately difficult case distinguishes the SPLADE paper's ACM publication-price boilerplate from a model-training expense. The observed price mention is retained in the audit rather than silently discarded.

## Evaluation entry points

```cmd
scripts\validate_40.cmd
```

This verifies frozen source spans, runs code/data tests, verifies the local NLI cache, and sequentially evaluates the preserved pre-grounding pipeline and the current pipeline on the same 40 questions. That is 80 case/arm pairs; empty scopes and some missing-evidence cases require no language-model generation. Matching saved rows are reused after interruption. Code/config/model/question changes produce a different run directory.

The generation baseline uses the same hybrid-reranked retrieval and the development snapshot in `docs/baseline/pre_grounding`. It differs from the source-based arm in answer generation and verification policy.

A read-only preflight is available:

```cmd
scripts\validate_40.cmd -CheckOnly
```

Supplementary Ragas evaluation is disabled by default. The following variant includes up to five locally judged answered cases after the primary comparison:

```cmd
scripts\validate_40.cmd -RagasLimit 5 -JudgeSeconds 1800
```

All inference stays local. RAGAS failures and time-budget limits remain explicit. It scores only successfully answered, answerable cases and cannot replace source review.

## Files and interpretation

- `evaluation/final_validation_40.jsonl`: frozen questions, references, filters and source evidence.
- `evaluation/final_validation_40.manifest.json`: paper inventory, snapshot, hash, protocol and limitations.
- `evaluation/final_validation_40_review.md`: readable questions, references and PDF-page pointers.
- `results/summary/validation40_run.json`: pointer to the actual versioned output folder.
- That folder contains baseline/current answers, source checkpoints, summary JSON and operational metric CSV/JSON.
- `results/summary/validation40_acceptance.json`: coverage/known-negative acceptance report.

Single-paper coverage and comparison coverage are reported separately. Comparison cited-paper recall checks which requested papers were actually cited, not merely retrieved. Metadata violations are reported separately. Answer coverage, literal citation validity and annotated-span recall are not factual accuracy. Unavailable factuality/support scores remain unscored.

The default coverage gate is 75% of answerable cases with zero runtime failures and correct known-negative refusals. This is an engineering acceptance criterion, not a research accuracy claim. Recorded failures and frozen reference labels are retained. Once outcomes inform tuning, repeated evaluations on this set constitute regression evidence.

Question wording is new and frozen before this dataset's first evaluation. The underlying papers and some related fact topics overlap earlier development data, so the set is not an unseen-paper benchmark.
