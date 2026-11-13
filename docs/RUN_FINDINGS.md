# Experimental findings

## Final 40-question comparison

Run: `results/grounding-validation-abad545bfc7a`. Both arms use hybrid retrieval and reranking. The generation baseline and source-based system use different answer policies; this is not a one-component ablation.

| Measure | Generation baseline | Source-based system |
|---|---:|---:|
| Answerable cases answered | 27/28 | 22/28 |
| Single-paper cases answered | 20/20 | 19/20 |
| Comparisons answered | 7/8 | 3/8 |
| Missing-evidence cases correctly refused | 0/8 | 8/8 |
| Empty scopes correctly refused | 4/4 | 4/4 |
| False refusals on answerable cases | 0 | 6 |
| Runtime/invalid-output failures | 1 | 0 |
| Mean seconds, all 40 cases | 27.10 | 33.79 |

Both arms have zero metadata violations and annotated-span recall@4 of 0.554. The source-based system's 78.6% answer coverage is not an accuracy score. Automated source-reference review finds seven complete, eight partial and seven wrong/unresponsive answers among its 22 answered cases. The baseline has no comparable correctness review. All 27 copied quotation spans match source pages, but genuine text may still be irrelevant.

## Earlier retrieval diagnostic

| Mode | Annotated-span recall@4 | Semantic nDCG@4 |
|---|---:|---:|
| BM25 | 0.236 | 0.613 |
| Dense | 0.000 | 0.613 |
| Hybrid | 0.236 | 0.733 |
| Hybrid + reranking | 0.361 | 0.866 |

Span recall uses annotated evidence from the earlier answerable test questions. Semantic nDCG covers six diagnostic questions with graded pooled passages. These denominators differ. Labels are automated source-reference judgments without independent human validation. Dense retrieval's zero exact span score does not establish zero semantic relevance.

The separate filtering experiment has 36 enabled and 36 disabled records. Disabled filtering returns 64 out-of-scope passages among 144 returned passages. Enabled filtering returns zero among 96 returned passages, including empty eligible scopes.

The earlier 120-answer generation run is retained in `results/answers-test-754021b26646`. Every retrieval mode answered all four absent-model questions in its test set. Valid citation IDs did not prevent unsupported claims. These earlier outputs evaluate a different answer policy from the final source-based run.

## Supplementary local judge

Ragas 0.4.3 and local Llama3.2 3B score earlier baseline answers. The saved ten-case sample planned 40 metric calls and attempted 29 within 1,800 seconds: 25 completed, four failed and 11 were not attempted. Its manifest is `results/ragas-857109279246/manifest.json`.

Native Ollama JSON schemas and validation address transport failures, not judge calibration. Raw calls and successful cached scores accompany the saved evaluator results. This incomplete sample does not measure the final source-based system's accuracy.

## Interpretation

The project demonstrates persistent local retrieval, explicit metadata constraints and traceable quotations. Question relevance, paper attribution and comparison completeness remain limitations. No independent human factuality score or broad generalization claim is supported.
