# Review of the completed 40-question run

Run: results/grounding-validation-abad545bfc7a. No new model inference was performed for this review. The saved answers, frozen references and selected source spans were inspected.

## Operational results

| Measure | Generation baseline | Grounded pipeline |
|---|---:|---:|
| Single-paper questions answered | 20/20 | 19/20 |
| Comparisons answered | 7/8 | 3/8 |
| Missing-evidence questions answered | 8/8 | 0/8 |
| Empty scopes correctly rejected | 4/4 | 4/4 |
| Runtime/invalid-output failures | 1 | 0 |
| Metadata constraint violations | 0 | 0 |

The grounded pipeline passes the configured coverage/control gate: 22/28 answerable cases received answers, all 12 negative controls were correctly refused, and runtime failures were zero. This gate does not establish semantic answer correctness.

## Automated source-reference review

Of the 22 grounded outputs marked answered:

- Seven directly answer the essential requested facts.
- Eight supply only part of the requested information.
- Seven are irrelevant, answer the wrong question, or use the wrong paper.

The six refused answerable cases are scored as unfulfilled answers, not as fabricated factual claims. The 12 correct negative-control refusals are recorded separately.

The rubric and notes are in the run's assistant_review_manifest.json and assistant_source_judgments.jsonl. These are automated source-reference judgments, not independent human validation. Semantic citation-support scores remain unscored. Twenty-seven selected quotation spans were checked against original PDF-page text and all matched; literal source provenance is therefore distinct from answer relevance.

Examples of semantic failures:

- DPR vector scoring: an excerpt about Euclidean-distance transformations does not identify the requested inner product.
- MTEB clustering: an excerpt about closest-pair cosine similarity does not identify v-measure.
- ColBERT punctuation filtering: a generic reference to Figure 2 does not identify the filtered token embeddings.
- Ragas statement extraction: the returned excerpt comes from ARES and describes a different evaluation component.
- BEIR versus MTEB comparison: genuine quotations do not provide the requested nDCG@10 and clustering v-measure comparison.

## Consequence

The source-bound output policy reduced unsupported responses on the known negative controls, but correct quotation alone has not produced reliable question answering. The pipeline still needs question-aware answer extraction and more reliable requested-paper attribution. The 78.6% figure measures answer coverage; it does not measure factual accuracy.

The review uses saved answers and frozen references. These observations provide development feedback for any future system version; a version tuned on this set would require separate testing.
