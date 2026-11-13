# Evidence and answer policy

Group 29's default policy selects source excerpts and renders them with paper/page citations. It can return an answer, insufficient evidence or an empty-scope result. Optional verified synthesis is separate from this default mode.

## Processing stages

1. Explicit paper, author, category and year constraints define eligibility before retrieval.
2. Requested identifiers, quoted entities and measurement units are matched against retrieved evidence. Case/separator differences are normalized while token boundaries remain significant. This lexical heuristic does not establish absence from an entire paper.
3. Excerpt ranking uses parent context and source-coverage quotas; page-relative character spans identify each candidate.
4. Qwen2.5 3B returns a schema-valid status and evidence IDs. An answered status has available IDs; a refusal has an empty selection.
5. Required-paper coverage and quotation matching verify source identity. Matching handles PDF whitespace, ligatures and line-wrap hyphenation while recovering original offsets.
6. The local NLI classifier applies an entailment threshold of 0.70 and a parent-contradiction veto of 0.50. Scores are uncalibrated.
7. Rejected excerpts are pruned while independently supported selections remain. Bounded focused rechecks precede an insufficient-evidence response. Model failures are recorded separately from refusals.
8. Python renders original quotations, paper titles, PDF pages and citation labels. Traces record rankings, decisions and timings.

The support model is `cross-encoder/nli-MiniLM2-L6-H768`, pinned to `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`. Its cache manifest records verified file hashes. The pipeline does not train or fine-tune this model.

## Evaluation and limits

The final 40-question source-based run answered 22/28 answerable cases and refused all 12 negative controls. It had zero runtime failures. Automated source-reference review of the 22 answers found seven complete, eight partial and seven wrong/unresponsive cases. There is no independent human validation.

A matching quotation can answer the wrong question or come from a different paper. NLI acceptance does not guarantee relevance, completeness or correct experimental interpretation. Comparisons, tables, equations and facts spread across multiple passages remain difficult. Older 16- and eight-question sets are regression evidence after development feedback.
