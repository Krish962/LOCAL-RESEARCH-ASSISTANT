# Validation 40: reference catalogue

Automatically source-checked references compiled before evaluated model answers. Annotation provenance is recorded in the paired JSONL file. These labels have no independent human validation.

Questions SHA256 fingerprint: `6df09fbf0420f93931858040ffa84dbcc41e847308f7f7a52d72f8716c9d35bd`

Page numbers below refer to physical PDF pages (one-based), not printed proceedings page numbers. Exact quotations, offsets and chunk IDs are in the JSONL dataset.

## 1. validation40-single-realm

As the document encoder changes during REALM pre-training, how is the document index refreshed?

Type: single; answerable: True

Scope: `{}`

Reference: REALM asynchronously re-embeds and re-indexes all documents every several hundred training steps.

- Supporting pages: [2002.08909](https://arxiv.org/abs/2002.08909), PDF p. 4


## 2. validation40-single-dpr

Which operation matches question vectors to relevant passage vectors in DPR?

Type: single; answerable: True

Scope: `{"paper_ids": ["2004.04906"]}`

Reference: DPR uses inner products between question and passage vectors.

- Supporting pages: [2004.04906](https://arxiv.org/abs/2004.04906), PDF p. 2


## 3. validation40-single-rag

What constitutes the non-parametric memory in the original RAG architecture?

Type: single; answerable: True

Scope: `{}`

Reference: The non-parametric memory is a dense vector index of Wikipedia.

- Supporting pages: [2005.11401](https://arxiv.org/abs/2005.11401), PDF p. 1, [2005.11401](https://arxiv.org/abs/2005.11401), PDF p. 2


## 4. validation40-single-fid

Which pretrained model family initializes the generative models in Fusion-in-Decoder?

Type: single; answerable: True

Scope: `{"paper_ids": ["2007.01282"]}`

Reference: FiD initializes its generative models with pretrained T5 models.

- Supporting pages: [2007.01282](https://arxiv.org/abs/2007.01282), PDF p. 4


## 5. validation40-single-colbert

Which document-token embeddings does ColBERT filter out to reduce its representation size?

Type: single; answerable: True

Scope: `{"paper_ids": ["2004.12832"]}`

Reference: ColBERT filters out document embeddings corresponding to punctuation symbols.

- Supporting pages: [2004.12832](https://arxiv.org/abs/2004.12832), PDF p. 4


## 6. validation40-single-pyserini

Which underlying libraries provide sparse search and dense or hybrid search in Pyserini?

Type: single; answerable: True

Scope: `{"categories": ["cs.IR"], "year_max": 2021, "year_min": 2021}`

Reference: Sparse retrieval comes from Lucene through Anserini. Pyserini integrates FAISS for dense and hybrid retrieval.

- Supporting pages: [2102.10073](https://arxiv.org/abs/2102.10073), PDF p. 1
- Supporting pages: [2102.10073](https://arxiv.org/abs/2102.10073), PDF p. 1


## 7. validation40-single-beir

Which ranking metric does BEIR compute across its datasets for the reported experiments?

Type: single; answerable: True

Scope: `{}`

Reference: BEIR computes nDCG@10 for all datasets.

- Supporting pages: [2104.08663](https://arxiv.org/abs/2104.08663), PDF p. 5


## 8. validation40-single-splade

How does SPLADE control its effectiveness-versus-efficiency trade-off?

Type: single; answerable: True

Scope: `{"paper_ids": ["2107.05720"]}`

Reference: SPLADE controls the contribution of sparsity regularization to adjust the effectiveness-efficiency trade-off.

- Supporting pages: [2107.05720](https://arxiv.org/abs/2107.05720), PDF p. 1


## 9. validation40-single-colbertv2

What information is stored to encode a vector using ColBERTv2 residual compression?

Type: single; answerable: True

Scope: `{"paper_ids": ["2112.01488"]}`

Reference: ColBERTv2 stores the index of the closest centroid and a quantized approximation of the residual vector.

- Supporting pages: [2112.01488](https://arxiv.org/abs/2112.01488), PDF p. 4


## 10. validation40-single-re2g

What score-comparability problem does Re2G address when merging retriever outputs?

Type: single; answerable: True

Scope: `{"paper_ids": ["2207.06300"]}`

Reference: Re2G reranks results from retrieval sources whose scores are not directly comparable.

- Supporting pages: [2207.06300](https://arxiv.org/abs/2207.06300), PDF p. 1


## 11. validation40-single-mteb

Which score is used to evaluate clustering models in MTEB?

Type: single; answerable: True

Scope: `{"authors": ["Niklas Muennighoff"], "year_max": 2022}`

Reference: MTEB scores clustering models using v-measure.

- Supporting pages: [2210.07316](https://arxiv.org/abs/2210.07316), PDF p. 3


## 12. validation40-single-hyde

What role does the encoder bottleneck play in HyDE when a hypothetical document contains incorrect details?

Type: single; answerable: True

Scope: `{}`

Reference: HyDE expects the dense bottleneck to filter incorrect or hallucinated details from the hypothetical-document embedding.

- Supporting pages: [2212.10496](https://arxiv.org/abs/2212.10496), PDF p. 1, [2212.10496](https://arxiv.org/abs/2212.10496), PDF p. 2


## 13. validation40-single-middle

Which two tasks are used in Lost in the Middle to study access to relevant context?

Type: single; answerable: True

Scope: `{"paper_ids": ["2307.03172"]}`

Reference: The tasks are multi-document question answering and key-value retrieval.

- Supporting pages: [2307.03172](https://arxiv.org/abs/2307.03172), PDF p. 1


## 14. validation40-single-ragas

How does Ragas prepare an answer for individual statement-level faithfulness checks?

Type: single; answerable: True

Scope: `{}`

Reference: Ragas uses an LLM to extract statements from the answer. Longer sentences are decomposed into shorter, focused assertions.

- Supporting pages: [2309.15217](https://arxiv.org/abs/2309.15217), PDF p. 3
- Supporting pages: [2309.15217](https://arxiv.org/abs/2309.15217), PDF p. 3


## 15. validation40-single-selfrag

How are reflection tokens incorporated into the language-model training objective in Self-RAG?

Type: single; answerable: True

Scope: `{"paper_ids": ["2310.11511"]}`

Reference: Self-RAG learns reflection tokens as next-token predictions from an expanded vocabulary.

- Supporting pages: [2310.11511](https://arxiv.org/abs/2310.11511), PDF p. 2, [2310.11511](https://arxiv.org/abs/2310.11511), PDF p. 4


## 16. validation40-single-ares

From what material does ARES generate the synthetic question-answer data used for its judges?

Type: single; answerable: True

Scope: `{"paper_ids": ["2311.09476"]}`

Reference: ARES generates synthetic queries and answers from in-domain corpus passages using a language model.

- Supporting pages: [2311.09476](https://arxiv.org/abs/2311.09476), PDF p. 2, [2311.09476](https://arxiv.org/abs/2311.09476), PDF p. 3


## 17. validation40-single-survey

Which two retrieval-related stages does the survey identify as optimization targets in Advanced RAG?

Type: single; answerable: True

Scope: `{"paper_ids": ["2312.10997"]}`

Reference: Advanced RAG employs pre-retrieval and post-retrieval strategies.

- Supporting pages: [2312.10997](https://arxiv.org/abs/2312.10997), PDF p. 3


## 18. validation40-single-ragtruth

Which three generation tasks are represented in the RAGTruth response collection?

Type: single; answerable: True

Scope: `{"year_max": 2023, "year_min": 2023}`

Reference: The tasks are question answering, data-to-text writing and news summarization.

- Supporting pages: [2401.00396](https://arxiv.org/abs/2401.00396), PDF p. 3


## 19. validation40-single-crag

Which three operations refine knowledge after a Correct retrieval decision in CRAG?

Type: single; answerable: True

Scope: `{}`

Reference: CRAG refines knowledge using decomposition, filtering and recomposition.

- Supporting pages: [2401.15884](https://arxiv.org/abs/2401.15884), PDF p. 3


## 20. validation40-single-longrag

How does the choice of retrieval-unit length in LongRAG affect the retriever workload?

Type: single; answerable: True

Scope: `{"year_max": 2024, "year_min": 2024}`

Reference: LongRAG operates on long retrieval units. The LongRAG retriever has a significantly reduced workload.

- Supporting pages: [2406.15319](https://arxiv.org/abs/2406.15319), PDF p. 2, [2406.15319](https://arxiv.org/abs/2406.15319), PDF p. 3
- Supporting pages: [2406.15319](https://arxiv.org/abs/2406.15319), PDF p. 2


## 21. validation40-compare-dense-and-compressed

Contrast DPR question-passage vector scoring with the vector representation stored by ColBERTv2 residual compression.

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2004.04906", "2112.01488"]}`

Reference: DPR uses inner products between question and passage vectors. ColBERTv2 stores the index of the closest centroid and a quantized approximation of the residual vector.

- Supporting pages: [2004.04906](https://arxiv.org/abs/2004.04906), PDF p. 2
- Supporting pages: [2112.01488](https://arxiv.org/abs/2112.01488), PDF p. 4


## 22. validation40-compare-representation-cost

How do ColBERT punctuation filtering and SPLADE sparsity regularization reduce or control representation and retrieval costs?

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2004.12832", "2107.05720"]}`

Reference: ColBERT filters out document embeddings corresponding to punctuation symbols. SPLADE controls the contribution of sparsity regularization to adjust the effectiveness-efficiency trade-off.

- Supporting pages: [2004.12832](https://arxiv.org/abs/2004.12832), PDF p. 4
- Supporting pages: [2107.05720](https://arxiv.org/abs/2107.05720), PDF p. 1


## 23. validation40-compare-hybrid-tooling-and-reranking

Compare the library Pyserini uses for dense or hybrid search with how Re2G handles incomparable scores from retrieval sources.

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2102.10073", "2207.06300"]}`

Reference: Pyserini integrates FAISS for dense and hybrid retrieval. Re2G reranks results from retrieval sources whose scores are not directly comparable.

- Supporting pages: [2102.10073](https://arxiv.org/abs/2102.10073), PDF p. 1
- Supporting pages: [2207.06300](https://arxiv.org/abs/2207.06300), PDF p. 1


## 24. validation40-compare-retrieval-work

How do REALM index maintenance and LongRAG retrieval-unit design change retrieval-side processing?

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2002.08909", "2406.15319"]}`

Reference: REALM asynchronously re-embeds and re-indexes all documents every several hundred training steps. LongRAG operates on long retrieval units.

- Supporting pages: [2002.08909](https://arxiv.org/abs/2002.08909), PDF p. 4
- Supporting pages: [2406.15319](https://arxiv.org/abs/2406.15319), PDF p. 2, [2406.15319](https://arxiv.org/abs/2406.15319), PDF p. 3


## 25. validation40-compare-evaluation-artifacts

How do Ragas and ARES differ in the generated text they use for evaluation: focused answer statements versus synthetic judge-training examples?

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2309.15217", "2311.09476"]}`

Reference: Longer sentences are decomposed into shorter, focused assertions. ARES generates synthetic queries and answers from in-domain corpus passages using a language model.

- Supporting pages: [2309.15217](https://arxiv.org/abs/2309.15217), PDF p. 3
- Supporting pages: [2311.09476](https://arxiv.org/abs/2311.09476), PDF p. 2, [2311.09476](https://arxiv.org/abs/2311.09476), PDF p. 3


## 26. validation40-compare-benchmark-metrics

Contrast the ranking metric used in BEIR experiments with the clustering metric used in MTEB.

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2104.08663", "2210.07316"]}`

Reference: BEIR computes nDCG@10 for all datasets. MTEB scores clustering models using v-measure.

- Supporting pages: [2104.08663](https://arxiv.org/abs/2104.08663), PDF p. 5
- Supporting pages: [2210.07316](https://arxiv.org/abs/2210.07316), PDF p. 3


## 27. validation40-compare-memory-and-generator

Compare the non-parametric memory in original RAG with the pretrained generator initialization described by Fusion-in-Decoder.

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2005.11401", "2007.01282"]}`

Reference: The non-parametric memory is a dense vector index of Wikipedia. FiD initializes its generative models with pretrained T5 models.

- Supporting pages: [2005.11401](https://arxiv.org/abs/2005.11401), PDF p. 1, [2005.11401](https://arxiv.org/abs/2005.11401), PDF p. 2
- Supporting pages: [2007.01282](https://arxiv.org/abs/2007.01282), PDF p. 4


## 28. validation40-compare-refinement-stages

What are CRAG knowledge-refinement operations, and which pre-retrieval or post-retrieval stages are optimization targets in the RAG survey?

Type: comparison; answerable: True

Scope: `{"paper_ids": ["2401.15884", "2312.10997"]}`

Reference: CRAG refines knowledge using decomposition, filtering and recomposition. Advanced RAG employs pre-retrieval and post-retrieval strategies.

- Supporting pages: [2401.15884](https://arxiv.org/abs/2401.15884), PDF p. 3
- Supporting pages: [2312.10997](https://arxiv.org/abs/2312.10997), PDF p. 3


## 29. validation40-missing-realm-qwen

What retrieval accuracy does REALM report for Qwen2.5-3B?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2002.08909"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 12 extracted pages checked; subtype missing_model.


## 30. validation40-missing-colbertv2-llama

What benchmark results does ColBERTv2 report for Llama 3.2 3B?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2112.01488"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 20 extracted pages checked; subtype missing_model.


## 31. validation40-missing-pyserini-gemma

What answer-generation accuracy does the Pyserini paper report for Gemma-2-2B?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2102.10073"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 8 extracted pages checked; subtype missing_model.


## 32. validation40-missing-beir-phi

What BEIR score is reported for Phi-3-mini in the selected paper?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2104.08663"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 24 extracted pages checked; subtype missing_model.


## 33. validation40-missing-ares-energy

How many kilowatt-hours does ARES explicitly report consuming to train its judges?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2311.09476"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 17 extracted pages checked; subtype missing_measurement.


## 34. validation40-missing-splade-dollar-cost

What total training expense in US dollars is explicitly reported for SPLADE?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2107.05720"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 5 extracted pages checked; subtype missing_measurement.
- Related monetary mention: ACM publication boilerplate, not a reported training expense.


## 35. validation40-missing-colbertv2-uncertainty

What 95% confidence interval does ColBERTv2 report for its MRR@10 result?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2112.01488"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 20 extracted pages checked; subtype unreported_statistics.


## 36. validation40-missing-crag-significance

What statistical significance p-value does CRAG report for its comparison with Self-RAG?

Type: missing_evidence; answerable: False

Scope: `{"paper_ids": ["2401.15884"]}`

Reference: The selected paper does not report the requested model result, measurement, or statistical result.

- Absence audit: 16 extracted pages checked; subtype unreported_statistics.


## 37. validation40-empty-early-years

Among papers in this library first submitted during 2010–2018, what retrieval methods are described?

Type: no_match; answerable: False

Scope: `{"year_max": 2018, "year_min": 2010}`

Reference: No indexed papers satisfy the explicit metadata scope.


## 38. validation40-empty-biology

Within indexed q-bio.GN papers, which retrieval algorithms are presented?

Type: no_match; answerable: False

Scope: `{"categories": ["q-bio.GN"]}`

Reference: No indexed papers satisfy the explicit metadata scope.


## 39. validation40-empty-absent-author

Which retrieval techniques are reported by the selected author in this library?

Type: no_match; answerable: False

Scope: `{"authors": ["Validation40_NoIndexedAuthor"]}`

Reference: No indexed papers satisfy the explicit metadata scope.


## 40. validation40-empty-pyserini-year-conflict

For the selected Pyserini paper restricted to first submissions from 2024 onward, what search backends are reported?

Type: no_match; answerable: False

Scope: `{"paper_ids": ["2102.10073"], "year_min": 2024}`

Reference: No indexed papers satisfy the explicit metadata scope.


