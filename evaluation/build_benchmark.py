"""Create source-checked course QA cases with exact, validated PDF text spans.

References were drafted from inspected paper text before test answer generation.
Review provenance is explicit: assistant source-check, not independent human gold.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import PROJECT_ROOT,fingerprint
from assistant.filters import eligible_papers
from assistant.schemas import Paper,RetrievalFilters

# Each fact is paired with a short exact-source phrase. Whitespace/ligatures are
# normalized only for finding offsets; the saved quotation is original page text.
FACTS = {
 'realm':('2002.08909',1,[('REALM uses masked language modeling to learn retrieval.','using masked language model-ing as the learning signal'),('Training gradients pass through the retrieval step.','backpropagating through a retrieval step')]),
 'dpr':('2004.04906',1,[('DPR learns question and passage embeddings using a dual encoder.','questions and passages by a simple dual-encoder framework')]),
 'rag':('2005.11401',1,[('One RAG formulation uses the same passages across a whole output sequence.','same retrieved passages across the whole generated sequence'),('Another formulation can use different passages per token.','different passages per token')]),
 'fid':('2007.01282',1,[('Performance improves as more passages are retrieved.','performance of this method sig-nificantly improves when increasing the num-ber of retrieved passages')]),
 'colbert':('2004.12832',1,[('ColBERT encodes queries and documents independently before late interaction.','late interaction architecture that indepen-dently encodes the query and the document using BERT'),('Document representations can be precomputed offline.','pre-compute document representations offline')]),
 'pyserini':('2102.10073',1,[('Pyserini supports sparse retrieval such as BM25.','BM25 scoring using bag-of-words representations'),('It supports dense retrieval.','dense re-trieval'),('It supports hybrid retrieval combining sparse and dense approaches.','hybrid retrieval that integrates both ap-proaches')]),
 'beir':('2104.08663',1,[('BEIR finds BM25 to be a robust baseline.','BM25 is a robust baseline'),('Reranking and late interaction have high computational costs.','re-ranking and late-interaction based models on average achieve the best zero-shot performances, however, at high computational costs')]),
 'splade':('2107.05720',1,[('SPLADE uses explicit sparsity regularization.','explicit sparsity regularization'),('It uses log-saturation on term weights.','a log-saturation effect on term weights')]),
 'colbert2':('2112.01488',1,[('ColBERTv2 uses residual compression.','aggressive residual compres-sion mechanism'),('It combines compression with denoised supervision.','denoised supervision strategy')]),
 're2g':('2207.06300',1,[('Re2G ensembles BM25 and neural initial retrieval.','an ensemble of BM25 and neural initial retrieval'),('It incorporates reranking and BART-based generation.','reranking into a BART-based sequence-to-sequence generation')]),
 'mteb':('2210.07316',1,[('The original MTEB spans eight embedding tasks.','MTEB spans 8 embedding tasks'),('It covers 58 datasets and 112 languages.','58 datasets and 112 languages')]),
 'hyde':('2212.10496',1,[('HyDE generates a hypothetical document using an instruction-following model.','an instruction-following language model (e.g. InstructGPT) to gen-erate a hypothetical document'),('A contrastively trained encoder embeds that document for retrieval.','un-supervised contrastively learned encoder')]),
 'middle':('2307.03172',1,[('Performance is often highest when evidence is at the beginning or end.','performance is often highest when rele-vant information occurs at the beginning or end'),('Performance degrades when relevant information is in the middle.','significantly degrades when models must access relevant information in the middle')]),
 'ragas':('2309.15217',1,[('Ragas proposes reference-free evaluation of RAG pipelines.','reference-free evaluation of Retrieval Aug-mented Generation (RAG) pipelines')]),
 'selfrag':('2310.11511',1,[('Self-RAG retrieves passages adaptively on demand.','adaptively retrieves passages on-demand'),('It uses special reflection tokens for retrieved passages and generations.','generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens')]),
 'ares':('2311.09476',1,[('ARES evaluates context relevance, answer faithfulness and answer relevance.','context relevance, answer faithfulness, and answer relevance'),('It fine-tunes lightweight judges on synthetic data.','its own synthetic training data, ARES finetunes lightweight LM judges'),('It uses human annotations for prediction-powered inference.','human-annotated datapoints for prediction-powered inference (PPI)')]),
 'survey':('2312.10997',1,[('The survey distinguishes Naive, Advanced and Modular RAG.','Naive RAG, the Advanced RAG, and the Modular RAG')]),
 'ragtruth':('2401.00396',1,[('RAGTruth contains nearly 18000 naturally generated responses.','nearly 18,000 nat-urally generated responses'),('Annotations exist at case and word levels.','indi-vidual case and word levels')]),
 'crag':('2401.15884',1,[('CRAG uses a lightweight retrieval evaluator to assess document quality.','a lightweight retrieval evaluator is designed to assess the overall quality'),('It can augment retrieval with web search.','large-scale web searches are utilized as an extension')]),
 'longrag':('2406.15319',1,[('LongRAG groups related documents into 4K-token retrieval units.','4K-token units by grouping related documents'),('Its reported setup can work without training.','Without requiring any training')]),
}

QUESTIONS = [
 ('realm','What learning signal does REALM use, and how do gradients reach the knowledge retriever during pretraining?'),
 ('dpr','What encoder framework does Dense Passage Retrieval use for questions and passages?'),
 ('rag','How do the two original RAG formulations differ in their use of retrieved passages during generation?'),
 ('fid','According to Leveraging Passage Retrieval with Generative Models, what happens when the number of retrieved passages increases?'),
 ('colbert','How does ColBERT use late interaction, and what can it precompute?'),
 ('pyserini','Which sparse, dense and hybrid retrieval capabilities does Pyserini support?'),
 ('beir','What does BEIR conclude about BM25 and the computational cost of reranking and late-interaction models?'),
 ('splade','Which two mechanisms does SPLADE use to produce sparse term representations?'),
 ('colbert2','Which two techniques does ColBERTv2 combine to improve late-interaction retrieval?'),
 ('re2g','Which retrieval methods can Re2G ensemble, and what generation model family does it use?'),
 ('mteb','How broad is the original MTEB benchmark in embedding tasks, datasets and languages?'),
 ('hyde','What are the generation and encoding steps in HyDE?'),
 ('middle','How does relevant-information position affect performance in Lost in the Middle?'),
 ('ragas','What kind of RAG evaluation does the Ragas paper propose?'),
 ('selfrag','How does Self-RAG control retrieval and reflection?'),
 ('ares','What quality dimensions and judging approach does ARES use?'),
]


def normalized_positions(text):
    characters,positions = [],[]
    for index,char in enumerate(text):
        for value in unicodedata.normalize('NFKC',char).casefold():
            if not value.isspace():
                characters.append(value); positions.append(index)
    return ''.join(characters),positions


def main():
    manifest = json.loads((PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))
    snapshot = PROJECT_ROOT/manifest['snapshot']
    pages = read_jsonl(snapshot/'new_pages.jsonl')
    chunks = read_jsonl(snapshot/'chunks.jsonl')
    papers = [Paper.from_dict(p) for p in manifest['papers']]
    page_map = {(p['paper_id'],p['page_number']):p['text'] for p in pages}
    fact_groups = {}
    for name,(paper_id,page_number,facts) in FACTS.items():
        page = page_map[paper_id,page_number]
        normalized,positions = normalized_positions(page)
        groups = []
        for fact,phrase in facts:
            needle,_ = normalized_positions(phrase)
            found = normalized.find(needle)
            if found < 0:
                raise ValueError(f'Source quote not found for {name}: {phrase}')
            a,b = positions[found],positions[found+len(needle)-1]+1
            matching = [c['chunk_id'] for c in chunks if c['paper_id']==paper_id and c['page_number']==page_number and c['start_char']<=a and c['end_char']>=b]
            if not matching:
                raise ValueError(f'Gold evidence cannot fit a frozen chunk: {name}: {phrase}')
            groups.append({'fact':fact,'alternatives':[{'paper_id':paper_id,'page_number':page_number,
                'start_char':a,'end_char':b,'quote':page[a:b],'chunk_ids':matching}]})
        fact_groups[name] = groups
    cases = []
    dev_facts = {'realm','rag','hyde','middle','selfrag'}
    def add(identifier,question,names,kind='single',filters=None,dev=False,required=(),groups_override=None):
        groups = groups_override if groups_override is not None else [g for name in names for g in fact_groups[name]]
        cases.append({'question_id':identifier,'family':'|'.join(sorted(names)) or identifier,'split':'dev' if dev else 'test',
            'type':kind,'question':question,'filters':filters or {},'required_papers':list(required),
            'answerable':True,'reference':' '.join(g['fact'] for g in groups),'evidence_groups':groups,
            'annotation_provenance':'assistant source-checked PDF text before held-out answer generation; independent human review pending'})
    for name,question in QUESTIONS:
        add('single-'+name,question,[name],dev=name in dev_facts)
    comparisons = [
        ('dpr-colbert','Compare the encoder framework of DPR with ColBERT late interaction and offline document precomputation.',['dpr','colbert'],False),
        ('splade-colbert2','Which representation and training techniques do SPLADE and ColBERTv2 use for sparse or compact retrieval?',['splade','colbert2'],False),
        ('re2g-crag','Compare the retrieval and generation components of Re2G with the retrieval-quality and augmentation components of CRAG.',['re2g','crag'],False),
        ('ragas-ares','Compare the evaluation approaches of Ragas and ARES, including judging and human-annotation use.',['ragas','ares'],False),
        ('middle-longrag','What context-position issue does Lost in the Middle report, and what retrieval-unit change does LongRAG propose?',['middle','longrag'],True),
        ('rag-selfrag','Contrast the passage-use formulations in original RAG with Self-RAG retrieval and reflection.',['rag','selfrag'],True),
        ('survey-ragtruth','Which paradigms does the RAG survey distinguish, and what size and annotation granularity does RAGTruth report?',['survey','ragtruth'],False),
        ('realm-hyde','Contrast retriever learning in REALM with the hypothetical-document approach in HyDE.',['realm','hyde'],True)]
    for key,q,names,dev in comparisons:
        groups = [g for name in names for g in fact_groups[name]]
        if key=='middle-longrag':
            groups = fact_groups['middle']+[fact_groups['longrag'][0]]
        required = [FACTS[name][0] for name in names]
        add('compare-'+key,q,names,'comparison',{'paper_ids':required},dev,required,groups)
    metadata = [
        ('survey','Among papers first submitted from 2023 onward, which three RAG paradigms does the survey distinguish?',{'year_min':2023}),
        ('ragtruth','Among papers first submitted from 2023 onward, what size and annotation granularity does RAGTruth report?',{'year_min':2023}),
        ('crag','In the papers first submitted in 2024, how does CRAG assess and supplement retrieved evidence?',{'year_min':2024,'year_max':2024}),
        ('longrag','Does the LongRAG paper first submitted in 2024 require training for its reported setup?',{'year_min':2024,'year_max':2024}),
        ('mteb','Among papers first submitted by 2022, what task, dataset and language coverage does MTEB report?',{'year_max':2022}),
        ('beir','Restricting to cs.IR papers first submitted by 2021, what baseline and cost tradeoff does BEIR report?',{'year_max':2021,'categories':['cs.IR']}),
        ('pyserini','In the selected Pyserini paper, which retrieval families are supported?',{'paper_ids':['2102.10073']}),
        ('re2g','In the selected Re2G paper first submitted in 2022, which retrieval methods are ensembled and what generator family is used?',{'paper_ids':['2207.06300'],'year_min':2022,'year_max':2022})]
    for name,q,filters in metadata:
        groups = [fact_groups[name][1]] if name=='longrag' else fact_groups[name]
        add('metadata-'+name,q,[name] if name!='longrag' else ['longrag-training'], 'metadata',filters,groups_override=groups)
    absent = [
        ('rag','What accuracy does the original RAG paper report for Qwen2.5-3B?','qwen'),
        ('dpr','What training results does the DPR paper report for Llama 3.2 3B?','llama'),
        ('middle','What fine-tuning GPU-hours does Lost in the Middle report for Qwen2.5?','qwen'),
        ('realm','What benchmark results does the REALM paper report for DeepSeek-R1?','deepseek')]
    for name,q,term in absent:
        paper_id = FACTS[name][0]
        full = '\n'.join(p['text'] for p in pages if p['paper_id']==paper_id)
        if term in full.casefold():
            raise ValueError(f'Absence audit failed for {name}:{term}')
        add('unanswerable-'+name,q,[],'unanswerable',{'paper_ids':[paper_id]})
        cases[-1].update({'answerable':False,'reference':'The selected paper does not report results for the requested model.',
                         'absence_audit':{'paper_id':paper_id,'missing_model_term':term,'all_extracted_pages_checked':True}})
    empty = [
        ('early-year',{'year_max':2019},True),
        ('biology-category',{'categories':['q-bio.GN']},True),
        ('negative-author',{'authors':['Course Benchmark Negative Author']},False),
        ('paper-year-conflict',{'paper_ids':['2005.11401'],'year_min':2024},False)]
    for key,filters,dev in empty:
        eligible,_ = eligible_papers(papers,RetrievalFilters.from_dict(filters))
        if eligible:
            raise ValueError('No-match case unexpectedly contains eligible papers')
        add('no-match-'+key,'What retrieval approach is described by the papers in this selected scope?',[],
            'no_match',filters,dev)
        cases[-1].update({'answerable':False,'reference':'No indexed papers satisfy the explicit scope.',
                         'absence_audit':{'expected_eligible_paper_ids':[]}})
    for case in cases:
        eligible,_ = eligible_papers(papers,RetrievalFilters.from_dict(case['filters']))
        allowed = {p.paper_id for p in eligible}
        for group in case['evidence_groups']:
            if not any(a['paper_id'] in allowed for a in group['alternatives']):
                raise ValueError(f"Gold excluded by filter: {case['question_id']}")
        case['snapshot_id'] = manifest['snapshot_id']
    assert len(cases)==40 and sum(c['split']=='dev' for c in cases)==10
    assert sum(c['answerable'] for c in cases)==32
    destination = PROJECT_ROOT/'evaluation/questions.jsonl'
    write_jsonl(destination,cases)
    atomic_json(PROJECT_ROOT/'evaluation/benchmark_manifest.json',{
        'status':'assistant_source_checked_human_review_pending','snapshot_id':manifest['snapshot_id'],
        'questions':40,'answerable':32,'dev':10,'test':30,'questions_hash':fingerprint(cases),
        'limitations':['Small fixed-corpus benchmark','Assistant source-checked, not independently human-validated',
                      'Evidence annotations are not exhaustive semantic relevance judgments','Current paper versions are frozen, not universally latest']})
    print(f'Validated and saved {len(cases)} cases with real PDF evidence spans.')


if __name__=='__main__':
    main()
