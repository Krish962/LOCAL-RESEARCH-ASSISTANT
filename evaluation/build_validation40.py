"""Build and verify a frozen, model-free 40-question evaluation on all 20 papers."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import PROJECT_ROOT,fingerprint
from assistant.filters import eligible_papers
from assistant.grounding import quotation_span,compact
from assistant.schemas import Paper,RetrievalFilters

DESTINATION=PROJECT_ROOT/'evaluation/final_validation_40.jsonl'

# Each item is (paper ID, natural question, [(reference fact, literal anchors)]).
# Anchors are resolved back to original PDF-page character spans, never generated evidence.
FACTS={
 'realm':('2002.08909','As the document encoder changes during REALM pre-training, how is the document index refreshed?',[
  ('REALM asynchronously re-embeds and re-indexes all documents every several hundred training steps.',
   ['asynchronously re-embedding and re-indexing all documents every several hundred training steps'])]),
 'dpr':('2004.04906','Which operation matches question vectors to relevant passage vectors in DPR?',[
  ('DPR uses inner products between question and passage vectors.',
   ['maximizing inner products of the question and relevant passage vectors'])]),
 'rag':('2005.11401','What constitutes the non-parametric memory in the original RAG architecture?',[
  ('The non-parametric memory is a dense vector index of Wikipedia.',
   ['non-parametric memory is a dense vector index of Wikipedia'])]),
 'fid':('2007.01282','Which pretrained model family initializes the generative models in Fusion-in-Decoder?',[
  ('FiD initializes its generative models with pretrained T5 models.',
   ['We initialize our models with the pretrained T5 models'])]),
 'colbert':('2004.12832','Which document-token embeddings does ColBERT filter out to reduce its representation size?',[
  ('ColBERT filters out document embeddings corresponding to punctuation symbols.',
   ['the document encoder /f_ilters out the embeddings corresponding to punctuation symbols'])]),
 'pyserini':('2102.10073','Which underlying libraries provide sparse search and dense or hybrid search in Pyserini?',[
  ('Sparse retrieval comes from Lucene through Anserini.',
   ['Sparse retrieval support in Pyserini comes entirely from Lucene (via Anserini)']),
  ('Pyserini integrates FAISS for dense and hybrid retrieval.',
   ['Pyserini integrates Facebook’s FAISS library'])]),
 'beir':('2104.08663','Which ranking metric does BEIR compute across its datasets for the reported experiments?',[
  ('BEIR computes nDCG@10 for all datasets.', ['compute nDCG@10 for all datasets'])]),
 'splade':('2107.05720','How does SPLADE control its effectiveness-versus-efficiency trade-off?',[
  ('SPLADE controls the contribution of sparsity regularization to adjust the effectiveness-efficiency trade-off.',
   ['trade-off between effectiveness and efficiency, by controlling the contribution of the sparsity regularization'])]),
 'colbertv2':('2112.01488','What information is stored to encode a vector using ColBERTv2 residual compression?',[
  ('ColBERTv2 stores the index of the closest centroid and a quantized approximation of the residual vector.',
   ['ColBERTv2 encodes each vector v as the index of its closest centroid Ct and a quantized vector ˜r that approximates the residual'])]),
 're2g':('2207.06300','What score-comparability problem does Re2G address when merging retriever outputs?',[
  ('Re2G reranks results from retrieval sources whose scores are not directly comparable.',
   ['reranking approach also permits merging retrieval results from sources with incomparable scores',
    'reranking approach permits merging retrieval results from sources with incomparable scores'])]),
 'mteb':('2210.07316','Which score is used to evaluate clustering models in MTEB?',[
  ('MTEB scores clustering models using v-measure.', ['The model is scored using v-measure'])]),
 'hyde':('2212.10496','What role does the encoder bottleneck play in HyDE when a hypothetical document contains incorrect details?',[
  ('HyDE expects the dense bottleneck to filter incorrect or hallucinated details from the hypothetical-document embedding.',
   ['encoder’s dense bottleneck filtering out the incorrect details',
    'dense bottleneck to serve a lossy compressor, where the extra (hallucinated) details are filtered out from the embedding'])]),
 'middle':('2307.03172','Which two tasks are used in Lost in the Middle to study access to relevant context?',[
  ('The tasks are multi-document question answering and key-value retrieval.',
   ['multi-document question answering and key-value retrieval'])]),
 'ragas':('2309.15217','How does Ragas prepare an answer for individual statement-level faithfulness checks?',[
  ('Ragas uses an LLM to extract statements from the answer.', ['we first use an LLM to extract a set of statements']),
  ('Longer sentences are decomposed into shorter, focused assertions.',
   ['decompose longer sentences into shorter and more focused assertions'])]),
 'selfrag':('2310.11511','How are reflection tokens incorporated into the language-model training objective in Self-RAG?',[
  ('Self-RAG learns reflection tokens as next-token predictions from an expanded vocabulary.',
   ['next token prediction from the expanded model vocabulary',
    'next token predictions from the expanded model vocabulary'])]),
 'ares':('2311.09476','From what material does ARES generate the synthetic question-answer data used for its judges?',[
  ('ARES generates synthetic queries and answers from in-domain corpus passages using a language model.',
   ['construct a synthetic dataset of question–answer pairs, derived from the passages in the corpus',
    'We generate synthetic queries and answers from the corpus passages using generative LLMs'])]),
 'survey':('2312.10997','Which two retrieval-related stages does the survey identify as optimization targets in Advanced RAG?',[
  ('Advanced RAG employs pre-retrieval and post-retrieval strategies.',
   ['it employs pre-retrieval and post-retrieval strategies'])]),
 'ragtruth':('2401.00396','Which three generation tasks are represented in the RAGTruth response collection?',[
  ('The tasks are question answering, data-to-text writing and news summarization.',
   ['Question Answering, Data-to-text Writing, and News Summarization'])]),
 'crag':('2401.15884','Which three operations refine knowledge after a Correct retrieval decision in CRAG?',[
  ('CRAG refines knowledge using decomposition, filtering and recomposition.',
   ['knowledge decomposition, filter, and recomposition'])]),
 'longrag':('2406.15319','How does the choice of retrieval-unit length in LongRAG affect the retriever workload?',[
  ('LongRAG operates on long retrieval units.', ['LongRAG operates on long retrieval units']),
  ('The LongRAG retriever has a significantly reduced workload.',
   ['retriever of LongRAG has a significantly reduced workload'])]),
}

COMPARISONS=[
 ('dense-and-compressed','Contrast DPR question-passage vector scoring with the vector representation stored by ColBERTv2 residual compression.', [('dpr',None),('colbertv2',None)]),
 ('representation-cost','How do ColBERT punctuation filtering and SPLADE sparsity regularization reduce or control representation and retrieval costs?', [('colbert',None),('splade',None)]),
 ('hybrid-tooling-and-reranking','Compare the library Pyserini uses for dense or hybrid search with how Re2G handles incomparable scores from retrieval sources.', [('pyserini',[1]),('re2g',None)]),
 ('retrieval-work','How do REALM index maintenance and LongRAG retrieval-unit design change retrieval-side processing?', [('realm',None),('longrag',[0])]),
 ('evaluation-artifacts','How do Ragas and ARES differ in the generated text they use for evaluation: focused answer statements versus synthetic judge-training examples?', [('ragas',[1]),('ares',None)]),
 ('benchmark-metrics','Contrast the ranking metric used in BEIR experiments with the clustering metric used in MTEB.', [('beir',None),('mteb',None)]),
 ('memory-and-generator','Compare the non-parametric memory in original RAG with the pretrained generator initialization described by Fusion-in-Decoder.', [('rag',None),('fid',None)]),
 ('refinement-stages','What are CRAG knowledge-refinement operations, and which pre-retrieval or post-retrieval stages are optimization targets in the RAG survey?', [('crag',None),('survey',None)]),
]

MISSING=[
 ('realm-qwen','2002.08909','What retrieval accuracy does REALM report for Qwen2.5-3B?', ['qwen'],[], 'missing_model'),
 ('colbertv2-llama','2112.01488','What benchmark results does ColBERTv2 report for Llama 3.2 3B?', ['llama'],[], 'missing_model'),
 ('pyserini-gemma','2102.10073','What answer-generation accuracy does the Pyserini paper report for Gemma-2-2B?', ['gemma'],[], 'missing_model'),
 ('beir-phi','2104.08663','What BEIR score is reported for Phi-3-mini in the selected paper?', ['phi-3','phi3','phi 3'],[], 'missing_model'),
 ('ares-energy','2311.09476','How many kilowatt-hours does ARES explicitly report consuming to train its judges?', ['kilowatt','kwh'],[], 'missing_measurement'),
 ('splade-dollar-cost','2107.05720','What total training expense in US dollars is explicitly reported for SPLADE?', ['dollar','usd'],[], 'missing_measurement'),
 ('colbertv2-uncertainty','2112.01488','What 95% confidence interval does ColBERTv2 report for its MRR@10 result?',
  ['confidence interval','bootstrap'],[r'confidence\s+interval'], 'unreported_statistics'),
 ('crag-significance','2401.15884','What statistical significance p-value does CRAG report for its comparison with Self-RAG?',
  ['p-value','p value','pvalue','statistical significance'],[r'p\s*[<=>]\s*0\.'], 'unreported_statistics'),
]


def load_corpus():
    corpus=json.loads((PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))
    snapshot=PROJECT_ROOT/corpus['snapshot']
    return corpus,read_jsonl(snapshot/'new_pages.jsonl'),read_jsonl(snapshot/'chunks.jsonl')


def evidence_group(pid,fact,anchors,pages,chunks):
    alternatives=[]
    for page in pages:
        if page['paper_id']!=pid:continue
        for anchor in anchors:
            span=quotation_span(page['text'],anchor)
            if span is None:continue
            matching=[c['chunk_id'] for c in chunks if c['paper_id']==pid and c['page_number']==page['page_number']
                      and c['start_char']<=span[0] and c['end_char']>=span[1]]
            if not matching:continue
            item={'paper_id':pid,'page_number':page['page_number'],'start_char':span[0],'end_char':span[1],
                  'quote':page['text'][span[0]:span[1]],'chunk_ids':matching}
            if item not in alternatives:alternatives.append(item)
    if not alternatives:raise ValueError(f'No chunk-backed source evidence for {pid}: {anchors}')
    return {'fact':fact,'alternatives':alternatives}


def audit_absence(pid,terms,patterns,pages):
    selected=[p for p in pages if p['paper_id']==pid]
    text='\n'.join(p['text'] for p in selected)
    folded=compact(text)
    if any(compact(term) in folded for term in terms) or any(re.search(pattern,text,re.I) for pattern in patterns):
        raise ValueError(f'Negative-control absence audit failed for {pid}')
    return {'paper_id':pid,'terms_checked':terms,'patterns_checked':patterns,'extracted_pages_checked':len(selected),
            'all_pages_ready':all(p['status']=='ready' for p in selected),
            'limitation':'Term/pattern absence plus source review; extraction can miss visual-only information.'}


def build(corpus,pages,chunks):
    papers={p['paper_id']:Paper.from_dict(p) for p in corpus['papers']}
    groups={name:[evidence_group(pid,fact,anchors,pages,chunks) for fact,anchors in items]
            for name,(pid,question,items) in FACTS.items()}
    questions=[]
    def add(identifier,question,kind,names,evidence,filters,required=(),**extra):
        pids=list(dict.fromkeys(FACTS[n][0] for n in names))
        questions.append({'question_id':'validation40-'+identifier,'question':question,'type':kind,
            'answerable':kind in {'single','comparison'},'filters':filters,'required_papers':list(required),
            'gold_paper_ids':pids,'paper_titles':[papers[p].title for p in pids],
            'reference':' '.join(g['fact'] for g in evidence),'evidence_groups':evidence,
            'split':'final_validation','snapshot_id':corpus['snapshot_id'],
            'annotation_provenance':'assistant source-checked before model answers; independent human review pending',**extra})
    global_names={'realm','rag','beir','hyde','ragas','crag'}
    for name,(pid,question,items) in FACTS.items():
        filters={} if name in global_names else {'paper_ids':[pid]}
        if name=='pyserini':filters={'categories':['cs.IR'],'year_min':2021,'year_max':2021}
        if name=='mteb':filters={'authors':[papers[pid].authors[0]],'year_max':2022}
        if name=='ragtruth':filters={'year_min':2023,'year_max':2023}
        if name=='longrag':filters={'year_min':2024,'year_max':2024}
        add('single-'+name,question,'single',[name],groups[name],filters)
    for identifier,question,parts in COMPARISONS:
        selected=[]
        for name,indices in parts:selected += groups[name] if indices is None else [groups[name][i] for i in indices]
        names=[name for name,indices in parts];pids=[FACTS[name][0] for name in names]
        add('compare-'+identifier,question,'comparison',names,selected,{'paper_ids':pids},pids)
    for identifier,pid,question,terms,patterns,subtype in MISSING:
        name=next(name for name,f in FACTS.items() if f[0]==pid)
        add('missing-'+identifier,question,'missing_evidence',[name],[],{'paper_ids':[pid]},
            absence_audit=audit_absence(pid,terms,patterns,pages),negative_subtype=subtype)
        questions[-1]['reference']='The selected paper does not report the requested model result, measurement, or statistical result.'
        if identifier=='splade-dollar-cost':
            # The only dollar value is an ACM publication boilerplate fee, not
            # a model-training expense. Retain it as a deliberately hard control.
            reviewed=[]
            for page in pages:
                if page['paper_id']!=pid:continue
                for match in re.finditer(r'\$\s*\d+(?:\.\d+)?',page['text']):
                    a=max(0,match.start()-60);b=min(len(page['text']),match.end()+30)
                    quote=page['text'][a:b]
                    if 'ACM ISBN' not in quote:raise ValueError('Unreviewed currency mention in SPLADE')
                    reviewed.append({'paper_id':pid,'page_number':page['page_number'],'start_char':a,'end_char':b,
                        'quote':quote,'interpretation':'ACM publication boilerplate, not a training-cost report.'})
            if not reviewed:raise ValueError('Expected reviewed SPLADE publication-price mention')
            questions[-1]['absence_audit']['related_non_answer_mentions']=reviewed
    empty=[
        ('early-years','Among papers in this library first submitted during 2010–2018, what retrieval methods are described?',{'year_min':2010,'year_max':2018}),
        ('biology','Within indexed q-bio.GN papers, which retrieval algorithms are presented?',{'categories':['q-bio.GN']}),
        ('absent-author','Which retrieval techniques are reported by the selected author in this library?',{'authors':['Validation40_NoIndexedAuthor']}),
        ('pyserini-year-conflict','For the selected Pyserini paper restricted to first submissions from 2024 onward, what search backends are reported?',
         {'paper_ids':['2102.10073'],'year_min':2024}),
    ]
    for identifier,question,filters in empty:
        if eligible_papers(list(papers.values()),RetrievalFilters.from_dict(filters))[0]:raise ValueError('Expected empty scope')
        add('empty-'+identifier,question,'no_match',[],[],filters,absence_audit={'expected_eligible_paper_ids':[]})
        questions[-1]['reference']='No indexed papers satisfy the explicit metadata scope.'
    return questions


def validate(questions,metadata,corpus,pages,chunks):
    if len(questions)!=40 or Counter(q['type'] for q in questions)!={'single':20,'comparison':8,'missing_evidence':8,'no_match':4}:
        raise ValueError('Expected 20 single, 8 comparison, 8 missing-evidence and 4 no-match cases')
    if metadata['questions_hash']!=fingerprint(questions) or metadata['snapshot_id']!=corpus['snapshot_id']:
        raise ValueError('Frozen dataset or corpus snapshot changed')
    if len({q['question_id'] for q in questions})!=40 or len({q['question'].casefold() for q in questions})!=40:
        raise ValueError('Duplicate question IDs or wording')
    if {q['gold_paper_ids'][0] for q in questions if q['type']=='single'}!={p['paper_id'] for p in corpus['papers']}:
        raise ValueError('Each indexed paper must have one single-paper question')
    paper_objects=[Paper.from_dict(p) for p in corpus['papers']]
    page_map={(p['paper_id'],p['page_number']):p['text'] for p in pages}
    by_chunk={c['chunk_id']:c for c in chunks}
    for q in questions:
        eligible={p.paper_id for p in eligible_papers(paper_objects,RetrievalFilters.from_dict(q['filters']))[0]}
        if q['answerable'] and not set(q['gold_paper_ids'])<=eligible:raise ValueError('Evidence excluded by metadata filter')
        if q['type']=='no_match' and eligible:raise ValueError('Empty-scope case has eligible papers')
        if q['type']=='comparison' and set(q['required_papers'])!=set(q['gold_paper_ids']):raise ValueError('Comparison paper mismatch')
        for group in q['evidence_groups']:
            for alternative in group['alternatives']:
                page=page_map[alternative['paper_id'],alternative['page_number']]
                if page[alternative['start_char']:alternative['end_char']]!=alternative['quote']:
                    raise ValueError('Evidence text no longer matches the original PDF span')
                for key in alternative['chunk_ids']:
                    chunk=by_chunk[key]
                    if (chunk['paper_id']!=alternative['paper_id'] or chunk['page_number']!=alternative['page_number']
                        or not chunk['start_char']<=alternative['start_char']<alternative['end_char']<=chunk['end_char']):
                        raise ValueError('Evidence chunk has wrong provenance')
        if q['type']=='missing_evidence':
            audit=q['absence_audit'];audit_absence(audit['paper_id'],audit['terms_checked'],audit['patterns_checked'],pages)
            for mention in audit.get('related_non_answer_mentions',[]):
                text=page_map[mention['paper_id'],mention['page_number']]
                if text[mention['start_char']:mention['end_char']]!=mention['quote']:
                    raise ValueError('Reviewed non-answer mention changed')
    old_texts=set()
    for name in ['questions.jsonl','grounding_questions.jsonl','grounding_confirmation.jsonl']:
        path=PROJECT_ROOT/'evaluation'/name
        if path.exists():old_texts.update(q['question'].casefold() for q in read_jsonl(path))
    if any(q['question'].casefold() in old_texts for q in questions):raise ValueError('Question wording duplicates an earlier set')
    return {'questions':40,'answerable':28,'missing_evidence':8,'empty_scope':4,'papers':len(corpus['papers']),
            'evidence_groups':sum(len(q['evidence_groups']) for q in questions),
            'source_alternatives':sum(len(g['alternatives']) for q in questions for g in q['evidence_groups'])}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true',help='Validate frozen files without rewriting them')
    args=parser.parse_args()
    corpus,pages,chunks=load_corpus();manifest_path=DESTINATION.with_suffix('.manifest.json')
    if DESTINATION.exists():
        questions=read_jsonl(DESTINATION);metadata=json.loads(manifest_path.read_text(encoding='utf-8'))
        print(json.dumps(validate(questions,metadata,corpus,pages,chunks),indent=2));return
    if args.check:raise FileNotFoundError('The 40-question dataset has not been built')
    questions=build(corpus,pages,chunks)
    metadata={'dataset_id':'course_validation_40_v1','questions_hash':fingerprint(questions),'snapshot_id':corpus['snapshot_id'],
        'questions':40,'single':20,'comparison':8,'missing_evidence':8,'no_match':4,'answerable':28,
        'papers':[{'paper_id':p['paper_id'],'title':p['title'],'arxiv_id':p['arxiv_id'],'pdf_sha256':p['pdf_sha256']} for p in corpus['papers']],
        'evaluation_protocol':'New question wording frozen before this dataset is evaluated; same corpus and some related fact topics as development sets.',
        'model_answers_generated_during_construction':False,'human_review':'pending',
        'limitations':['Assistant-source-checked annotations, not independent human gold.',
            'Evidence quotations are not exhaustive relevance judgments.','Term-absence audits can miss visual-only PDF content.',
            'Do not tune the pipeline on these outcomes and continue calling the set independent validation.']}
    stats=validate(questions,metadata,corpus,pages,chunks)
    write_jsonl(DESTINATION,questions);atomic_json(manifest_path,metadata)
    print(json.dumps(stats,indent=2));print(f'Frozen dataset: {DESTINATION}')


if __name__=='__main__':main()
