"""Freeze new source-checked validation questions before observing their answers."""
from __future__ import annotations
import json
from assistant.config import PROJECT_ROOT, fingerprint
from assistant.artifacts import atomic_json, read_jsonl, write_jsonl
from assistant.schemas import Paper, RetrievalFilters
from assistant.filters import eligible_papers
from assistant.grounding import quotation_span


POSITIVES = [
 ('dpr-negatives','2004.04906',3,'How does DPR reuse passages from other questions in a training mini-batch?',
  'In-batch negatives','DPR reuses the other relevant passages in a mini-batch as negative passages for each question.'),
 ('fid-architecture','2007.01282',3,'Where does Fusion-in-Decoder process passages independently, and where does it process them jointly?',
  'By processing passages independently in the encoder, but jointly in the decoder',
  'FiD processes passages independently in the encoder and jointly in the decoder.'),
 ('beir-size','2104.08663',1,'How many publicly available datasets were selected in the frozen BEIR paper?',
  '18 publicly available datasets','The frozen BEIR paper reports 18 publicly available datasets.'),
 ('mteb-retrieval-metric','2210.07316',4,'Which main metric does MTEB use for its retrieval task?',
  'nDCG@10 serves as the main metric','The main retrieval metric is nDCG@10.'),
 ('splade-flops','2107.05720',3,'What computational quantity does the SPLADE paper say the FLOPS regularizer approximates?',
  'a smooth relaxation of the average number of floating-point operations necessary to compute the score of a document',
  'The FLOPS regularizer relaxes the average number of floating-point operations needed to score a document.'),
 ('crag-actions','2401.15884',3,'What three retrieval actions are triggered by the confidence degrees in CRAG?',
  '{Correct, Incorrect, Ambiguous}','The three actions are Correct, Incorrect and Ambiguous.'),
 ('selfrag-utility','2310.11511',5,'Which Self-RAG token predicts the overall utility of a response?',
  'predicts the overall utility token ISUSE','ISUSE predicts the overall utility of a response.'),
 ('colbert-aggregation','2004.12832',4,'How does ColBERT aggregate the maximum similarity scores across query terms?',
  'combine the outputs via summation','ColBERT sums the per-query-term maximum similarity scores.'),
]
NEGATIVES = [
 ('gemma-result','2004.04906','What accuracy does the DPR paper report for Gemma-2-2B?',['gemma']),
 ('olmo-result','2004.12832','What benchmark accuracy does the ColBERT paper report for OLMo-7B?',['olmo']),
 ('chatgpt-result','2107.05720','What fine-tuning results does the SPLADE paper report for ChatGPT?',['chatgpt']),
 ('hyde-energy','2212.10496','What exact energy consumption in kilowatt-hours does the HyDE paper report?',['kilowatt','kwh']),
 ('dpr-financial-cost','2004.04906','What exact financial cost in US dollars does the DPR paper report for training?',['dollar','usd']),
 ('colbert-energy','2004.12832','What measured energy consumption in kilowatt-hours does ColBERT report?',['kilowatt','kwh']),
]


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    positives=POSITIVES;negatives=NEGATIVES
    destination=PROJECT_ROOT/'evaluation'/('grounding_confirmation.jsonl' if args.confirmation else 'grounding_questions.jsonl')
    if args.confirmation:
        positives=[
            ('dpr-corpus-size','2004.04906',2,'Approximately how many passages does DPR say are in its Wikipedia collection?',
             '21 million passages','DPR reports approximately 21 million passages.'),
            ('fid-batch-size','2007.01282',4,'What batch size does the FiD paper report for fine-tuning its generative models?',
             'with a batch size of 64','The reported training batch size is 64.'),
            ('mteb-pair-metric','2210.07316',3,'What is the main metric for pair classification in MTEB?',
             'The average precision score based on cosine similarity is the main metric',
             'Average precision based on cosine similarity is the main pair-classification metric.'),
            ('crag-discard','2401.15884',3,'What happens to retrieved documents when CRAG triggers its Incorrect action?',
             'If the action Incorrect is triggered, the retrieved documents will be discarded',
             'The retrieved documents are discarded when CRAG triggers Incorrect.')]
        negatives=[
            ('phi-result','2004.04906','What accuracy does the DPR paper report for Phi-3-mini?',['phi-3','phi3','phi 3']),
            ('starcoder-result','2212.10496','What retrieval results does HyDE report for StarCoder2-3B?',['starcoder'])]
    if destination.exists():
        print('Existing grounding validation is frozen; not rewriting it.')
        return
    manifest=json.loads((PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))
    snapshot=PROJECT_ROOT/manifest['snapshot']
    pages=read_jsonl(snapshot/'new_pages.jsonl')
    chunks=read_jsonl(snapshot/'chunks.jsonl')
    papers=[Paper.from_dict(p) for p in manifest['papers']]
    questions=[]
    for name,paper_id,page_no,question,quote,reference in positives:
        page=next(p for p in pages if (p['paper_id'],p['page_number'])==(paper_id,page_no))
        span=quotation_span(page['text'],quote)
        if span is None: raise ValueError(f'New evidence not found: {name}')
        alternatives=[]
        for c in chunks:
            if c['paper_id']==paper_id and c['page_number']==page_no and c['start_char']<=span[0] and c['end_char']>=span[1]:
                alternatives.append(c['chunk_id'])
        if not alternatives: raise ValueError(f'Evidence does not fit a frozen chunk: {name}')
        questions.append({'question_id':'grounding-'+name,'type':'answerable','answerable':True,
            'question':question,'reference':reference,'filters':{'paper_ids':[paper_id]},'required_papers':[],
            'evidence_groups':[{'fact':reference,'alternatives':[{'paper_id':paper_id,'page_number':page_no,
                'start_char':span[0],'end_char':span[1],'quote':page['text'][span[0]:span[1]],'chunk_ids':alternatives}]}]})
    for name,paper_id,question,terms in negatives:
        full='\n'.join(p['text'] for p in pages if p['paper_id']==paper_id).casefold()
        if any(t in full for t in terms):raise ValueError(f'Absence audit failed: {name}')
        questions.append({'question_id':'grounding-'+name,'type':'missing_evidence','answerable':False,
            'question':question,'reference':'The selected paper does not report the requested result.',
            'filters':{'paper_ids':[paper_id]},'required_papers':[],'evidence_groups':[],
            'absence_audit':{'terms':terms,'all_extracted_pages_checked':True}})
    empty=[('absent-author',{'authors':['GroundingValidationNoSuchAuthor']}),
           ('conflicting-scope',{'paper_ids':['2210.07316'],'year_max':2020})]
    if args.confirmation:empty=[('confirmation-author',{'authors':['GroundingConfirmationNoSuchAuthor']}),
                                ('confirmation-year',{'paper_ids':['2104.08663'],'year_max':2019})]
    for name,filters in empty:
        if eligible_papers(papers,RetrievalFilters.from_dict(filters))[0]:raise ValueError('Expected empty scope')
        questions.append({'question_id':'grounding-'+name,'type':'no_match','answerable':False,
            'question':'What retrieval method do papers in this scope describe?','reference':'No indexed papers satisfy the scope.',
            'filters':filters,'required_papers':[],'evidence_groups':[]})
    for q in questions:
        q.update({'split':'new_validation','snapshot_id':manifest['snapshot_id'],
                  'annotation_provenance':'assistant source-checked before answer generation; human review pending'})
    write_jsonl(destination,questions)
    atomic_json(destination.with_suffix('.manifest.json'),{'questions':len(questions),'answerable':len(positives),
        'missing_evidence':len(negatives),'no_match':len(empty),'questions_hash':fingerprint(questions),'snapshot_id':manifest['snapshot_id'],
        'scope':'New questions/facts on the same frozen corpus; not new papers or independent human labels',
        'frozen_before_answer_generation':True})
    print(f'Frozen {len(questions)} source-checked validation questions.')


if __name__=='__main__':main()
