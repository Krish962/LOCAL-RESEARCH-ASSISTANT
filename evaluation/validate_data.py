"""Verify the frozen 40-question set against indexed PDF spans and metadata."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from assistant.artifacts import read_jsonl
from assistant.config import PROJECT_ROOT, fingerprint
from assistant.filters import eligible_papers
from assistant.grounding import compact
from assistant.schemas import Paper, RetrievalFilters

DESTINATION = PROJECT_ROOT / 'evaluation/final_validation_40.jsonl'


def load_corpus():
    corpus=json.loads((PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))
    snapshot=PROJECT_ROOT/corpus['snapshot']
    return corpus,read_jsonl(snapshot/'new_pages.jsonl'),read_jsonl(snapshot/'chunks.jsonl')


def audit_absence(pid,terms,patterns,pages):
    selected=[p for p in pages if p['paper_id']==pid]
    text='\n'.join(p['text'] for p in selected)
    folded=compact(text)
    if any(compact(term) in folded for term in terms) or any(re.search(pattern,text,re.I) for pattern in patterns):
        raise ValueError(f'Negative-control absence audit failed for {pid}')
    return {'paper_id':pid,'terms_checked':terms,'patterns_checked':patterns,'extracted_pages_checked':len(selected),
            'all_pages_ready':all(p['status']=='ready' for p in selected),
            'limitation':'Term/pattern absence plus source review; extraction can miss visual-only information.'}


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
    for name in ['questions.jsonl']:
        path=PROJECT_ROOT/'evaluation'/name
        if path.exists():old_texts.update(q['question'].casefold() for q in read_jsonl(path))
    if any(q['question'].casefold() in old_texts for q in questions):raise ValueError('Question wording duplicates an earlier set')
    return {'questions':40,'answerable':28,'missing_evidence':8,'empty_scope':4,'papers':len(corpus['papers']),
            'evidence_groups':sum(len(q['evidence_groups']) for q in questions),
            'source_alternatives':sum(len(g['alternatives']) for q in questions for g in q['evidence_groups'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate the stored dataset without inference')
    parser.parse_args()
    corpus, pages, chunks = load_corpus()
    questions = read_jsonl(DESTINATION)
    metadata = json.loads(DESTINATION.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    print(json.dumps(validate(questions, metadata, corpus, pages, chunks), indent=2))

if __name__ == '__main__':
    main()
