from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import PROJECT_ROOT,fingerprint
from assistant.context import pack_context
from assistant.schemas import RetrievalFilters
from assistant.service import AssistantService
from .metrics import covers,evidence_metrics,graded_metrics

MODES = ('bm25','dense','hybrid','hybrid_rerank')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',choices=['all','dev','test'],default='all')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/retrieval-primary')
    args = parser.parse_args()
    service = AssistantService()
    questions = read_jsonl(PROJECT_ROOT/'evaluation/questions.jsonl')
    selected = [q for q in questions if args.split=='all' or q['split']==args.split]
    rows,pools = [],[]
    qrels = read_jsonl(PROJECT_ROOT/'evaluation/qrels.jsonl') if (PROJECT_ROOT/'evaluation/qrels.jsonl').exists() else []
    semantic_qrels = {}
    for judgment in qrels:
        semantic_qrels.setdefault(judgment['question_id'],{})[judgment['chunk_id']] = judgment['grade']
    try:
        snapshot = service._open()
        for judgment in qrels:
            actual = snapshot.by_id[judgment['chunk_id']]
            if hashlib.sha256(actual.text.encode()).hexdigest() != judgment['text_sha256']:
                raise ValueError('Graded relevance judgment refers to changed passage text')
        for question in selected:
            groups = question['evidence_groups']
            # Operational, reproducible known-span criterion, not exhaustive semantic relevance.
            span_qrels = {c.chunk_id:2 if any(covers(c.to_dict(),a) for g in groups for a in g['alternatives']) else 0
                          for c in snapshot.chunks} if groups else {}
            for mode in MODES:
                hits,trace = service.retrieve(question['question'],RetrievalFilters.from_dict(question['filters']),mode)
                sources,packing = pack_context(question['question'],hits,service.generator.count,service.config,question['required_papers'])
                trace.data.update({'question_id':question['question_id'],'sources':sources,'context':packing,'status':'retrieval_only'})
                path = trace.save()
                stages = {name:items for name,items in trace.data.get('retrieval',{}).items()}
                stages['ranked'] = [h.to_dict() for h in hits]
                stages['context'] = sources
                measurements = {}
                for name,items in stages.items():
                    k = service.config.context_chunks if name=='context' else service.config.branch_top_k
                    measurements[name] = evidence_metrics(items,groups,k)
                    for cutoff in (4,10):
                        measurements[name][f'known_span_at_{cutoff}'] = graded_metrics(items,span_qrels,cutoff)
                eligible = set(trace.data['eligibility'].get('eligible_paper_ids',[]))
                violation_count = sum(s['paper_id'] not in eligible for s in sources)
                rows.append({'question_id':question['question_id'],'split':question['split'],'type':question['type'],
                    'answerable':question['answerable'],'mode':mode,'metrics':measurements,
                    'semantic_at_4':graded_metrics(sources,semantic_qrels.get(question['question_id'],{}),4),
                    'constraint_violations':violation_count,'eligible_papers':len(eligible),
                    'missing_required_papers':packing['missing_required_papers'],'timings_ms':trace.data['timings_ms'],
                    'trace_path':path,'ranked_chunk_ids':[h.chunk.chunk_id for h in hits],
                    'context_chunk_ids':[s['chunk_id'] for s in sources]})
                for hit in hits[:10]:
                    pools.append({'question_id':question['question_id'],'mode':mode,**hit.to_dict()})
                print(f"{question['question_id']} {mode}: evidence Recall@4={measurements['context']['evidence_recall']}",flush=True)
            write_jsonl(args.output/'measurements.jsonl',rows)
            write_jsonl(args.output/'pool_for_semantic_judging.jsonl',pools)
        atomic_json(args.output/'manifest.json',{'snapshot_id':service.library()['snapshot_id'],
            'questions_hash':fingerprint(selected),'config_hash':service.config.identity(),'cases':len(selected),
            'definitions':{'known_span_ndcg_precision':'Exact coverage of annotated source spans; not exhaustive semantic relevance. Grade 2=contains an annotated supporting quote; 0=does not contain that span. Other valid evidence may be unannotated.',
                           'evidence_recall':'Verified evidence groups, alternatives supported by a returned passage.'},
            'semantic_qrels_status':'58 assistant source-checked pooled judgments across six diagnostic questions; independent human review pending',
            'qrels_hash':fingerprint(qrels)})
    finally:
        service.close()


if __name__=='__main__':
    main()
