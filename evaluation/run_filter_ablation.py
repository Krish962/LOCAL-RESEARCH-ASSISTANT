from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import PROJECT_ROOT
from assistant.context import pack_context
from assistant.filters import eligible_papers
from assistant.schemas import Paper,RetrievalFilters
from assistant.service import AssistantService
from .metrics import evidence_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare retrieval with metadata filters enabled and disabled')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'results/filter-reproduction')
    args = parser.parse_args()
    service = AssistantService()
    rows = []
    try:
        papers = [Paper.from_dict(p) for p in service.library()['papers']]
        cases = [q for q in read_jsonl(PROJECT_ROOT/'evaluation/questions.jsonl') if q['type'] in {'metadata','no_match'}]
        for case in cases:
            original = RetrievalFilters.from_dict(case['filters'])
            expected,_ = eligible_papers(papers,original)
            expected_ids = {p.paper_id for p in expected}
            for mode in ('dense','hybrid','hybrid_rerank'):
                for enabled in (False,True):
                    used = original if enabled else RetrievalFilters()
                    hits,trace = service.retrieve(case['question'],used,mode)
                    sources,packing = pack_context(case['question'],hits,service.generator.count,service.config)
                    violations = sum(s['paper_id'] not in expected_ids for s in sources)
                    rows.append({'question_id':case['question_id'],'split':case['split'],'mode':mode,'filters_enabled':enabled,
                        'original_filters':asdict(original),'returned':len(sources),'constraint_violations':violations,
                        'violation_rate':violations/len(sources) if sources else None,
                        'correct_empty_match':not sources if not expected_ids else None,
                        'evidence':evidence_metrics(sources,case['evidence_groups'],4),'timings_ms':trace.data['timings_ms']})
        path = args.output
        write_jsonl(path/'measurements.jsonl',rows)
        atomic_json(path/'manifest.json',{'cases':len(cases),'arms':6,'status':'complete',
            'scope':'Experimental disabling of user filters only; active corpus/readiness remain enforced. Production never relaxes filters.'})
        print(f'Saved {len(rows)} actual filtering comparisons.')
    finally:
        service.close()


if __name__=='__main__':
    main()
