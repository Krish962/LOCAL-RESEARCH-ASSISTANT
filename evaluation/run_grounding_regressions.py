"""Recheck the four previously observed failures across all retrieval modes."""
from assistant.config import PROJECT_ROOT,Config
from assistant.artifacts import read_jsonl,write_jsonl,atomic_json
from assistant.service import AssistantService
from assistant.schemas import RetrievalFilters


def main():
    service=AssistantService()
    rows=[]
    try:
        for case in read_jsonl(PROJECT_ROOT/'evaluation/questions.jsonl'):
            if case['type']!='unanswerable':continue
            for mode in ['bm25','dense','hybrid','hybrid_rerank']:
                result=service.ask(case['question'],RetrievalFilters.from_dict(case['filters']),mode,
                    case['required_papers'],case['question_id'])
                rows.append({'question_id':case['question_id'],'mode':mode,**result})
                print(case['question_id'],mode,result['status'],flush=True)
        path=PROJECT_ROOT/'results/grounding-regressions'
        write_jsonl(path/'answers.jsonl',rows)
        atomic_json(path/'manifest.json',{'cases':len(rows),'config_hash':Config.load().identity(),
            'correct_refusals':sum(r['status']=='insufficient_evidence' for r in rows),
            'runtime_failures':sum(r['status']=='error' for r in rows),
            'scope':'Previously inspected failures; regression tests, not a new held-out benchmark'})
        if len(rows)!=16 or any(r['status']!='insufficient_evidence' for r in rows):
            raise RuntimeError('Previously observed unsupported-answer regressions remain')
    finally:service.close()


if __name__=='__main__':main()
