from __future__ import annotations

import argparse
import json
import hashlib
import shutil
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import PROJECT_ROOT,fingerprint
from assistant.generation import SYSTEM,RESPONSE_SCHEMA
from assistant.schemas import RetrievalFilters
from assistant.service import AssistantService
from .metrics import abstention_counts,citation_ids,evidence_metrics
from .run_retrieval import MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','test'],default='dev')
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--question-id',action='append')
    parser.add_argument('--resume',type=Path,help='Resume a saved run after validating data, model and relevant source code')
    args = parser.parse_args()
    service = AssistantService()
    cases = [q for q in read_jsonl(PROJECT_ROOT/'evaluation/questions.jsonl') if q['split']==args.split
             and (not args.question_id or q['question_id'] in args.question_id)]
    models = service.generator.client.list().model_dump(mode='json')
    digest = next((m['digest'] for m in models['models'] if m['model']==service.config.generator_model),None)
    if not digest:
        raise RuntimeError('Configured generator is not available locally')
    code_hash = fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'assistant').glob('*.py')})
    identity = fingerprint({'questions':cases,'config':service.config.identity(),'system_prompt':SYSTEM,
                            'schema':RESPONSE_SCHEMA,'model_digest':digest,'code_hash':code_hash})
    folder = PROJECT_ROOT/'results'/f'answers-{args.split}-{identity[:12]}'
    if args.resume:
        folder = args.resume.resolve()
        saved = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        expected = fingerprint({'questions':cases,'config':service.config.identity(),'system_prompt':SYSTEM,
            'schema':RESPONSE_SCHEMA,'model_digest':digest,'code_hash':saved['code_hash']})
        if expected != saved['identity'] or saved['split'] != args.split:
            raise ValueError('Cannot resume: benchmark, model or configuration changed')
        for source in (PROJECT_ROOT/'assistant').glob('*.py'):
            # The live UI can be fixed independently of the saved backend experiment.
            if source.name == 'ui.py':
                continue
            checkpoint = folder/'code_checkpoint/assistant'/source.name
            if not checkpoint.exists() or checkpoint.read_bytes() != source.read_bytes():
                raise ValueError(f'Cannot resume after backend change: {source.name}')
        identity,code_hash = saved['identity'],saved['code_hash']
    path = folder/'answers.jsonl'
    rows = read_jsonl(path) if path.exists() else []
    completed = {(r['question_id'],r['mode']) for r in rows}
    print(f'Run {folder.name}: {len(completed)} saved answers; completed pairs will be skipped.',flush=True)
    try:
        for source in (PROJECT_ROOT/'assistant').glob('*.py'):
            target = folder/'code_checkpoint/assistant'/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                shutil.copy2(source,target)
        run_manifest = {'identity':identity,'split':args.split,'cases':len(cases),'modes':args.modes,'status':'running',
            'config_hash':service.config.identity(),'snapshot_id':service.library()['snapshot_id'],
            'model_digest':digest,'code_hash':code_hash,
            'prompt_hash':fingerprint(SYSTEM),'annotation_provenance':'assistant source-checked; human review pending',
            'scoring':'Source-span retrieval and citation IDs automated; factual/citation support requires separate source-checked judgments.'}
        atomic_json(folder/'manifest.json',run_manifest)
        for case in cases:
            for mode in args.modes:
                if (case['question_id'],mode) in completed:
                    continue
                result = service.ask(case['question'],RetrievalFilters.from_dict(case['filters']),mode,
                    case['required_papers'],case['question_id'])
                row = {'question_id':case['question_id'],'split':case['split'],'type':case['type'],'family':case['family'],
                    'answerable':case['answerable'],'mode':mode,'reference':case['reference'],**result,
                    'retrieval_metrics':evidence_metrics(result['sources'],case['evidence_groups'],4),
                    'citation_ids':citation_ids(result['answer'],result['sources']),
                    'abstention':abstention_counts(case['answerable'],result['status']),
                    'factual_correctness_judgment':None,'citation_support_judgment':None}
                rows.append(row)
                write_jsonl(path,rows)
                print(f"{len(rows)}: {case['question_id']} {mode}: {result['status']} ({result['timings_ms']['total']/1000:.1f}s)",flush=True)
        print(f'Saved actual answer traces: {folder}',flush=True)
        run_manifest.update({'status':'complete','actual_answers':len(rows)})
        atomic_json(folder/'manifest.json',run_manifest)
    finally:
        service.close()


if __name__=='__main__':
    main()
