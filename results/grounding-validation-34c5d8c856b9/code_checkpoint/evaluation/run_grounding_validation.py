"""Frozen-v1 versus grounded-v2 validation, with resumable sequential local runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from assistant.config import PROJECT_ROOT,Config,fingerprint
from assistant.artifacts import atomic_json,read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['both','baseline','grounded'],default='both')
    args=parser.parse_args()
    questions=PROJECT_ROOT/'evaluation/grounding_questions.jsonl'
    cases=read_jsonl(questions)
    frozen=json.loads(questions.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if frozen['questions_hash']!=fingerprint(cases):raise ValueError('Frozen validation questions changed')
    checkpoint=PROJECT_ROOT/'docs/baseline/pre_grounding'
    inventory=json.loads((checkpoint/'inventory.json').read_text(encoding='utf-8'))
    for record in inventory['files']:
        if hashlib.sha256((checkpoint/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Preserved pre-grounding package changed')
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'assistant').glob('*.py')}
    identity={'questions_hash':fingerprint(cases),'config_hash':Config.load().identity(),'code_hash':fingerprint(code),
              'baseline_checkpoint_hash':fingerprint(inventory)}
    output=PROJECT_ROOT/'results'/('grounding-validation-'+fingerprint(identity)[:12])
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'manifest.json',{'identity':identity,'cases':len(cases),'status':'running',
        'baseline_checkpoint':'docs/baseline/pre_grounding','mode':'hybrid_rerank',
        'scope':'New questions on the original corpus; assistant-source-checked labels; human review pending'})
    for arm in (['baseline','grounded'] if args.arm=='both' else [args.arm]):
        subprocess.run([sys.executable,'-X','utf8','-u','-m','evaluation.grounding_worker','--arm',arm,
            '--questions',str(questions),'--output',str(output/(arm+'_answers.jsonl'))],cwd=PROJECT_ROOT,check=True)
    summaries=[]
    for arm in ['baseline','grounded']:
        path=output/(arm+'_answers.jsonl')
        if not path.exists():continue
        rows=read_jsonl(path)
        positive=[r for r in rows if r['answerable']]
        negative=[r for r in rows if r['type']=='missing_evidence']
        summaries.append({'arm':arm,'cases':len(rows),'answerable_cases':len(positive),
            'answered_answerable':sum(r['status']=='answered' for r in positive),
            'false_refusals':sum(r['status'] in {'no_match','insufficient_evidence'} for r in positive),
            'missing_evidence_cases':len(negative),'answered_missing_evidence':sum(r['status']=='answered' for r in negative),
            'correct_missing_evidence_refusals':sum(r['status'] in {'no_match','insufficient_evidence'} for r in negative),
            'runtime_failures':sum(r['status'] in {'error','invalid_output'} for r in rows),
            'mean_seconds':sum(r['timings_ms']['total']/1000 for r in rows)/len(rows)})
    atomic_json(output/'summary.json',{'arms':summaries,'limitations':['Answer/refusal counts are not independent factuality scores.',
        'Original test questions were already inspected; this validation uses newly frozen questions on the same corpus.']})
    atomic_json(output/'manifest.json',{'identity':identity,'cases':len(cases),
        'status':'complete' if len(summaries)==2 and all(r['cases']==len(cases) for r in summaries) else 'partial',
        'baseline_checkpoint':'docs/baseline/pre_grounding','mode':'hybrid_rerank',
        'annotation_provenance':'assistant source-checked; independent human review pending'})
    print(f'Validation artifacts: {output}',flush=True)


if __name__=='__main__':main()
