"""Frozen-v1 versus grounded-v2 validation, with resumable sequential local runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path
from assistant.config import PROJECT_ROOT,Config,fingerprint
from assistant.artifacts import atomic_json,read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['both','baseline','grounded'],default='both')
    parser.add_argument('--questions',type=Path,default=PROJECT_ROOT/'evaluation/grounding_questions.jsonl')
    parser.add_argument('--reuse-baseline',type=Path)
    args=parser.parse_args()
    questions=args.questions.resolve()
    cases=read_jsonl(questions)
    frozen=json.loads(questions.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if frozen['questions_hash']!=fingerprint(cases):raise ValueError('Frozen validation questions changed')
    checkpoint=PROJECT_ROOT/'docs/baseline/pre_grounding'
    inventory=json.loads((checkpoint/'inventory.json').read_text(encoding='utf-8'))
    for record in inventory['files']:
        if hashlib.sha256((checkpoint/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Preserved pre-grounding package changed')
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'assistant').glob('*.py')}
    import ollama
    cfg=Config.load()
    tags=ollama.Client(host=cfg.ollama_url).list().model_dump(mode='json')['models']
    digest=next(m['digest'] for m in tags if m['model']==cfg.generator_model)
    identity={'questions_hash':fingerprint(cases),'config_hash':Config.load().identity(),'code_hash':fingerprint(code),
              'baseline_checkpoint_hash':fingerprint(inventory),'generator_digest':digest,
              'worker_hash':hashlib.sha256(Path(__file__).with_name('grounding_worker.py').read_bytes()).hexdigest()}
    output=PROJECT_ROOT/'results'/('grounding-validation-'+fingerprint(identity)[:12])
    output.mkdir(parents=True,exist_ok=True)
    provenance=None
    if args.reuse_baseline:
        previous=json.loads((args.reuse_baseline/'manifest.json').read_text(encoding='utf-8'))
        if any(previous['identity'][key]!=identity[key] for key in ['questions_hash','baseline_checkpoint_hash']):
            raise ValueError('The baseline has different questions or a changed preserved package')
        old_worker=args.reuse_baseline/'code_checkpoint/evaluation/grounding_worker.py'
        if not old_worker.exists() or old_worker.read_bytes()!=Path(__file__).with_name('grounding_worker.py').read_bytes():
            raise ValueError('The baseline worker code changed')
        original=json.loads((PROJECT_ROOT/'results/answers-test-754021b26646/manifest.json').read_text(encoding='utf-8'))
        if original['model_digest']!=digest:raise ValueError('Baseline model digest changed')
        source=args.reuse_baseline/'baseline_answers.jsonl'
        if {r['question_id'] for r in read_jsonl(source)}!={q['question_id'] for q in cases}:
            raise ValueError('Baseline is incomplete')
        destination=output/'baseline_answers.jsonl'
        if destination.exists():
            if destination.read_bytes()!=source.read_bytes():raise ValueError('Existing baseline differs')
        else:shutil.copy2(source,destination)
        provenance={'source_run':str(args.reuse_baseline),'answers_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    for source in [*list((PROJECT_ROOT/'assistant').glob('*.py')),PROJECT_ROOT/'configs/default.json',
                   Path(__file__),Path(__file__).with_name('grounding_worker.py'),questions]:
        destination=output/'code_checkpoint'/source.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():shutil.copy2(source,destination)
        elif destination.read_bytes()!=source.read_bytes():raise ValueError('A saved source checkpoint differs')
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
        'baseline_reused_from':provenance,'questions_path':str(questions),
        'annotation_provenance':'assistant source-checked; independent human review pending'})
    print(f'Validation artifacts: {output}',flush=True)


if __name__=='__main__':main()
