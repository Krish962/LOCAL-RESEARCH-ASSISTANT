"""Functional/coverage acceptance, explicitly separate from factual accuracy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from assistant.config import PROJECT_ROOT,Config,fingerprint
from assistant.artifacts import atomic_json,read_jsonl


def assess(rows,minimum_coverage=.75):
    positive=[r for r in rows if r['answerable']]
    negative=[r for r in rows if not r['answerable']]
    answered=sum(r['status']=='answered' for r in positive)
    coverage=answered/len(positive) if positive else None
    failures=sum(r['status'] in {'error','invalid_output'} for r in rows)
    risks=sum(r['status']=='answered' for r in negative)
    correct_refusals=sum(r['status'] in {'no_match','insufficient_evidence'} for r in negative)
    passed=(coverage is not None and coverage>=minimum_coverage and failures==0 and risks==0
            and correct_refusals==len(negative))
    return {'passed':passed,'answerable_cases':len(positive),'answered_answerable':answered,
        'coverage':coverage,'minimum_coverage':minimum_coverage,'unanswerable_cases':len(negative),
        'answered_unanswerable':risks,'correct_refusals':correct_refusals,'runtime_failures':failures,
        'interpretation':'Coverage and known-negative controls only; not independently reviewed factual accuracy.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--minimum-coverage',type=float,default=.75)
    parser.add_argument('--questions',type=Path,nargs='+',help='Evaluate specific frozen question sets')
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/summary/grounding_acceptance.json')
    args=parser.parse_args()
    if not 0<args.minimum_coverage<=1:parser.error('Coverage must be in (0,1]')
    code=fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'assistant').glob('*.py')})
    config=Config.load().identity()
    reports=[]
    paths=args.questions or [PROJECT_ROOT/'evaluation'/name for name in ['grounding_questions.jsonl','grounding_confirmation.jsonl']]
    for qpath in paths:
        name=qpath.name
        expected=fingerprint(read_jsonl(qpath))
        matching=[]
        for manifest in (PROJECT_ROOT/'results').glob('grounding-validation-*/manifest.json'):
            data=json.loads(manifest.read_text(encoding='utf-8'))
            identity=data['identity']
            if data['status']=='complete' and identity['questions_hash']==expected and identity['code_hash']==code and identity['config_hash']==config:
                matching.append(manifest)
        if not matching:
            reports.append({'questions':name,'passed':False,'reason':'No completed run matches the current code/configuration.'})
            continue
        run=max(matching,key=lambda p:p.stat().st_mtime).parent
        rows=read_jsonl(run/'grounded_answers.jsonl')
        report=assess(rows,args.minimum_coverage)
        report.update({'questions':name,'run':str(run)})
        if len(rows)!=len(read_jsonl(qpath)):
            report.update({'passed':False,'reason':'Incomplete saved answers'})
        reports.append(report)
    passed=all(r['passed'] for r in reports)
    result={'status':'coverage_checks_passed' if passed else 'quality_review_required','sets':reports,
        'limitation':'Coverage is not factual accuracy. Check each dataset protocol and review answers against sources.'}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
