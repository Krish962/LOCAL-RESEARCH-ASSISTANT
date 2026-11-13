"""Operational grounding metrics, with explicit denominators and separate review labels."""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import re
from pathlib import Path
from assistant.config import PROJECT_ROOT,fingerprint
from assistant.artifacts import atomic_json,read_jsonl
from .metrics import citation_ids,evidence_metrics
from assistant.schemas import Paper,RetrievalFilters
from assistant.filters import eligible_papers


def mean(values):
    return statistics.mean(values) if values else None


def saved_path(value):
    """Resolve saved local paths after relocating the project directory."""
    path = Path(value)
    if path.exists():
        return path
    if not path.is_absolute():
        return PROJECT_ROOT / path
    try:
        relative = path.relative_to(PROJECT_ROOT.parent / 'AI-Assistant')
    except ValueError:
        return path
    return PROJECT_ROOT / relative


def comparison_paper_recall(row,question):
    required=set(question.get('required_papers',[]))
    if not required:return None
    if row['status']!='answered':return 0.
    labels=set(re.findall(r'\[(S\d+)\]',row['answer']))
    cited={s['paper_id'] for s in row['sources'] if s['label'] in labels}
    return len(required & cited)/len(required)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path)
    parser.add_argument('--questions',type=Path)
    args=parser.parse_args()
    manifests=list((PROJECT_ROOT/'results').glob('grounding-validation-*/manifest.json'))
    if args.questions:
        wanted=fingerprint(read_jsonl(args.questions))
        manifests=[p for p in manifests if json.loads(p.read_text(encoding='utf-8'))['identity']['questions_hash']==wanted]
    run=args.run or max(manifests,key=lambda p:p.stat().st_mtime).parent
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    question_path=saved_path(args.questions or manifest.get('questions_path',PROJECT_ROOT/'evaluation/grounding_questions.jsonl'))
    questions={q['question_id']:q for q in read_jsonl(question_path)}
    corpus=json.loads((PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))
    papers=[Paper.from_dict(p) for p in corpus['papers']]
    labels_path=run/'assistant_source_judgments.jsonl'
    labels={(r['question_id'],r['arm']):r for r in read_jsonl(labels_path)} if labels_path.exists() else {}
    summaries=[]
    for arm in ['baseline','grounded']:
        path=run/(arm+'_answers.jsonl')
        if not path.exists():continue
        rows=read_jsonl(path)
        positives=[r for r in rows if r['answerable']]
        missing=[r for r in rows if r['type']=='missing_evidence']
        drafts=[]; traces=[]
        for row in rows:
            trace=json.loads(saved_path(row['trace_path']).read_text(encoding='utf-8'))
            traces.append(trace);drafts.extend(trace.get('generation_attempts',[]))
        supports=[a['grounding'] for a in drafts if a.get('grounding')]
        extractive_checks=[]
        for t in traces:
            if 'extractive_attempts' in t:
                extractive_checks.extend(a['support'] for a in t['extractive_attempts'] if a.get('support'))
            elif t.get('extractive_grounding'):
                extractive_checks.append(t['extractive_grounding']['support'])
        errors=[e for decision in supports for c in decision['claims'] for e in c['errors']]
        judged=[labels[(r['question_id'],arm)] for r in rows if (r['question_id'],arm) in labels]
        answered=[r for r in rows if r['status']=='answered']
        singles=[r for r in positives if r['type'] in {'single','answerable','metadata'}]
        comparisons=[r for r in rows if r['type']=='comparison']
        violations=0
        for r in rows:
            allowed={p.paper_id for p in eligible_papers(papers,RetrievalFilters.from_dict(questions[r['question_id']]['filters']))[0]}
            violations+=sum(s['paper_id'] not in allowed for s in r['sources'])
        summary={'arm':arm,'completed_cases':len(rows),'answerable_cases':len(positives),
            'answerable_answer_rate':mean([int(r['status']=='answered') for r in positives]),
            'false_refusals':sum(r['status'] in {'insufficient_evidence','no_match'} for r in positives),
            'single_paper_answerable_cases':len(singles),
            'single_paper_answer_rate':mean([int(r['status']=='answered') for r in singles]),
            'comparison_cases':len(comparisons),
            'comparison_answer_rate':mean([int(r['status']=='answered') for r in comparisons]),
            'comparison_cited_paper_recall':mean([comparison_paper_recall(r,questions[r['question_id']]) for r in comparisons]),
            'metadata_constraint_violations':violations,
            'missing_evidence_cases':len(missing),
            'answered_missing_evidence':sum(r['status']=='answered' for r in missing),
            'missing_evidence_refusal_rate':mean([int(r['status'] in {'insufficient_evidence','no_match'}) for r in missing]),
            'correct_empty_scope_refusals':sum(r['status']=='no_match' for r in rows if r['type']=='no_match'),
            'runtime_failures':sum(r['status'] in {'error','invalid_output'} for r in rows),
            'mean_seconds_all':mean([r['timings_ms']['total']/1000 for r in rows]),
            'mean_seconds_answerable':mean([r['timings_ms']['total']/1000 for r in positives]),
            'mean_claim_verification_seconds':mean([sum(v for k,v in r['timings_ms'].items() if k.startswith(('claim_verification','extractive_verification')))/1000 for r in answered]),
            'literal_citation_id_validity':mean([citation_ids(r['answer'],r['sources'])['validity'] for r in answered
                if citation_ids(r['answer'],r['sources'])['validity'] is not None]),
            'annotated_span_recall_at_4':mean([evidence_metrics(r['sources'],questions[r['question_id']]['evidence_groups'])['evidence_recall'] for r in positives]),
            'pre_generation_identifier_refusals':sum(not t.get('grounding',{}).get('question_coverage',{}).get('passed',True) for t in traces),
            'generated_drafts':len(drafts),'support_checked_drafts':len(supports),
            'evidence_selection_calls':sum(sum(k.startswith('evidence_selection') for k in t.get('timings_ms',{})) for t in traces),
            'pruned_excerpts':sum(len(a.get('pruned_evidence_ids',[])) for t in traces for a in t.get('extractive_attempts',[])),
            'extractive_answers':sum(r.get('answer_mode')=='extractive' for r in answered),
            'extractive_support_checks':len(extractive_checks),
            'extractive_support_rejections':sum(not c['passed'] for c in extractive_checks),
            'support_rejected_drafts':sum(not s['passed'] for s in supports),
            'quotation_rejection_events':sum('verbatim' in e for e in errors),
            'number_rejection_events':sum('number absent' in e for e in errors),
            'identifier_rejection_events':sum('identifier absent' in e for e in errors),
            'entailment_rejection_events':sum('entail' in e for e in errors),
            'contradiction_rejection_events':sum('contradicts' in e for e in errors),
            'assistant_reviewed_cases':len(judged),
            'assistant_source_checked_answer_fulfillment_0_to_2':mean([r['correctness'] for r in judged]),
            'assistant_source_checked_claim_support':mean([r['cited_claim_support'] for r in judged if r.get('cited_claim_support') is not None])}
        summaries.append(summary)
    atomic_json(run/'operational_metrics.json',{'arms':summaries,'definitions':{
        'answerable_answer_rate':'Coverage, not factual accuracy.',
        'support_checked_drafts':'Internal policy acceptance checks, not an independent quality judgment.',
        'comparison_cited_paper_recall':'Fraction of required paper IDs actually cited; not semantic citation correctness.',
        'assistant_review':'Separate source review by the assistant if supplied; not independent human evaluation.',
        'latency':'Sequential CPU run; generation length and caching differ between versions.'}})
    with (run/'operational_metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
