"""Aggregate actual measurements; unreviewed factuality remains unscored."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl
from assistant.config import PROJECT_ROOT


def average(values):
    actual = [float(v) for v in values if v is not None]
    return statistics.mean(actual) if actual else None


def main():
    output = PROJECT_ROOT/'results/summary'
    output.mkdir(parents=True,exist_ok=True)
    retrieval_path = PROJECT_ROOT/'results/retrieval-final/measurements.jsonl'
    retrieval = read_jsonl(retrieval_path) if retrieval_path.exists() else []
    # Existing preliminary runs remain separate; never relabel them final.
    final_runs = [p for p in (PROJECT_ROOT/'results').glob('answers-test-*/manifest.json')
                  if json.loads(p.read_text(encoding='utf-8')).get('status')=='complete']
    answer_folder = max(final_runs,key=lambda p:p.stat().st_mtime).parent if final_runs else None
    answers = read_jsonl(answer_folder/'answers.jsonl') if answer_folder else []
    judgments = read_jsonl(answer_folder/'source_judgments.jsonl') if answer_folder and (answer_folder/'source_judgments.jsonl').exists() else []
    by_key = {(j['question_id'],j['mode']):j for j in judgments}
    summaries = []
    for mode in ('bm25','dense','hybrid','hybrid_rerank'):
        rr = [r for r in retrieval if r['mode']==mode and r['split']=='test']
        aa = [a for a in answers if a['mode']==mode]
        semantic = [r['semantic_at_4'] for r in rr if r['semantic_at_4'].get('ndcg') is not None]
        judged = [by_key[(a['question_id'],mode)] for a in aa if (a['question_id'],mode) in by_key]
        negative = [a for a in aa if not a['answerable']]
        unsupported = sum(a['abstention']['unsupported_answer_risk'] for a in aa)
        summaries.append({'mode':mode,'heldout_answers':len(aa),
            'annotated_span_recall_at_4':average([r['metrics']['context']['evidence_recall'] for r in rr]),
            'semantic_ndcg_at_4':average([s['ndcg'] for s in semantic]),
            'semantic_precision_at_4':average([s['precision'] for s in semantic]),
            'semantic_direct_precision_at_4':average([s['direct_precision'] for s in semantic]),
            'semantic_direct_mrr_at_4':average([s['mrr_direct'] for s in semantic]),
            'semantic_judged_questions':len(semantic),
            'constraint_violations':sum(r['constraint_violations'] for r in rr),
            'valid_citation_id_rate':average([a['citation_ids']['validity'] for a in aa if a['status']=='answered']),
            'correct_refusals':sum(a['abstention']['correct_refusal'] for a in aa),
            'false_refusals':sum(a['abstention']['false_refusal'] for a in aa),
            'unanswerable_questions':len(negative),
            'answered_unanswerable_questions':unsupported,
            'unanswerable_answer_rate':unsupported/len(negative) if negative else None,
            'runtime_failures':sum(a['abstention']['runtime_failure'] for a in aa),
            'mean_end_to_end_seconds':average([a['timings_ms']['total']/1000 for a in aa]),
            'reviewed_answers':len(judged),'reviewed_correctness_mean_0_to_2':average([j['correctness'] for j in judged]),
            'reviewed_cited_claim_support':average([j.get('cited_claim_support') for j in judged])})
    atomic_json(output/'metrics.json',{'metrics':summaries,'answer_run':str(answer_folder) if answer_folder else None,
        'limitations':['Annotated-span recall is not exhaustive semantic evidence recall.',
                      'Semantic ranking metrics cover six diagnostic questions, not a random representative sample.',
                      'Source annotations and graded qrels were assistant-checked; independent human review pending.',
                      'Factual correctness and semantic citation support remain null until source_judgments are supplied.']})
    with (output/'metrics.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    print(f'Saved actual metrics to {output}. Reviewed factuality scores are not fabricated when labels are absent.')


if __name__=='__main__':
    main()
