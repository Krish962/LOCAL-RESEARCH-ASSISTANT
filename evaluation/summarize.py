"""Recalculate the reported experiments from saved measurements and answers."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from assistant.artifacts import read_jsonl
from assistant.config import PROJECT_ROOT, fingerprint
from assistant.filters import eligible_papers
from assistant.grounding import quotation_span
from assistant.schemas import Paper, RetrievalFilters
from .metrics import citation_ids


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def answer_metrics(run):
    questions = {q['question_id']: q for q in read_jsonl(PROJECT_ROOT/'evaluation/final_validation_40.jsonl')}
    papers = [Paper.from_dict(p) for p in json.loads(
        (PROJECT_ROOT/'data/manifests/course_rag_v1.json').read_text(encoding='utf-8'))['papers']]
    path = run/'assistant_source_judgments.jsonl'
    judgments = {(j['question_id'], j['arm']): j for j in read_jsonl(path)} if path.exists() else {}
    result = []
    for arm in ['baseline', 'grounded']:
        rows = read_jsonl(run/(arm+'_answers.jsonl'))
        positives = [r for r in rows if r['answerable']]
        answered = [r for r in rows if r['status'] == 'answered']
        negative = [r for r in rows if r['type'] == 'missing_evidence']
        singles = [r for r in positives if r['type'] == 'single']
        comparisons = [r for r in positives if r['type'] == 'comparison']
        reviewed = [judgments[(r['question_id'], arm)] for r in answered
                    if (r['question_id'], arm) in judgments]
        violations = 0
        quotes, matched = 0, 0
        for row in rows:
            eligible = {p.paper_id for p in eligible_papers(
                papers, RetrievalFilters.from_dict(questions[row['question_id']]['filters']))[0]}
            violations += sum(s['paper_id'] not in eligible for s in row['sources'])
            sources = {s['label']: s for s in row['sources']}
            if row.get('answer_mode') == 'extractive' and row['status'] == 'answered':
                for claim in row.get('claims', []):
                    for evidence in claim.get('evidence', []):
                        quotes += 1
                        source = sources.get(evidence['source_id'])
                        matched += int(source is not None and quotation_span(source['text'], evidence['quote']) is not None)
        result.append({
            'arm': arm, 'completed_cases': len(rows), 'answerable_cases': len(positives),
            'answered_answerable': sum(r['status'] == 'answered' for r in positives),
            'answerable_answer_rate': mean([int(r['status'] == 'answered') for r in positives]),
            'single_paper_answered': sum(r['status'] == 'answered' for r in singles), 'single_paper_cases': len(singles),
            'comparisons_answered': sum(r['status'] == 'answered' for r in comparisons), 'comparison_cases': len(comparisons),
            'missing_evidence_cases': len(negative),
            'missing_evidence_refusals': sum(r['status'] in {'insufficient_evidence', 'no_match'} for r in negative),
            'correct_empty_scope_refusals': sum(r['status'] == 'no_match' for r in rows if r['type'] == 'no_match'),
            'false_refusals': sum(r['status'] in {'insufficient_evidence', 'no_match'} for r in positives),
            'runtime_failures': sum(r['status'] in {'error', 'invalid_output'} for r in rows),
            'mean_seconds_all': mean([r['timings_ms']['total']/1000 for r in rows]),
            'metadata_constraint_violations': violations,
            'literal_citation_id_validity': mean([citation_ids(r['answer'], r['sources'])['validity'] for r in answered]),
            'selected_quotation_spans': quotes, 'quotation_spans_matching_saved_sources': matched,
            'reviewed_answered_cases': len(reviewed),
            'review_complete': sum(j['correctness'] == 2 for j in reviewed) if reviewed else None,
            'review_partial': sum(j['correctness'] == 1 for j in reviewed) if reviewed else None,
            'review_wrong_or_unresponsive': sum(j['correctness'] == 0 for j in reviewed) if reviewed else None})
    return result


def report_metrics(run):
    rows = read_jsonl(PROJECT_ROOT/'results/retrieval-final/measurements.jsonl')
    retrieval = []
    for mode in ['bm25', 'dense', 'hybrid', 'hybrid_rerank']:
        selected = [r for r in rows if r['mode'] == mode and r['split'] == 'test']
        semantic = [r['semantic_at_4']['ndcg'] for r in selected if r['semantic_at_4']['ndcg'] is not None]
        retrieval.append({'mode': mode, 'ndcg_at_4': mean(semantic), 'graded_questions': len(semantic),
                          'annotated_span_recall_at_4': mean([r['metrics']['context']['evidence_recall'] for r in selected])})
    filters = read_jsonl(PROJECT_ROOT/'results/filter-ablation/measurements.jsonl')
    filtering = [{'filters_enabled': flag, 'records': len([r for r in filters if r['filters_enabled'] == flag]),
                  'returned_passages': sum(r['returned'] for r in filters if r['filters_enabled'] == flag),
                  'out_of_scope_passages': sum(r['constraint_violations'] for r in filters if r['filters_enabled'] == flag)}
                 for flag in [False, True]]
    folder = PROJECT_ROOT/'results/ragas-857109279246'
    ragas = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    scores = read_jsonl(folder/'scores.jsonl')
    inputs = read_jsonl(folder/'inputs.jsonl')
    assert fingerprint(inputs) == ragas['identity']['answers_hash'], 'Ragas input records changed'
    assert sum(r['status'] == 'completed' for r in scores) == ragas['completed_metric_calls']
    assert sum(r['status'] == 'failed' for r in scores) == ragas['failed_metric_calls']
    return {'retrieval': retrieval, 'filtering': filtering, 'answers': answer_metrics(run),
            'ragas': {k: ragas[k] for k in ['selected_cases', 'planned_metric_calls', 'attempted_metric_calls',
                                            'completed_metric_calls', 'failed_metric_calls', 'time_budget_seconds']},
            'extraction': json.loads((PROJECT_ROOT/'results/extraction-pilot/review.json').read_text(encoding='utf-8')),
            'interpretation': 'Coverage and literal quotation matching are separate from correctness; review labels are not independent human validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT_ROOT/'results/grounding-validation-abad545bfc7a')
    args = parser.parse_args()
    print(json.dumps(report_metrics(args.run), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
