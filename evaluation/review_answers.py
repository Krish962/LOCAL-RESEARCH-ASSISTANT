"""Terminal source review. This creates actual reviewer labels, never inferred gold."""
from __future__ import annotations

import argparse
from pathlib import Path
from assistant.artifacts import read_jsonl,write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--reviewer',required=True)
    parser.add_argument('--mode',help='Optionally review one baseline at a time')
    parser.add_argument('--questions',type=Path,default=Path(__file__).with_name('questions.jsonl'))
    args = parser.parse_args()
    questions = {q['question_id']:q['question'] for q in read_jsonl(args.questions)}
    output = args.answers.parent/'source_judgments.jsonl'
    judgments = read_jsonl(output) if output.exists() else []
    existing = {(r['question_id'],r['mode']) for r in judgments}
    for row in read_jsonl(args.answers):
        mode=row.get('mode',row.get('arm'))
        if args.mode and mode != args.mode:
            continue
        key = row['question_id'],mode
        if key in existing:
            continue
        print(f"\n{key}\nQUESTION: {row.get('question',questions.get(row['question_id'], row['question_id']))}\nREFERENCE: {row['reference']}\nANSWER: {row['answer']}\nSTATUS: {row['status']}")
        for source in row['sources']:
            print(f"\n[{source['label']}] {source['title']} PDF page {source['page_number']}\n{source['text']}")
        rating = input('Correctness 0=wrong, 1=partial, 2=correct; s=skip; q=quit: ').strip()
        if rating=='q':
            break
        if rating=='s':
            continue
        if rating not in {'0','1','2'}:
            print('Not saved: invalid rating.')
            continue
        support = input('Fraction of factual claims supported by their cited passages, 0..1; blank=unjudged: ').strip()
        if support:
            try:
                support_value = float(support)
                if not 0 <= support_value <= 1:
                    raise ValueError()
            except ValueError:
                print('Not saved: support must be in [0,1].')
                continue
        else:
            support_value = None
        judgments.append({'question_id':row['question_id'],'mode':mode,'trace_id':row['trace_id'],
            'reviewer':args.reviewer,'correctness':int(rating),'cited_claim_support':support_value,
            'notes':input('Evidence/limitations note: ').strip()})
        write_jsonl(output,judgments)
    print(f'Saved {len(judgments)} reviewer judgments to {output}')


if __name__=='__main__':
    main()
