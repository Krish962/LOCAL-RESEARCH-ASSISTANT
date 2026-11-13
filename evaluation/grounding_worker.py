"""One sequential validation arm, optionally using the preserved v1 package."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['baseline','grounded'],required=True)
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    os.environ.setdefault('HF_HOME',str(root/'.cache/huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    checkpoint=root/'evaluation/baselines/generation'
    if args.arm=='baseline':sys.path.insert(0,str(checkpoint))
    from assistant.config import Config
    from assistant.service import AssistantService
    from assistant.schemas import RetrievalFilters
    from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
    config=Config.load(checkpoint/'configs/default.json') if args.arm=='baseline' else Config.load()
    service=AssistantService(config,root)
    rows=read_jsonl(args.output) if args.output.exists() else []
    done={r['question_id'] for r in rows}
    try:
        for case in read_jsonl(args.questions):
            if case['question_id'] in done:continue
            result=service.ask(case['question'],RetrievalFilters.from_dict(case['filters']),
                'hybrid_rerank',case['required_papers'],case['question_id'])
            rows.append({'question_id':case['question_id'],'question':case['question'],'arm':args.arm,
                'answerable':case['answerable'],'type':case['type'],'reference':case['reference'],**result})
            write_jsonl(args.output,rows)
            print(f"{args.arm} {len(rows)}: {case['question_id']} {result['status']}",flush=True)
    finally:service.close()


if __name__=='__main__':main()
