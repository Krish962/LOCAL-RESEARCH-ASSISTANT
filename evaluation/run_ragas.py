"""Offline, local-only supplementary RAGAS evaluation of saved answers.

Uses a separate environment; never calls a paid endpoint or re-generates answers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from importlib.metadata import version

os.environ.setdefault('RAGAS_DO_NOT_TRACK','true')
os.environ.setdefault('HF_HUB_OFFLINE','1')
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import Config,PROJECT_ROOT,configure_cache,fingerprint
configure_cache()


async def evaluate(args):
    import ollama
    from .local_judge import OllamaStructuredJudge, ADAPTER_VERSION
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics.collections import Faithfulness,ContextRecall,ContextPrecision,AnswerRelevancy
    config = Config.load()
    from urllib.parse import urlparse
    if urlparse(config.ollama_url).hostname not in {'localhost','127.0.0.1','::1'}:
        raise ValueError('This evaluator is local-only; use a loopback Ollama endpoint')
    configure_cache()
    rows = read_jsonl(args.answers)
    questions = {q['question_id']:q for q in read_jsonl(args.questions)}
    selected = [r for r in rows if r.get('mode','hybrid_rerank')==args.mode and r['answerable'] and r['status']=='answered'][:args.limit]
    if not selected:
        raise ValueError('No answerable, answered cases match this selection')
    tags = ollama.Client(host=config.ollama_url).list().model_dump(mode='json')
    judge_digest = next((m['digest'] for m in tags['models'] if m['model']==args.judge),None)
    if not judge_digest:
        raise ValueError(f'Local judge is not installed: {args.judge}')
    llm = OllamaStructuredJudge(args.judge, config.ollama_url,
        PROJECT_ROOT/'results/ragas-native-calls', config.cpu_threads)
    available = {'faithfulness':Faithfulness(llm=llm),'context_recall':ContextRecall(llm=llm),
                 'context_precision':ContextPrecision(llm=llm)}
    if 'answer_relevancy' in args.metrics:
        embeddings = HuggingFaceEmbeddings(model=config.embedding_model,use_api=False,device='cpu',batch_size=16)
        available['answer_relevancy'] = AnswerRelevancy(llm=llm,embeddings=embeddings)
    identity = {'judge':args.judge,'digest':judge_digest,'ragas':version('ragas'),'metric_names':args.metrics,
                'mode':args.mode,'answers_hash':fingerprint(selected),'adapter':ADAPTER_VERSION,
                'judge_context':llm.context_window,'judge_output_tokens':llm.output_tokens}
    output = args.output or PROJECT_ROOT/'results'/('ragas-'+fingerprint(identity)[:12])
    cache = PROJECT_ROOT/'results/ragas-cache'
    completed = []
    start = time.monotonic()
    try:
        for row in selected:
            case = questions[row['question_id']]
            for name in args.metrics:
                elapsed = time.monotonic()-start
                if elapsed >= args.max_seconds:
                    break
                contexts = [s['text'] for s in row['sources']]
                payload = {'user_input':case['question'],'response':row['answer'],
                           'reference':case['reference'],'retrieved_contexts':contexts}
                arguments = {'faithfulness':{k:payload[k] for k in ['user_input','response','retrieved_contexts']},
                    'context_recall':{k:payload[k] for k in ['user_input','retrieved_contexts','reference']},
                    'context_precision':{k:payload[k] for k in ['user_input','reference','retrieved_contexts']},
                    'answer_relevancy':{k:payload[k] for k in ['user_input','response']}}[name]
                key = fingerprint({'judge':judge_digest,'ragas':version('ragas'),'metric':name,'payload':arguments,
                                   'adapter':ADAPTER_VERSION,'context':llm.context_window,'output':llm.output_tokens})
                cached = cache/(key+'.json')
                cached_result = json.loads(cached.read_text(encoding='utf-8')) if cached.exists() else None
                if cached_result and cached_result.get('status')=='completed':
                    result = cached_result
                    result['cache_hit'] = True
                else:
                    begin = time.monotonic()
                    result = {'question_id':row['question_id'],'metric':name,'status':'failed','cache_hit':False}
                    try:
                        score = await asyncio.wait_for(available[name].ascore(**arguments),
                            timeout=min(180,args.max_seconds-elapsed))
                        value = float(score.value)
                        if not math.isfinite(value):
                            raise ValueError('Judge returned an undefined score')
                        result.update({'status':'completed','value':value,'reason':score.reason})
                    except Exception as exc:
                        result.update({'value':None,'error_type':type(exc).__name__,'error':str(exc)})
                    result['seconds'] = time.monotonic()-begin
                    atomic_json(cached,result)
                completed.append(result)
                write_jsonl(output/'scores.jsonl',completed)
                print(f"{row['question_id']} {name}: {result['status']} {result.get('value')}",flush=True)
            if time.monotonic()-start >= args.max_seconds:
                break
        successful = sum(r['status']=='completed' for r in completed)
        atomic_json(output/'manifest.json',{'identity':identity,'selected_cases':len(selected),
            'planned_metric_calls':len(selected)*len(args.metrics),'attempted_metric_calls':len(completed),
            'completed_metric_calls':successful,'failed_metric_calls':len(completed)-successful,
            'wall_seconds':time.monotonic()-start,'time_budget_seconds':args.max_seconds,
            'status':'complete' if len(completed)==len(selected)*len(args.metrics) else 'time_budget_reached',
            'interpretation':'Supplementary local-judge scores; not independent human validation or exact cited-passage support. Refusals excluded and scored separately.'})
    finally:
        await llm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--questions',type=Path,default=PROJECT_ROOT/'evaluation/questions.jsonl')
    parser.add_argument('--mode',default='hybrid_rerank')
    parser.add_argument('--judge',default='llama3.2:3b')
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--max-seconds',type=int,default=1800)
    parser.add_argument('--metrics',nargs='+',choices=['faithfulness','context_recall','context_precision','answer_relevancy'],
                        default=['faithfulness','context_recall','context_precision','answer_relevancy'])
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.limit < 1 or args.max_seconds < 1:
        parser.error('Limit and time budget must be positive')
    asyncio.run(evaluate(args))


if __name__=='__main__':
    main()
