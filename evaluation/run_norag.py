"""Metadata-only, no-passage generation baseline; no source citations apply."""
from __future__ import annotations
import json
from pathlib import Path
from assistant.artifacts import atomic_json,read_jsonl,write_jsonl
from assistant.config import Config,PROJECT_ROOT,fingerprint
from assistant.filters import eligible_papers
from assistant.schemas import Paper,RetrievalFilters
from .metrics import abstention_counts


def main():
    import ollama
    import time
    cfg = Config.load()
    manifest = json.loads((PROJECT_ROOT/'data/manifests'/f'{cfg.corpus_id}.json').read_text(encoding='utf-8'))
    papers = [Paper.from_dict(p) for p in manifest['papers']]
    cases = [q for q in read_jsonl(PROJECT_ROOT/'evaluation/questions.jsonl') if q['split']=='test']
    folder = PROJECT_ROOT/'results'/('no-passages-'+fingerprint({'questions':cases,'config':cfg.identity()})[:12])
    path = folder/'answers.jsonl'
    rows = read_jsonl(path) if path.exists() else []
    done = {r['question_id'] for r in rows}
    client = ollama.Client(host=cfg.ollama_url,timeout=300.)
    schema = {'type':'object','properties':{'status':{'type':'string','enum':['answered','insufficient_evidence']},
              'answer':{'type':'string'}},'required':['status','answer'],'additionalProperties':False}
    for case in cases:
        if case['question_id'] in done:
            continue
        begin = time.perf_counter()
        selected,_ = eligible_papers(papers,RetrievalFilters.from_dict(case['filters']))
        raw = None
        if not selected:
            parsed = {'status':'no_match','answer':'No indexed papers satisfy the selected scope.'}
        else:
            prompt = 'Question: '+case['question']+'\nSelected paper titles (no passages supplied):\n'+'\n'.join(p.title for p in selected)
            try:
                raw = client.chat(model=cfg.generator_model,stream=False,format=schema,messages=[
                    {'role':'system','content':'Answer from your existing knowledge. No paper passages are provided. If you do not know, say insufficient_evidence. Return JSON with status and a concise answer.'},
                    {'role':'user','content':prompt}],options={'num_ctx':cfg.context_window,'num_predict':cfg.output_tokens,
                        'temperature':cfg.temperature,'seed':cfg.seed,'num_thread':cfg.cpu_threads}).model_dump(mode='json')
                parsed = json.loads(raw['message']['content'])
                if parsed.get('status') not in {'answered','insufficient_evidence'} or not isinstance(parsed.get('answer'),str):
                    raise ValueError('Invalid no-passage response')
            except Exception as exc:
                parsed = {'status':'error','answer':str(exc)}
        rows.append({'question_id':case['question_id'],'mode':'no_passages','reference':case['reference'],
            'answerable':case['answerable'],**parsed,'seconds':time.perf_counter()-begin,'raw':raw,
            'abstention':abstention_counts(case['answerable'],parsed['status']),
            'citation_metrics':'not applicable: no source passages supplied'})
        write_jsonl(path,rows)
        print(f"No passages {len(rows)}/{len(cases)}: {case['question_id']} {parsed['status']}",flush=True)
    atomic_json(folder/'manifest.json',{'status':'complete','cases':len(rows),'snapshot_id':manifest['snapshot_id'],
        'definition':'Same scope metadata and generator, without passage retrieval. Citation/faithfulness scores do not apply.'})


if __name__=='__main__':
    main()
