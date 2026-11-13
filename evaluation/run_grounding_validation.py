"""Resumable comparison of the generation baseline and source-based application."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from assistant.config import PROJECT_ROOT, Config, fingerprint
from assistant.artifacts import atomic_json, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['both', 'baseline', 'grounded'], default='both')
    parser.add_argument('--questions', type=Path, default=PROJECT_ROOT/'evaluation/final_validation_40.jsonl')
    parser.add_argument('--run-file', type=Path)
    args = parser.parse_args()
    questions = args.questions.resolve()
    cases = read_jsonl(questions)
    frozen = json.loads(questions.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if frozen['questions_hash'] != fingerprint(cases):
        raise ValueError('Frozen evaluation questions changed')
    checkpoint = PROJECT_ROOT/'evaluation/baselines/generation'
    inventory = json.loads((checkpoint/'inventory.json').read_text(encoding='utf-8'))
    for record in inventory['files']:
        if hashlib.sha256((checkpoint/record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Generation-baseline source changed')
    import ollama
    cfg = Config.load()
    models = ollama.Client(host=cfg.ollama_url).list().model_dump(mode='json')['models']
    digest = next(m['digest'] for m in models if m['model'] == cfg.generator_model)
    identity = {
        'questions_hash': fingerprint(cases), 'config_hash': cfg.identity(),
        'code_hash': fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (PROJECT_ROOT/'assistant').glob('*.py')}),
        'baseline_checkpoint_hash': fingerprint(inventory), 'generator_digest': digest,
        'worker_hash': hashlib.sha256(Path(__file__).with_name('grounding_worker.py').read_bytes()).hexdigest()}
    output = PROJECT_ROOT/'results'/('grounding-validation-'+fingerprint(identity)[:12])
    output.mkdir(parents=True, exist_ok=True)
    if args.run_file:
        atomic_json(args.run_file, {'run': str(output), 'questions': str(questions), 'identity': identity})
    manifest = {'identity': identity, 'cases': len(cases), 'status': 'running',
                'baseline_checkpoint': 'evaluation/baselines/generation', 'mode': 'hybrid_rerank',
                'questions_path': str(questions), 'evaluation_protocol': frozen['evaluation_protocol'],
                'annotation_provenance': frozen.get('limitations', [])}
    atomic_json(output/'manifest.json', manifest)
    for arm in (['baseline', 'grounded'] if args.arm == 'both' else [args.arm]):
        subprocess.run([sys.executable, '-X', 'utf8', '-u', '-m', 'evaluation.grounding_worker',
                        '--arm', arm, '--questions', str(questions),
                        '--output', str(output/(arm+'_answers.jsonl'))], cwd=PROJECT_ROOT, check=True)
    complete = all((output/(arm+'_answers.jsonl')).is_file()
                   and len(read_jsonl(output/(arm+'_answers.jsonl'))) == len(cases)
                   for arm in ['baseline', 'grounded'])
    manifest['status'] = 'complete' if complete else 'partial'
    atomic_json(output/'manifest.json', manifest)
    print(f'Evaluation records: {output}', flush=True)


if __name__ == '__main__':
    main()
