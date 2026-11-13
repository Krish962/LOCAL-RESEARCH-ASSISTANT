"""Fetch pinned public NLI weights; verify the model's LFS SHA256 and byte size."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from assistant.config import Config, PROJECT_ROOT
from assistant.grounding import model_cache
from assistant.artifacts import atomic_json

FILES = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'merges.txt']


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl.exe','-fSL','--retry','2','--retry-all-errors','--connect-timeout','15',
                    '--max-time','600',url,'-o',str(destination)],check=True)


def main():
    config = Config.load()
    path = model_cache(config)
    verified = path/'verified.json'
    if verified.exists():
        saved = json.loads(verified.read_text(encoding='utf-8'))
        if (saved.get('model')==config.grounding_model and saved.get('revision')==config.grounding_revision
            and {r['file'] for r in saved.get('files',[])}==set(FILES)
            and all((path/r['file']).is_file() and hashlib.sha256((path/r['file']).read_bytes()).hexdigest()==r['sha256']
                    for r in saved['files'])):
            print('All pinned grounding model files verified from the local cache.',flush=True)
            return
    metadata = path/'hub-metadata.json'
    fetch(f'https://huggingface.co/api/models/{config.grounding_model}/revision/{config.grounding_revision}?blobs=true',metadata)
    info = json.loads(metadata.read_text(encoding='utf-8'))
    if info['sha'] != config.grounding_revision:
        raise ValueError('Requested model revision does not match Hub metadata')
    entries = {s['rfilename']:s for s in info['siblings']}
    records = []
    for name in FILES:
        target = path/name
        expected = entries[name].get('lfs',{}).get('sha256')
        valid = target.exists() and target.stat().st_size == entries[name]['size']
        if valid and expected:
            valid = hashlib.sha256(target.read_bytes()).hexdigest() == expected
        if not valid:
            partial = path/(name+'.partial')
            fetch(f'https://huggingface.co/{config.grounding_model}/resolve/{config.grounding_revision}/{name}',partial)
            if partial.stat().st_size != entries[name]['size']:
                raise ValueError(f'Wrong download size for {name}')
            if expected and hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Model checksum mismatch for {name}')
            partial.replace(target)
        records.append({'file': name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'bytes': target.stat().st_size, 'hub_lfs_sha256': expected})
        print(f'Verified {name}',flush=True)
    atomic_json(path/'verified.json',{'model':config.grounding_model,'revision':config.grounding_revision,
        'license':info.get('cardData',{}).get('license'),'files':records})
    print(f'Grounding model prepared at {path}',flush=True)


if __name__=='__main__':
    main()
