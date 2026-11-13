from __future__ import annotations
import hashlib
import json
from assistant.artifacts import atomic_json
from assistant.config import PROJECT_ROOT


def main():
    original = json.loads((PROJECT_ROOT/'docs/baseline/inventory.json').read_text(encoding='utf-8'))
    checked = []
    for item in original['files']:
        if item['path'] in {'app.py','enhanced_app.py','README.md','requirements.txt'}:
            path = PROJECT_ROOT/'docs/baseline'/item['path']
        else:
            path = PROJECT_ROOT/item['path']
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle,'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError(f"Preserved artifact changed: {item['path']}")
        checked.append(item['path'])
    atomic_json(PROJECT_ROOT/'results/preservation-check.json',{'status':'passed','files_checked':checked})
    print(f'All {len(checked)} original/checkpoint artifact hashes match.')


if __name__=='__main__':
    main()
