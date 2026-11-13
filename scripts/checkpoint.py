"""Preserve the original application without importing or launching it."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    destination = ROOT / "docs" / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    inventory_path = destination / "inventory.json"
    if inventory_path.exists():
        print("Original checkpoint already exists; leaving it unchanged.")
        return
    records = []
    paths = [ROOT / name for name in ("app.py", "enhanced_app.py", "requirements.txt", "README.md")]
    for directory in (ROOT / "arxiv_papers", ROOT / "tmp" / "local_qdrant"):
        if directory.exists():
            paths.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    for path in paths:
        if not path.is_file():
            continue
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        relative = path.relative_to(ROOT).as_posix()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        if path.parent == ROOT:
            shutil.copy2(path, destination / path.name)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "files": records,
        "notes": "Copied original source; PDF/index files remain in place. Original app was not launched.",
    }
    inventory_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Preserved {len(records)} file hashes and original source in {destination}")


if __name__ == "__main__":
    main()
