from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .artifacts import atomic_json


class Trace:
    def __init__(self, root: Path, kind="query"):
        self.root = root
        self.started = time.perf_counter()
        self.data = {"run_id": f"{kind}-{uuid.uuid4().hex}",
                     "created_at": datetime.now(timezone.utc).isoformat(), "timings_ms": {}}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.data["timings_ms"][name] = (time.perf_counter() - start) * 1000

    def save(self):
        self.data["timings_ms"]["total"] = (time.perf_counter() - self.started) * 1000
        path = self.root / "results" / self.data["run_id"] / "trace.json"
        atomic_json(path, self.data)
        return str(path)
