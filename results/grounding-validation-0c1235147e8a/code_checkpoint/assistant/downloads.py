from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from .schemas import Paper


def download_pdf(paper: Paper, root: Path, attempts=3) -> Path:
    import httpx
    from pypdf import PdfReader
    if not re.fullmatch(r"[A-Za-z0-9./_-]+", paper.arxiv_id) or ".." in paper.arxiv_id:
        raise ValueError("Unsafe arXiv identifier")
    parsed = urlparse(paper.pdf_url)
    if parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}:
        raise ValueError("Downloads must use a verified arXiv HTTPS URL")
    folder = root / "arxiv_papers"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (paper.arxiv_id.replace("/", "_") + ".pdf")
    temporary = path.with_suffix(".pdf.partial")
    if not path.exists():
        errors = []
        for attempt in range(attempts):
            try:
                try:
                    with httpx.stream("GET", paper.pdf_url, follow_redirects=True,
                                      timeout=httpx.Timeout(60., connect=15.)) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for block in response.iter_bytes():
                                handle.write(block)
                except httpx.TransportError:
                    # Windows TLS fallback without disabling certificate validation.
                    if os.name != "nt":
                        raise
                    subprocess.run(["curl.exe", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
                                    "-o", str(temporary), paper.pdf_url], check=True, capture_output=True)
                if not temporary.read_bytes().startswith(b"%PDF-") or len(PdfReader(temporary).pages) < 1:
                    raise ValueError("Response is not a readable nonempty PDF")
                os.replace(temporary, path)
                break
            except Exception as exc:
                errors.append(str(exc))
                if temporary.exists():
                    temporary.unlink()
                if attempt + 1 < attempts:
                    time.sleep(min(2 ** attempt, 4))
        else:
            raise RuntimeError(f"PDF download failed after {attempts} attempts: {errors}")
    with path.open("rb") as handle:
        paper.pdf_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    paper.pdf_path = str(path.resolve())
    return path
