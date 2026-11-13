"""Bounded official arXiv Atom access, with serialized request pacing."""
from __future__ import annotations

import subprocess
import threading
import time
from datetime import datetime
from urllib.parse import urlencode

from .schemas import Paper

_lock = threading.Lock()
_last_request = 0.


def fetch(query="", ids=(), limit=40, start=0) -> list[Paper]:
    import feedparser
    import httpx
    global _last_request
    values = {"search_query":query,"id_list":",".join(ids),"max_results":limit,
              "start":start,"sortBy":"relevance","sortOrder":"descending"}
    url = "https://export.arxiv.org/api/query?" + urlencode(values)
    with _lock:
        time.sleep(max(0.,3.-(time.monotonic()-_last_request)))
        _last_request = time.monotonic()
        try:
            result = httpx.get(url,timeout=httpx.Timeout(25.,connect=10.),follow_redirects=True)
            result.raise_for_status()
            content = result.content
        except (httpx.TransportError,httpx.HTTPStatusError):
            content = subprocess.run(["curl.exe","-fsSL","--connect-timeout","10","--max-time","40",
                "--max-filesize","5242880",url],check=True,capture_output=True).stdout
    feed = feedparser.parse(content)
    papers = []
    for entry in feed.entries:
        if "/abs/" not in entry.id:
            raise ValueError(f"arXiv returned an API error: {entry.get('summary','unknown')}")
        identifier = entry.id.rsplit("/abs/",1)[-1]
        import re
        canonical = re.sub(r"v\d+$","",identifier)
        pdf = next((link.href for link in entry.get("links",[]) if link.get("type")=="application/pdf"),None)
        if not pdf:
            raise ValueError(f"No PDF URL supplied for {identifier}")
        published = entry.published
        primary = entry.get("arxiv_primary_category",{}).get("term")
        papers.append(Paper(canonical,identifier," ".join(entry.title.split()),
            [author.name for author in entry.get("authors",[])],entry.summary,
            [tag.term for tag in entry.get("tags",[])],primary,published,entry.updated,
            datetime.fromisoformat(published.replace("Z","+00:00")).year,
            entry.id.replace("http://","https://"),pdf.replace("http://","https://"),
            entry.get("arxiv_doi"),entry.get("arxiv_journal_ref")))
    if not papers and feed.get("bozo"):
        raise ValueError("arXiv did not return a valid Atom feed")
    return papers
