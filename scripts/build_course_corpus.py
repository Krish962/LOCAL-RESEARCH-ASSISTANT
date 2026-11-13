"""Verify a course-related candidate list, then acquire the actual PDFs.

Curator-seeded IDs are not represented as held-out discovery-ranking results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from assistant.arxiv_api import fetch
from assistant.artifacts import atomic_json
from assistant.config import PROJECT_ROOT
from assistant.downloads import download_pdf

SEEDS = {
    "2005.11401":"Retrieval-Augmented Generation",
    "2004.04906":"Dense Passage Retrieval",
    "2007.01282":"Leveraging Passage Retrieval",
    "2207.06300":"Re2G",
    "2002.08909":"REALM",
    "2102.10073":"Pyserini",
    "2212.10496":"Precise Zero-Shot Dense Retrieval",
    "2004.12832":"ColBERT",
    "2112.01488":"ColBERTv2",
    "2107.05720":"SPLADE",
    "2104.08663":"BEIR",
    "2210.07316":"MTEB",
    "2307.03172":"Lost in the Middle",
    "2310.11511":"Self-RAG",
    "2401.15884":"Corrective Retrieval",
    "2309.15217":"RAGAS",
    "2311.09476":"ARES",
    "2401.00396":"RAGTruth",
    "2312.10997":"Retrieval-Augmented Generation",
    "2406.15319":"LongRAG",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",action="store_true")
    args = parser.parse_args()
    papers = fetch(ids=list(SEEDS),limit=len(SEEDS))
    if {p.paper_id for p in papers} != set(SEEDS):
        raise RuntimeError("The API did not verify every requested paper ID")
    for paper in papers:
        if SEEDS[paper.paper_id].casefold() not in paper.title.casefold():
            raise ValueError(f"Candidate ID/title mismatch: {paper.paper_id}: {paper.title}")
        print(f"Verified {paper.arxiv_id}: {paper.title}",flush=True)
    record = {"status":"metadata_verified","selection":"curator-seeded course concepts; not a discovery benchmark",
              "papers":[p.to_dict() for p in papers],"download_failures":[]}
    path = PROJECT_ROOT / "data/corpora/course_rag_v1.source.json"
    atomic_json(path,record)
    if args.download:
        for paper in papers:
            try:
                download_pdf(paper,PROJECT_ROOT)
                print(f"Downloaded {paper.arxiv_id}",flush=True)
            except Exception as exc:
                record["download_failures"].append({"paper_id":paper.paper_id,"error":str(exc)})
                print(f"Failed {paper.arxiv_id}: {exc}",flush=True)
            record["papers"] = [p.to_dict() for p in papers]
            atomic_json(path,record)
        record["status"] = "downloaded" if not record["download_failures"] else "partial_download"
        atomic_json(path,record)


if __name__ == "__main__":
    main()
