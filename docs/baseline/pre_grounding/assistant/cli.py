from __future__ import annotations
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from .artifacts import atomic_json
from .config import Config
from .downloads import download_pdf
from .schemas import Paper, RetrievalFilters
from .service import AssistantService


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Persistent research assistant")
    parser.add_argument("--corpus")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("library")
    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--paper", action="append", default=[])
    ask.add_argument("--year-min", type=int)
    ask.add_argument("--year-max", type=int)
    ask.add_argument("--compare", action="store_true")
    ask.add_argument("--mode", choices=["bm25","dense","hybrid","hybrid_rerank"], default="hybrid_rerank")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("manifest", type=Path)
    ingest.add_argument("--download", action="store_true")
    search = commands.add_parser("search")
    search.add_argument("query", help="arXiv Boolean query")
    search.add_argument("--limit", type=int, default=40)
    search.add_argument("--output", type=Path, default=Path("data/search-selection.json"))
    search.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    config = Config.load()
    if args.corpus:
        config = replace(config, corpus_id=args.corpus)
    service = AssistantService(config)
    try:
        if args.command == "library":
            result = service.library()
        elif args.command == "ask":
            filters = RetrievalFilters(tuple(args.paper), year_min=args.year_min, year_max=args.year_max)
            result = service.ask(args.question, filters, args.mode, args.paper if args.compare else ())
        elif args.command == "ingest":
            papers = [Paper.from_dict(p) for p in json.loads(args.manifest.read_text(encoding="utf-8"))["papers"]]
            if args.download:
                for paper in papers:
                    download_pdf(paper, service.root, config.download_attempts)
            result = service.ingest(papers)
        else:
            from .discovery import Discovery
            discovery = Discovery(service.root, config, service.encoder, service.reranker)
            result = discovery.search(args.query, RetrievalFilters(), args.limit, args.refresh)
            atomic_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        service.close()


if __name__ == "__main__":
    main()
