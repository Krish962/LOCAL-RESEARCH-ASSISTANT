from __future__ import annotations
import atexit
import json
import os
import re
import threading
from .schemas import Paper, RetrievalFilters
from .service import AssistantService

_service = None
_lock = threading.Lock()


def optional_year(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    if not re.fullmatch(r"\d{4}",text):
        raise ValueError("Use a four-digit year, or leave the field empty")
    return int(text)


def get_service():
    global _service
    with _lock:
        if _service is None:
            _service = AssistantService()
            atexit.register(_service.close)
    return _service


def build_ui():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    def library():
        data = get_service().library()
        papers = data.get("papers", [])
        choices = [(f"{p['paper_id']} — {p['title']}",p["paper_id"]) for p in papers]
        categories = sorted({c for p in papers for c in p["categories"]})
        authors = sorted({a for p in papers for a in p["authors"]})
        summary = f"{data['corpus_id']}: {len(papers)} ready papers, {data.get('chunks',0)} chunks ({data['status']})"
        return summary, gr.update(choices=choices,value=[]), gr.update(choices=categories,value=[]), gr.update(choices=authors,value=[])

    def search(query, ranking_text, count, categories, year_min, year_max, refresh):
        from .discovery import Discovery
        service = get_service()
        with service.lock:
            data = Discovery(service.root,service.config,service.encoder,service.reranker).search(
                query, RetrievalFilters(categories=tuple(categories or []),year_min=optional_year(year_min),year_max=optional_year(year_max)),
                int(count),refresh,ranking_text or None)
        rows = [[p["paper_id"],p["title"],p["first_submitted_year"],", ".join(p["categories"]),p["arxiv_url"]] for p in data["papers"]]
        return data, rows, gr.update(choices=[(p["title"],p["paper_id"]) for p in data["papers"]],value=[])

    def ingest(data, selected):
        from .downloads import download_pdf
        if not data or not selected:
            raise gr.Error("Select at least one search result")
        service = get_service()
        available = {p["paper_id"]:p for p in data["papers"]}
        papers, failures = [], []
        with service.lock:
            for key in selected:
                if key not in available:
                    raise gr.Error("Paper is not in these search results")
                paper = Paper.from_dict(available[key])
                try:
                    download_pdf(paper,service.root,service.config.download_attempts)
                    papers.append(paper)
                except Exception as exc:
                    failures.append({"paper_id":key,"error":str(exc)})
            result = service.ingest(papers) if papers else {"status":"download_failed","added":0}
        result["download_failures"] = failures
        return json.dumps(result,ensure_ascii=False,indent=2), *library()

    def ask(question, ids, cats, authors, year_min, year_max, compare):
        try:
            filters = RetrievalFilters(tuple(ids or []),tuple(cats or []),tuple(authors or []),
                optional_year(year_min),optional_year(year_max))
            result = get_service().ask(question,filters,required_papers=tuple(ids or []) if compare else ())
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return result["answer"],result["sources"],{k:result[k] for k in ("status","filters","timings_ms","trace_id","evidence_checks") if k in result}

    with gr.Blocks(title="AI Research Assistant",analytics_enabled=False) as demo:
        gr.Markdown("# AI Research Assistant\nIndex papers once. Ask questions with exact paper/page evidence.")
        state = gr.State()
        status = gr.Textbox(label="Library status",interactive=False)
        refresh_library = gr.Button("Refresh library")
        with gr.Tab("Find and index"):
            query = gr.Textbox(label="arXiv query (title, abstract, category/date syntax supported)",
                value='(ti:"retrieval augmented" OR abs:"retrieval augmented") AND cat:cs.CL')
            ranking_text = gr.Textbox(label="Research question for ranking titles and abstracts",value="Research on improving retrieval-augmented generation")
            count = gr.Slider(5,100,value=40,step=1,label="Metadata candidates")
            discovery_categories = gr.Dropdown(["cs.CL","cs.IR","cs.AI","cs.LG","stat.ML"],multiselect=True,label="Discovery category restriction")
            with gr.Row():
                discovery_min = gr.Textbox(value="",placeholder="e.g. 2023",label="First submission year from (blank = any)")
                discovery_max = gr.Textbox(value="",placeholder="e.g. 2026",label="Through year (blank = any)")
            refresh = gr.Checkbox(label="Refresh cached metadata",value=False)
            search_button = gr.Button("Search Papers",variant="primary")
            results = gr.Dataframe(headers=["ID","Title","First submission year","Categories","Source"],interactive=False)
            selected = gr.Dropdown(choices=[],multiselect=True,label="Select papers to index")
            ingest_button = gr.Button("Index Selected Papers")
            ingest_status = gr.Textbox(label="Indexing status",interactive=False,lines=4)
        with gr.Tab("Ask the library"):
            ids = gr.Dropdown(choices=[],multiselect=True,label="Indexed papers (empty = all)")
            cats = gr.Dropdown(choices=[],multiselect=True,label="Categories")
            authors = gr.Dropdown(choices=[],multiselect=True,label="Authors (names may be ambiguous)")
            with gr.Row():
                year_min = gr.Textbox(value="",placeholder="e.g. 2023",label="First submission year from")
                year_max = gr.Textbox(value="",placeholder="e.g. 2026",label="Through year")
            compare = gr.Checkbox(label="Require evidence from each selected paper",value=False)
            question = gr.Textbox(label="Question",lines=3)
            ask_button = gr.Button("Ask Question",variant="primary")
            answer = gr.Textbox(label="Answer",lines=8,interactive=False)
            evidence = gr.JSON(label="Exact cited passages")
            details = gr.JSON(label="Status, active filters and timings")
        refresh_library.click(library,outputs=[status,ids,cats,authors])
        search_button.click(search,[query,ranking_text,count,discovery_categories,discovery_min,discovery_max,refresh],[state,results,selected])
        ingest_button.click(ingest,[state,selected],[ingest_status,status,ids,cats,authors])
        ask_button.click(ask,[question,ids,cats,authors,year_min,year_max,compare],[answer,evidence,details])
        demo.load(library,outputs=[status,ids,cats,authors])
    return demo
