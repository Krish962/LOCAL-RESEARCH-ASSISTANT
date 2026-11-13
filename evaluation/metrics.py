from __future__ import annotations

import math
import re


def chunk_of(hit: dict) -> dict:
    return hit.get("chunk", hit)


def covers(chunk: dict, alternative: dict) -> bool:
    if chunk.get("paper_id") != alternative["paper_id"] or chunk.get("page_number") != alternative["page_number"]:
        return False
    if "quote" in alternative:
        return alternative["quote"] in chunk.get("text", "")
    return chunk["start_char"] <= alternative["start_char"] and chunk["end_char"] >= alternative["end_char"]


def evidence_metrics(hits: list[dict], groups: list[dict], k=4) -> dict:
    chunks = [chunk_of(h) for h in hits[:k]]
    matched = {i for i,g in enumerate(groups) if any(covers(c,a) for c in chunks for a in g["alternatives"])}
    relevant = [any(covers(c,a) for g in groups for a in g["alternatives"]) for c in chunks]
    first = next((i for i,value in enumerate(relevant,1) if value), None)
    gold_pages = {(a["paper_id"],a["page_number"]) for g in groups for a in g["alternatives"]}
    got_pages = {(c["paper_id"],c["page_number"]) for c in chunks}
    return {"k":k,"returned":len(chunks),"gold_groups":len(groups),"matched_groups":len(matched),
            "evidence_recall":len(matched)/len(groups) if groups else None,
            "hit":int(bool(matched)) if groups else None,
            "mrr":(1/first if first else 0.) if groups else None,
            "page_recall":len(gold_pages & got_pages)/len(gold_pages) if gold_pages else None}


def graded_metrics(hits: list[dict], qrels: dict[str,int], k=4) -> dict:
    ids = [chunk_of(h)["chunk_id"] for h in hits[:k]]
    judged = sum(key in qrels for key in ids)
    result = {"k":k,"returned":len(ids),"judged":judged,"judgment_coverage":judged/len(ids) if ids else None,
              "precision":None,"direct_precision":None,"mrr_direct":None,"ndcg":None}
    if not qrels or judged != len(ids):
        return result
    grades = [qrels[key] for key in ids]
    result["precision"] = sum(g>0 for g in grades)/k
    result["direct_precision"] = sum(g==2 for g in grades)/k
    first_direct = next((i for i,g in enumerate(grades,1) if g==2),None)
    result["mrr_direct"] = 1/first_direct if first_direct else 0.
    dcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(grades))
    ideal = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(sorted(qrels.values(),reverse=True)[:k]))
    result["ndcg"] = dcg/ideal if ideal else None
    return result


def citation_ids(answer: str, sources: list[dict]) -> dict:
    labels = re.findall(r"\[S(\d+)\]",answer)
    valid = sum(f"S{n}" in {s["label"] for s in sources} for n in labels)
    return {"emitted":len(labels),"valid":valid,"validity":valid/len(labels) if labels else None,
            "missing_citations":not bool(labels)}
