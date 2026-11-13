from __future__ import annotations

import json
import re

LABEL = re.compile(r"\[S(\d+)\]")


def validate_response(raw: str, sources: list[dict], required_papers=()) -> tuple[dict | None, list[str]]:
    normalized_claims = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, ["Response is not a JSON object"]
    if isinstance(data, dict) and data.get("status") == "answered" and set(data) == {"status","claims"}:
        claims = data["claims"]
        if not isinstance(claims,list) or not 1 <= len(claims) <= 4:
            return None,["Answered responses require one to four sourced claims"]
        allowed_labels = {s["label"] for s in sources}
        rendered = []
        normalized_claims = []
        for claim in claims:
            if not isinstance(claim,dict) or set(claim) not in ({"text","source_ids"},{"text","source_ids","evidence"}):
                return None,["Each claim needs text and source_ids"]
            ids = claim["source_ids"]
            if not isinstance(claim["text"],str) or not claim["text"].strip() or not isinstance(ids,list) or not ids:
                return None,["Empty claim or missing supporting source IDs"]
            if any(not isinstance(label,str) or label not in allowed_labels for label in ids):
                return None,["Claim source IDs must identify supplied evidence"]
            if 'evidence' in claim:
                evidence = claim['evidence']
                if not isinstance(evidence,list) or not 1 <= len(evidence) <= 4:
                    return None,['Claims need one to four supporting quotations']
                for item in evidence:
                    if (not isinstance(item,dict) or set(item) != {'source_id','quote'}
                        or not isinstance(item['source_id'],str) or item['source_id'] not in ids
                        or not isinstance(item['quote'],str) or not 8 <= len(item['quote'].strip()) <= 400):
                        return None,['Each quotation needs a cited source_id and 8–400 verbatim characters']
                if {item['source_id'] for item in evidence} != set(ids):
                    return None,['Every cited source requires a supporting quotation']
            normalized_claims.append(claim)
            rendered.append(claim["text"].strip() + " " + " ".join(f"[{label}]" for label in dict.fromkeys(ids)))
        data = {"status":"answered","answer":"\n\n".join(rendered)}
    elif isinstance(data,dict) and data.get("status") == "insufficient_evidence" and set(data) == {"status","claims","reason"}:
        if data["claims"] != [] or not isinstance(data["reason"],str) or not data["reason"].strip():
            return None,["Insufficient-evidence responses need empty claims and a reason"]
        data = {"status":"insufficient_evidence","answer":data["reason"].strip()}
    if not isinstance(data, dict) or set(data) != {"status", "answer"}:
        return None, ["Response needs exactly status and answer fields"]
    if data["status"] not in {"answered", "insufficient_evidence"} or not isinstance(data["answer"], str) or not data["answer"].strip():
        return None, ["Invalid or empty status/answer"]
    cited = LABEL.findall(data["answer"])
    allowed = {str(i) for i in range(1, len(sources) + 1)}
    errors = []
    if set(cited) - allowed:
        errors.append("Citations must refer only to supplied source labels")
    if data["status"] == "answered" and not cited:
        errors.append("Answered factual content must cite supplied source labels")
    if data["status"] == "answered" and required_papers:
        by_label = {s["label"]:s.get("paper_id") for s in sources}
        cited_papers = {by_label.get("S"+number) for number in cited}
        if set(required_papers) - cited_papers:
            errors.append("A comparison must address and cite evidence from every requested paper")
    if normalized_claims is not None and not errors:
        data['claims'] = normalized_claims
    return (None if errors else data), errors
