import json
from dataclasses import replace
import pytest
from assistant.artifacts import read_jsonl
from assistant.citations import validate_response
from assistant.filters import eligible_papers
from assistant.retrieval import reciprocal_rank_fusion
from assistant.schemas import Hit, Paper, RetrievalFilters
from assistant.service import AssistantService


def test_provenance_persistence_and_no_document_embedding_during_qa(prepared):
    service,papers,encoder,reranker,generator = prepared
    result = service.ingest(papers)
    assert result["status"] == "ready"
    snapshot = service._open()
    pages = {(p["paper_id"],p["page_number"]):p for p in read_jsonl(snapshot.path/"new_pages.jsonl")}
    assert {(c.paper_id,c.page_number) for c in snapshot.chunks} == {(p.paper_id,n) for p in papers for n in (1,2)}
    for chunk in snapshot.chunks:
        assert chunk.paper_id in {p.paper_id for p in papers}
        assert chunk.start_char < chunk.end_char
        assert chunk.pdf_sha256 and chunk.chunk_id
        assert chunk.text == pages[(chunk.paper_id,chunk.page_number)]["text"][chunk.start_char:chunk.end_char]
    before = service.manifest_path.read_bytes()
    document_calls = len(encoder.calls)
    response = service.ask("What does the paper describe?",RetrievalFilters(paper_ids=("paper-a",)))
    assert response["status"] == "answered"
    assert {s["paper_id"] for s in response["sources"]} == {"paper-a"}
    assert reranker.calls == 1 and generator.calls == 1
    assert len(encoder.calls) == document_calls+1 and len(encoder.calls[-1]) == 1
    assert service.manifest_path.read_bytes() == before
    service.close()
    reopened = AssistantService(service.config,service.root,encoder,reranker,generator)
    try:
        assert reopened.library()["chunks"] == result["chunks"]
        assert reopened.ask("What is described?",RetrievalFilters(year_min=2024))["sources"][0]["paper_id"] == "paper-b"
    finally:
        reopened.close()


def test_idempotency_and_failed_ingestion_preserve_active_snapshot(prepared):
    service,papers,encoder,*_ = prepared
    service.ingest(papers)
    before = service.manifest_path.read_bytes()
    calls = len(encoder.calls)
    assert service.ingest(papers)["status"] == "unchanged"
    assert len(encoder.calls) == calls
    bad = Paper("bad","badv1","Broken",pdf_path=str(service.root/"absent.pdf"))
    with pytest.raises(ValueError):
        service.ingest([bad])
    assert service.manifest_path.read_bytes() == before
    assert service.ask("What is retrieval?")["status"] == "answered"


@pytest.mark.parametrize("mode",["bm25","dense","hybrid","hybrid_rerank"])
def test_metadata_applies_before_branch_top_k_and_no_silent_fallback(prepared,mode):
    service,papers,encoder,reranker,generator = prepared
    service.ingest(papers)
    hits,trace = service.retrieve("paper model retrieval",RetrievalFilters(year_min=2024),mode)
    assert hits and {h.chunk.paper_id for h in hits} == {"paper-b"}
    before = generator.calls
    response = service.ask("What is retrieval?",RetrievalFilters(year_min=2026),mode)
    assert response["status"] == "no_match" and not response["sources"]
    assert generator.calls == before


def test_filter_validation_and_unknown_metadata():
    with pytest.raises(ValueError):
        RetrievalFilters(year_min=2025,year_max=2020)
    with pytest.raises(TypeError):
        RetrievalFilters.from_dict({"arbitrary_operator":"execute"})
    paper = Paper("p","pv1","Unknown year")
    found,stats = eligible_papers([paper],RetrievalFilters(year_min=2020))
    assert not found and stats["excluded_unknown_metadata"] == 1
    assert eligible_papers([paper],RetrievalFilters())[0] == [paper]
    with pytest.raises(ValueError):
        eligible_papers([paper],RetrievalFilters(paper_ids=("missing",)))


def test_citation_ids_and_failures_are_not_correct_refusals():
    sources = [{"label":"S1"}]
    assert validate_response('{"status":"answered","answer":"Fact [S1]."}',sources)[0]
    assert not validate_response('{"status":"answered","answer":"Fact [S2]."}',sources)[0]
    assert not validate_response('{"status":"answered","answer":"No citation."}',sources)[0]
    assert not validate_response('not json',sources)[0]
    assert validate_response('{"status":"insufficient_evidence","answer":"Insufficient evidence."}',sources)[0]


def test_reciprocal_rank_fusion_known_ranks(prepared):
    service,papers,*_ = prepared
    service.ingest(papers)
    a,b = service._open().chunks[:2]
    result = reciprocal_rank_fusion({"x":[Hit(a),Hit(b)],"y":[Hit(b)]})
    assert result[0].chunk.chunk_id == b.chunk_id
    assert result[0].scores["rrf"] == pytest.approx(1/62+1/61)
    assert "cross_encoder" not in result[0].scores


def test_changed_embedding_identity_is_rejected(prepared):
    service,papers,encoder,*_ = prepared
    service.ingest(papers)
    encoder.identity = "different-revision"
    with pytest.raises(ValueError,match="different embedding revision"):
        service.retrieve("retrieval")


def test_prompt_counter_requests_token_ids_not_batchencoding():
    from assistant.generation import Generator
    class ChangedTokenizerAPI:
        def apply_chat_template(self,messages,**options):
            assert options["return_dict"] is False
            return list(range(123))
    generator = Generator.__new__(Generator)
    generator.tokenizer = ChangedTokenizerAPI()
    assert generator.count([{"role":"user","content":"fixture"}]) == 123


def test_structured_claims_render_only_model_selected_existing_sources():
    raw = '{"status":"answered","claims":[{"text":"A supported claim.","source_ids":["S1"]}]}'
    parsed,errors = validate_response(raw,[{"label":"S1"}])
    assert not errors and parsed["answer"] == "A supported claim. [S1]"
    assert not validate_response(raw.replace('"S1"','"S99"'),[{"label":"S1"}])[0]
    assert not validate_response('{"status":"answered","claims":[]}',[{"label":"S1"}])[0]


def test_answered_comparison_must_cite_each_required_paper():
    sources = [{"label":"S1","paper_id":"a"},{"label":"S2","paper_id":"b"}]
    raw = '{"status":"answered","claims":[{"text":"A comparison.","source_ids":["S1"]}]}'
    assert not validate_response(raw,sources,("a","b"))[0]
    assert validate_response(raw.replace('["S1"]','["S1","S2"]'),sources,("a","b"))[0]
