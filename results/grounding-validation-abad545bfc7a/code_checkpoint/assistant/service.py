from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .artifacts import atomic_json, write_jsonl
from .chunking import chunk_page
from .citations import validate_response
from .config import Config, PROJECT_ROOT, fingerprint
from .context import pack_context
from .extraction import extract_pages
from .filters import eligible_papers
from .lexical import LexicalIndex
from .retrieval import reciprocal_rank_fusion
from .schemas import Paper, RetrievalFilters, validate_corpus_id
from .storage import Snapshot, build_snapshot
from .telemetry import Trace


class AssistantService:
    def __init__(self, config: Config | None = None, root: Path = PROJECT_ROOT,
                 encoder=None, reranker=None, generator=None, grounder=None):
        self.config, self.root = config or Config.load(), root.resolve()
        validate_corpus_id(self.config.corpus_id)
        self._encoder, self._reranker, self._generator = encoder, reranker, generator
        self._snapshot, self._lexical = None, None
        self._grounder = grounder
        self.lock = threading.RLock()
        self.manifest_path = self.root / "data/manifests" / f"{self.config.corpus_id}.json"

    @property
    def encoder(self):
        if self._encoder is None:
            from .embeddings import Encoder
            self._encoder = Encoder(self.config, self.root)
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            from .reranking import Reranker
            self._reranker = Reranker(self.config)
        return self._reranker

    @property
    def generator(self):
        if self._generator is None:
            from .generation import Generator
            self._generator = Generator(self.config)
        return self._generator

    @property
    def grounder(self):
        if self._grounder is None:
            from .grounding import GroundingVerifier
            self._grounder = GroundingVerifier(self.config, self.root)
        return self._grounder

    def _manifest(self):
        if not self.manifest_path.exists():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if data.get("status") != "ready" or data.get("corpus_id") != self.config.corpus_id:
            raise ValueError("Active manifest is not a ready snapshot of the selected corpus")
        return data

    def _open(self):
        manifest = self._manifest()
        if manifest is None:
            return None
        path = (self.root / manifest["snapshot"]).resolve()
        if not path.is_relative_to(self.root / "data/indexes"):
            raise ValueError("Snapshot path escapes the project index directory")
        if self._snapshot is None or self._snapshot.path != path:
            self.close()
            self._snapshot = Snapshot(path, self.config)
            self._lexical = LexicalIndex(self._snapshot.chunks, self.config.bm25_k1, self.config.bm25_b)
        return self._snapshot

    def close(self):
        if self._snapshot is not None:
            self._snapshot.close()
            self._snapshot, self._lexical = None, None

    def library(self) -> dict:
        with self.lock:
            manifest = self._manifest()
            return manifest or {"status":"empty","corpus_id":self.config.corpus_id,"papers":[],"chunks":0}

    def ingest(self, papers: list[Paper]) -> dict:
        import numpy as np
        with self.lock:
            trace = Trace(self.root, "ingestion")
            try:
                previous = self._open()
                old_papers = {p.paper_id:p for p in previous.papers} if previous else {}
                merged = dict(old_papers)
                if len({p.paper_id for p in papers}) != len(papers):
                    raise ValueError("Select only one version per canonical paper ID")
                for paper in papers:
                    path = Path(paper.pdf_path)
                    if not path.is_file():
                        raise ValueError(f"PDF does not exist for {paper.paper_id}")
                    with path.open("rb") as handle:
                        actual = hashlib.file_digest(handle, "sha256").hexdigest()
                    if paper.pdf_sha256 and actual != paper.pdf_sha256:
                        raise ValueError(f"PDF fingerprint changed for {paper.paper_id}")
                    paper.pdf_sha256 = actual
                    merged[paper.paper_id] = paper
                if not merged:
                    raise ValueError("No papers selected")
                encoder = self.encoder
                index_id = fingerprint({"embedding":encoder.identity,
                    "chunking":{"tokens":self.config.chunk_tokens,"overlap":self.config.chunk_overlap,"version":1},
                    "extractor":self.config.extractor,"extractor_version":version(self.config.extractor),"normalization":"nfc-v1"})
                old_manifest = self._manifest()
                old_chunks = previous.chunks if previous and old_manifest["index_config_id"] == index_id else []
                old_vectors = {c.chunk_id:v for c,v in zip(previous.chunks, previous.vectors)} if old_chunks else {}
                unchanged = {key for key,p in merged.items() if key in old_papers and p.to_dict() == old_papers[key].to_dict()}
                if old_chunks and len(unchanged) == len(merged):
                    trace.data.update({"status":"unchanged","document_embeddings":0,"skipped":len(merged)})
                    trace.save()
                    return {"status":"unchanged","added":0,"skipped":len(merged),"failed":0,"trace_id":trace.data["run_id"]}
                chunks, pages = [], []
                with trace.stage("extract_and_chunk"):
                    for paper_id in sorted(merged):
                        paper = merged[paper_id]
                        reusable = [c for c in old_chunks if c.paper_id == paper_id] if paper_id in unchanged else []
                        if reusable:
                            chunks.extend(reusable)
                            continue
                        extracted = extract_pages(paper, self.config.extractor)
                        pages.extend(asdict(p) for p in extracted)
                        created = [c for page in extracted for c in chunk_page(page, paper, encoder.tokenizer, self.config, index_id)]
                        if not created:
                            raise ValueError(f"No extractable content in {paper_id}; previous corpus remains active")
                        chunks.extend(created)
                new = [c for c in chunks if c.chunk_id not in old_vectors]
                with trace.stage("document_embedding"):
                    values = encoder.encode([c.text for c in new])
                old_vectors.update({c.chunk_id:v for c,v in zip(new, values)})
                vectors = np.asarray([old_vectors[c.chunk_id] for c in chunks], dtype=np.float32)
                ready_papers = [merged[key] for key in sorted(merged)]
                snapshot_id = fingerprint({"index_id":index_id,"papers":[p.to_dict() for p in ready_papers]})[:24]
                snapshot_path = self.root / "data/indexes" / self.config.corpus_id / snapshot_id
                self.close()
                with trace.stage("index_write"):
                    state = build_snapshot(snapshot_path, self.config, ready_papers, chunks, vectors)
                write_jsonl(snapshot_path / "new_pages.jsonl", pages)
                manifest = {"status":"ready","corpus_id":self.config.corpus_id,"snapshot_id":snapshot_id,
                    "snapshot":snapshot_path.relative_to(self.root).as_posix(),"index_config_id":index_id,
                    "embedding_identity":encoder.identity,"papers":[p.to_dict() for p in ready_papers],
                    "chunks":len(chunks),"state":state,"config":asdict(self.config)}
                # Activate only after files and real Qdrant state have been verified.
                reopened = Snapshot(snapshot_path, self.config)
                reopened.close()
                atomic_json(self.manifest_path, manifest)
                trace.data.update({"status":"ready","document_embeddings":len(new),"papers":len(ready_papers),
                    "chunks":len(chunks),"page_status_counts":{s:sum(p['status']==s for p in pages) for s in {p['status'] for p in pages}},
                    "snapshot_id":snapshot_id})
                return {"status":"ready","added":len(merged)-len(unchanged),"skipped":len(unchanged),
                        "failed":0,"chunks":len(chunks),"trace_id":trace.data["run_id"]}
            except Exception as exc:
                trace.data.update({"status":"error","error":str(exc)})
                raise
            finally:
                trace.save()

    def retrieve(self, question: str, filters: RetrievalFilters | None = None,
                 mode="hybrid_rerank", trace: Trace | None = None):
        with self.lock:
            if mode not in {"bm25","dense","hybrid","hybrid_rerank"}:
                raise ValueError("Unsupported retrieval mode")
            question = question.strip()
            if not question or len(question) > 1600:
                raise ValueError("Question must contain 1-1600 characters")
            trace = trace or Trace(self.root)
            trace.data.update({"question":question,"filters":asdict(filters or RetrievalFilters()),
                               "mode":mode,"config":asdict(self.config),"config_hash":self.config.identity()})
            snapshot = self._open()
            if snapshot is None:
                trace.data.update({"eligibility":{"eligible_papers":0},"retrieval":{}})
                return [], trace
            manifest = self._manifest()
            if manifest["embedding_identity"] != self.encoder.identity:
                raise ValueError("Active snapshot uses a different embedding revision; re-ingest into a new snapshot")
            if self.encoder.count(question) > 128:
                raise ValueError("Please shorten the question to at most 126 encoder wordpieces")
            with trace.stage("metadata_filter"):
                papers, stats = eligible_papers(snapshot.papers, filters or RetrievalFilters())
            eligible = {p.paper_id for p in papers}
            trace.data.update({"eligibility":stats,"snapshot_id":manifest["snapshot_id"],
                               "index_config_id":manifest["index_config_id"]})
            if not eligible:
                trace.data["retrieval"] = {}
                return [], trace
            branches = {}
            if mode in {"bm25","hybrid","hybrid_rerank"}:
                with trace.stage("bm25"):
                    branches["bm25"] = self._lexical.search(question, eligible, self.config.branch_top_k)
            if mode in {"dense","hybrid","hybrid_rerank"}:
                with trace.stage("query_embedding"):
                    vector = self.encoder.encode([question])[0]
                with trace.stage("dense"):
                    branches["dense"] = snapshot.search(vector, eligible, self.config.branch_top_k)
            with trace.stage("fusion"):
                fused = reciprocal_rank_fusion(branches, self.config.rrf_constant)
            trace.data["retrieval"] = {name:[h.to_dict() for h in hits] for name,hits in branches.items()}
            trace.data["retrieval"]["fused"] = [h.to_dict() for h in fused]
            candidates = fused[:self.config.rerank_top_n]
            if mode == "hybrid_rerank" and candidates:
                with trace.stage("reranking"):
                    candidates, info = self.reranker.rerank(question, candidates)
                trace.data["reranking"] = info
            trace.data["ranked"] = [h.to_dict() for h in candidates]
            if any(h.chunk.paper_id not in eligible for h in candidates):
                raise RuntimeError("A result escaped the metadata constraints")
            return candidates, trace

    def _extractive_answer(self, question, sources, required_papers, trace):
        from .evidence_selection import units,render_selection,supported_selection,focused_review
        from .grounding import question_coverage
        fallback={'status':'insufficient_evidence','answer':'The retrieved excerpts do not provide a verifiable answer to the requested question.','sources':sources}
        with trace.stage('evidence_unit_ranking'):
            candidates=units(question,sources,self.reranker,required_papers=required_papers)
        trace.data['evidence_units']=candidates
        if not candidates:return fallback
        batches=[candidates]
        focused=focused_review(candidates,required_papers)
        if focused and max(c['relevance_score'] for c in focused)>=0:
            # Small models can miss a direct answer in a long list. Recheck at
            # most two individual excerpts; never force a positive decision.
            batches += ([focused] if required_papers else [[c] for c in focused[:2]])
        trace.data['extractive_attempts']=[]
        for attempt,batch in enumerate(batches):
            with trace.stage('evidence_selection' if attempt==0 else f'evidence_selection_retry_{attempt}'):
                selected=self.generator.select_evidence(question,batch,required_papers)
            if attempt==0:trace.data['evidence_selection']=selected
            visible=set(selected.get('visible_evidence_ids',[c['evidence_id'] for c in batch]))
            available=[c for c in batch if c['evidence_id'] in visible]
            data=json.loads(selected['message']['content'])
            extracted=render_selection(data,available,sources,required_papers)
            record={'selection':selected,'status':extracted['status']}
            trace.data['extractive_attempts'].append(record)
            if extracted['status']!='answered':continue
            chosen_labels={c['source_id'] for c in extracted['selected_evidence']}
            chosen_sources=[s for s in sources if s['label'] in chosen_labels]
            with trace.stage('extractive_verification' if attempt==0 else f'extractive_verification_retry_{attempt}'):
                support=self.grounder.verify(extracted,chosen_sources)
            record['support']=support
            trace.data['extractive_grounding']={'support':support}
            kept=supported_selection(data,available,support)
            record['pruned_evidence_ids']=[i for i in data['evidence_ids'] if i not in kept['evidence_ids']]
            if kept['status']!='answered':continue
            extracted=render_selection(kept,available,sources,required_papers)
            if extracted['status']!='answered':continue
            selected_coverage=question_coverage(question,[{'text':c['title']+'\n'+c['quote']} for c in extracted['selected_evidence']])
            selected_coverage['scope']='selected excerpts plus verified paper titles'
            trace.data['extractive_grounding']['coverage']=selected_coverage
            if not selected_coverage['passed']:continue
            accepted=[c for c in support['claims'] if c['passed']]
            trace.data['grounding']['claim_verification']={**support,'passed':True,'errors':[],
                'claims':[{**c,'original_claim_index':c['claim_index'],'claim_index':i} for i,c in enumerate(accepted)]}
            return {**extracted,'sources':sources}
        return fallback

    def ask(self, question: str, filters: RetrievalFilters | None = None,
            mode="hybrid_rerank", required_papers=(), question_id=None) -> dict:
        with self.lock:
            trace = Trace(self.root)
            trace.data["question_id"] = question_id
            try:
                hits, trace = self.retrieve(question, filters, mode, trace)
                if not hits:
                    result = {"status":"no_match","answer":"No indexed passages match the selected scope. The filters were not relaxed.","sources":[]}
                else:
                    if set(required_papers) - set(trace.data["eligibility"]["eligible_paper_ids"]):
                        raise ValueError("Required comparison papers must be eligible in the selected scope")
                    with trace.stage("context_packing"):
                        sources, context_info = pack_context(question, hits, self.generator.count, self.config, required_papers)
                    trace.data.update({"context":context_info,"sources":sources})
                    from .grounding import question_coverage
                    with trace.stage('question_coverage'):
                        coverage = question_coverage(question, sources) if self.config.grounding_enabled else {'passed':True,'disabled':True}
                    trace.data['grounding'] = {'enabled':self.config.grounding_enabled,'question_coverage':coverage}
                    if not sources or context_info["missing_required_papers"]:
                        result = {"status":"insufficient_evidence","answer":"The retrieved evidence does not cover the requested sources within the context budget.","sources":sources}
                    elif not coverage['passed']:
                        result = {'status':'insufficient_evidence', 'answer':'The retrieved passages do not mention the requested identifier or measurement unit: '
                            + ', '.join(coverage['missing_identifiers']+coverage.get('missing_units',[])) + '. I cannot verify the requested result from this evidence.', 'sources':sources}
                    elif (self.config.grounding_enabled and self.config.grounding_extractive_first
                          and callable(getattr(self.generator,'select_evidence',None))):
                        result=self._extractive_answer(question,sources,required_papers,trace)
                    else:
                        attempts, parsed, errors = [], None, []
                        rejected_support = False
                        for attempt in range(2):
                            with trace.stage(f"generation_{attempt}"):
                                response = self.generator.generate(question, sources, "; ".join(errors)[:350])
                            raw = response["message"]["content"]
                            parsed, errors = validate_response(raw, sources, required_papers)
                            support = None
                            if parsed and parsed['status']=='answered' and self.config.grounding_enabled:
                                with trace.stage(f'claim_verification_{attempt}'):
                                    support = self.grounder.verify(parsed, sources)
                                trace.data['grounding']['claim_verification'] = support
                                if not support['passed']:
                                    rejected_support = True
                                    parsed = None
                                    errors = support['errors'] or ['Provide verifiable supporting quotations or state insufficient evidence']
                            attempts.append({"response":response,"validation_errors":errors,'grounding':support})
                            if parsed:
                                break
                            if self.config.grounding_extractive_fallback and callable(getattr(self.generator,'select_evidence',None)):
                                break
                        trace.data["generation_attempts"] = attempts
                        fallback = ({'status':'insufficient_evidence','answer':'I could not verify the requested answer against the cited passages.'}
                            if rejected_support else {"status":"invalid_output","answer":"The model did not return a valid cited answer. Inspect the retrieved passages."})
                        result = {**(parsed or fallback),"sources":sources}
                        if (result['status'] in {'insufficient_evidence','invalid_output'} and self.config.grounding_enabled
                            and self.config.grounding_extractive_fallback and callable(getattr(self.generator,'select_evidence',None))):
                            extracted=self._extractive_answer(question,sources,required_papers,trace)
                            if extracted['status']=='answered':result=extracted
                if self.config.grounding_enabled:
                    verification = trace.data.get('grounding', {})
                    supported = result['status']=='answered' and verification.get('claim_verification',{}).get('passed',False)
                    result['evidence_checks'] = {'enabled':True,'passed':supported,
                        'outcome':'supported_by_checks' if supported else result['status']}
                trace.data["result"] = result
            except Exception as exc:
                result = {"status":"error","answer":str(exc),"sources":trace.data.get("sources",[])}
                trace.data.update({"result":result,"error_type":type(exc).__name__})
            trace.data["status"] = result["status"]
            path = trace.save()
            return {**result,"trace_id":trace.data["run_id"],"trace_path":path,
                    "timings_ms":trace.data["timings_ms"],"filters":asdict(filters or RetrievalFilters())}
