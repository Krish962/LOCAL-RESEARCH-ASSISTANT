"""Real CPU component checks, independent of the scientific QA benchmark."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from assistant.artifacts import atomic_json
from assistant.config import Config, PROJECT_ROOT
from assistant.embeddings import Encoder
from assistant.reranking import Reranker


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=PROJECT_ROOT/'results/runtime-pilot.json')
    args = parser.parse_args()
    import numpy as np
    import psutil
    config = Config.load()
    record = {"kind":"real_component_pilot","config":asdict(config),"components":{},
              "versions":{name:version(name) for name in ["torch","sentence-transformers","transformers","qdrant-client","ollama"]},
              "memory_available_before":psutil.virtual_memory().available}
    output = args.output
    try:
        started = time.perf_counter()
        encoder = Encoder(config)
        embeddings = encoder.encode(["A cat sits on a mat.", "An animal rests on a rug.", "An equation describes black holes."])
        assert embeddings.shape == (3,384) and np.isfinite(embeddings).all()
        assert np.allclose(np.linalg.norm(embeddings,axis=1), 1, atol=1e-4)
        record["components"]["encoder"] = {"status":"passed","load_and_encode_seconds":time.perf_counter()-started,
                                             "identity":encoder.identity,"revision":encoder.revision}
        print("Real encoder passed", flush=True)
        atomic_json(output, record)
        started = time.perf_counter()
        reranker = Reranker(config)
        scores, lengths = reranker.scores("Which planet is called the Red Planet?", [
            "Mars is called the Red Planet.", "Venus has a thick atmosphere."])
        assert scores[0] > scores[1]
        record["components"]["reranker"] = {"status":"passed","scores":scores,"pair_lengths":lengths,
                                              "revision":reranker.revision,"seconds":time.perf_counter()-started}
        print("Real cross-encoder passed", flush=True)
        atomic_json(output, record)
        from assistant.generation import Generator
        from assistant.citations import validate_response
        generator = Generator(config)
        source = {"label":"S1","title":"Runtime fixture","arxiv_id":"fixture","page_number":1,
                  "text":"Mars is commonly called the Red Planet.","paper_id":"fixture","chunk_id":"fixture","start_char":0}
        models = generator.client.list().model_dump(mode="json")
        record["ollama_models"] = models
        started = time.perf_counter()
        answer = generator.generate("Which planet is called the Red Planet?", [source])
        assert answer["counted_prompt_tokens"] > 20
        assert abs(answer["counted_prompt_tokens"] - answer["prompt_eval_count"]) <= 8
        parsed, errors = validate_response(answer["message"]["content"], [source])
        assert parsed and not errors, errors
        assert parsed["status"] == "answered" and "mars" in parsed["answer"].lower()
        record["components"]["generator"] = {"status":"passed","seconds":time.perf_counter()-started,
                                              "raw":answer,"parsed":parsed}
        print("Real Qwen grounded answer passed", flush=True)
        if config.grounding_enabled:
            from assistant.grounding import GroundingVerifier
            verifier = GroundingVerifier(config)
            support = verifier.verify(parsed,[source])
            assert support['passed'],support
            bad = {'status':'answered','claims':[{'text':'Mars is called the Blue Planet.',
                'source_ids':['S1'],'evidence':[{'source_id':'S1','quote':source['text']}]}]}
            rejected = verifier.verify(bad,[source])
            assert not rejected['passed']
            record['components']['grounding']={'status':'passed','supported_fixture':support,'rejected_fixture':rejected}
            print('Real NLI claim checks passed',flush=True)
            from assistant.evidence_selection import render_selection
            candidates=[{'evidence_id':'E1','source_id':'S1','title':source['title'],'paper_id':'fixture',
                'page_number':1,'chunk_id':'fixture','quote':source['text'],'start_char':0,'end_char':len(source['text'])}]
            selected=generator.select_evidence('Which planet is called the Red Planet?',candidates)
            extracted=render_selection(json.loads(selected['message']['content']),candidates,[source])
            assert extracted['status']=='answered' and verifier.verify(extracted,[source])['passed']
            record['components']['evidence_selection']={'status':'passed','raw':selected,'parsed':extracted}
            print('Real source-ID evidence selection passed',flush=True)
        record["status"] = "passed"
    except Exception as exc:
        record.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise
    finally:
        record["memory_available_after"] = psutil.virtual_memory().available
        atomic_json(output, record)


if __name__ == "__main__":
    main()
