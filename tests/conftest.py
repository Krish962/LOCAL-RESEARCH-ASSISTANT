from __future__ import annotations

import re
from dataclasses import replace
import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from assistant.config import Config
from assistant.schemas import Paper
from assistant.service import AssistantService


class Tokenizer:
    is_fast = True

    def __call__(self,text,**kwargs):
        spans = [m.span() for m in re.finditer(r"\S+",text)]
        return {"input_ids":list(range(len(spans))),"offset_mapping":spans}


class Encoder:
    tokenizer = Tokenizer()
    identity = "test-encoder-v1"

    def __init__(self):
        self.calls = []

    def count(self,text):
        return len(text.split())

    def encode(self,texts):
        self.calls.append(list(texts))
        values = []
        for text in texts:
            vector = np.array([1+text.lower().count(w) for w in ["retrieval","training","paper","model"]],dtype=np.float32)
            values.append(vector/np.linalg.norm(vector))
        return np.asarray(values,dtype=np.float32).reshape(-1,4)


class Reranker:
    def __init__(self):
        self.calls = 0

    def rerank(self,question,hits):
        self.calls += 1
        for hit in hits:
            hit.scores["cross_encoder"] = float(hit.chunk.text.lower().count("retrieval"))
        return sorted(hits,key=lambda h:-h.scores["cross_encoder"]),{"executed":True,"pairs":len(hits)}

    def scores(self,question,texts):
        return [float(t.lower().count('retrieval')) for t in texts],[len(t.split()) for t in texts]


class Generator:
    def __init__(self):
        self.calls = 0

    def count(self,messages):
        return sum(len(m["content"].split()) for m in messages)

    def generate(self,question,sources,repair=""):
        self.calls += 1
        return {"message":{"content":'{"status":"answered","answer":"The supplied paper discusses retrieval [S1]."}'}}


def make_pdf(path,pages):
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612,height=792)
        font = DictionaryObject({NameObject("/Type"):NameObject("/Font"),NameObject("/Subtype"):NameObject("/Type1"),NameObject("/BaseFont"):NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):writer._add_object(font)})})
        stream = DecodedStreamObject()
        escaped = text.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
        stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


@pytest.fixture
def prepared(tmp_path):
    config = replace(Config(),corpus_id="test_corpus",embedding_dimension=4,chunk_tokens=12,chunk_overlap=2,grounding_enabled=False)
    encoder,reranker,generator = Encoder(),Reranker(),Generator()
    service = AssistantService(config,tmp_path,encoder,reranker,generator)
    papers = []
    for key,year,text in [("paper-a",2020,"Paper alpha describes retrieval with dense vectors and clear evidence."),
                          ("paper-b",2024,"Paper beta describes model training with carefully measured data.")]:
        path = tmp_path / f"{key}.pdf"
        make_pdf(path,[text,"A second page has a distinct example and a reported limitation."])
        papers.append(Paper(key,key+"v1",key,[key+" author"],categories=["cs.IR" if year==2020 else "cs.CL"],
                            first_submitted_year=year,pdf_path=str(path)))
    yield service,papers,encoder,reranker,generator
    service.close()
