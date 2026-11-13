import json
from dataclasses import replace
import pytest
from assistant.config import Config
from assistant.citations import validate_response
from assistant.grounding import GroundingVerifier, identifiers, contains_identifier, numbers, question_coverage, quotation_span


SOURCE = {'label':'S1','chunk_id':'chunk-a','paper_id':'paper-a','page_number':1,
          'text':'Atlas-7B achieves 42.5% accuracy on the test set. It uses a dual encoder.'}


class NLI:
    identity = {'model':'test-only'}
    def __init__(self, entails=True):
        self.calls = []
        self.entails = entails
    def score(self, pairs):
        self.calls.extend(pairs)
        return [{'entailment':.95 if self.entails else .03,'neutral':.04 if self.entails else .96,'contradiction':.01} for _ in pairs]


def claim(text='Atlas-7B achieves 42.5% accuracy.', quote=None):
    return {'status':'answered','claims':[{'text':text,'source_ids':['S1'],
        'evidence':[{'source_id':'S1','quote':quote or SOURCE['text'].split(' It ')[0]}]}]}


def test_identifiers_and_requested_version_are_not_matched_by_prefix():
    assert identifiers('What does Atlas-7B report for Orion 3.2 3B?') == ['Atlas-7B','Orion 3.2 3B']
    assert contains_identifier('An ATLAS 7B model.', 'Atlas-7B')
    assert not contains_identifier('An Atlas-70B model.', 'Atlas-7B')
    assert not contains_identifier('GPT-40 is present.', 'GPT-4')
    assert not question_coverage('What does Orion-13B score?', [SOURCE])['passed']
    assert not question_coverage('what does orion-13b score?', [SOURCE])['passed']
    assert identifiers('What changed in 2020?')==[]
    assert not question_coverage('What does ChatGPT report?', [SOURCE])['passed']
    assert not question_coverage('What energy in kilowatt-hours is reported?', [SOURCE])['passed']
    assert not question_coverage('What financial cost in US dollars is reported?', [SOURCE])['passed']
    assert question_coverage('What energy in kilowatt-hours is reported?', [{'text':'The energy use is 2.5kWh.'}])['passed']


def test_exact_quote_and_parent_both_require_entailment():
    model=NLI()
    result=GroundingVerifier(Config(),model=model).verify(claim(),[SOURCE])
    assert result['passed'] and len(model.calls)==2
    assert result['claims'][0]['evidence'][0]['chunk_id']=='chunk-a'
    assert not GroundingVerifier(Config(),model=NLI(False)).verify(claim(),[SOURCE])['passed']


@pytest.mark.parametrize('payload',[
    claim(quote='Atlas-7B achieves 99% accuracy.'),
    claim(text='Atlas-7B achieves 99% accuracy.'),
    claim(text='Orion-7B achieves 42.5% accuracy.'),
    {'status':'answered','answer':'A claim with an existing citation [S1].'},
])
def test_fabricated_quotes_numbers_entities_and_unverified_legacy_output_are_rejected(payload):
    model=NLI()
    result=GroundingVerifier(Config(),model=model).verify(payload,[SOURCE])
    assert not result['passed'] and not model.calls


def test_parent_passage_can_reject_a_cherry_picked_quote():
    class ParentReject(NLI):
        def score(self,pairs):
            return [{'entailment':.95,'contradiction':.01,'neutral':.04},
                    {'entailment':.01,'contradiction':.98,'neutral':.01}]
    assert not GroundingVerifier(Config(),model=ParentReject()).verify(claim(),[SOURCE])['passed']


def test_neutral_parent_does_not_overrule_a_supported_quote():
    class ParentNeutral(NLI):
        def score(self,pairs):
            return [{'entailment':.95,'contradiction':.01,'neutral':.04},
                    {'entailment':.15,'contradiction':.05,'neutral':.8}]
    assert GroundingVerifier(Config(),model=ParentNeutral()).verify(claim(),[SOURCE])['passed']


def test_quote_matching_allows_layout_whitespace_without_changing_numbers():
    source={**SOURCE,'text':SOURCE['text'].replace('42.5% accuracy','42.5%\naccuracy')}
    assert GroundingVerifier(Config(),model=NLI()).verify(claim(),[source])['passed']
    assert numbers('1,000 samples and 0.50 accuracy')==numbers('1000 samples and .50 accuracy'.replace('.50','0.5'))
    text='The knowl-\nedge retriever learns a sig-\nnal.'
    span=quotation_span(text,'The knowledge retriever learns a signal.')
    assert span==(0,len(text))
    assert quotation_span(text,'The knowledge retriever never learns a signal.') is None


def test_schema_keeps_evidence_for_semantic_verification():
    parsed,errors=validate_response(json.dumps(claim()),[SOURCE])
    assert not errors and parsed['claims'][0]['evidence']


def test_service_refuses_missing_identifiers_before_generation(prepared):
    service,papers,encoder,reranker,generator=prepared
    service.config=replace(service.config,grounding_enabled=True)
    service.ingest(papers)
    result=service.ask('What accuracy does Orion-13B achieve?')
    assert result['status']=='insufficient_evidence' and generator.calls==0


def test_service_rejects_semantic_draft_and_retries_without_exposing_it(prepared):
    service,papers,encoder,reranker,generator=prepared
    service.ingest(papers)
    service.config=replace(service.config,grounding_enabled=True)
    service._grounder=GroundingVerifier(service.config,model=NLI(False))
    def draft(question,sources,repair=''):
        generator.calls+=1
        s=sources[0]
        return {'message':{'content':json.dumps({'status':'answered','claims':[{
            'text':'The paper guarantees perfect retrieval.','source_ids':[s['label']],
            'evidence':[{'source_id':s['label'],'quote':s['text']}]}]})}}
    generator.generate=draft
    result=service.ask('What retrieval guarantee is stated?')
    assert result['status']=='insufficient_evidence' and generator.calls==2
    assert 'perfect retrieval' not in result['answer']


def test_verifier_runtime_failure_is_reported_as_error_not_correct_refusal(prepared):
    service,papers,encoder,reranker,generator=prepared
    service.ingest(papers)
    service.config=replace(service.config,grounding_enabled=True)
    class Broken:
        def verify(self,*args): raise RuntimeError('model failure')
    service._grounder=Broken()
    generator.generate=lambda *args,**kwargs: {'message':{'content':'{"status":"answered","claims":[{"text":"A fact.","source_ids":["S1"]}]}'}}
    assert service.ask('What does this describe?')['status']=='error'


def test_extractive_fallback_copies_source_text_instead_of_an_unsupported_claim(prepared):
    from assistant.evidence_selection import excerpt_text
    service,papers,encoder,reranker,generator=prepared
    service.ingest(papers)
    service.config=replace(service.config,grounding_enabled=True,grounding_extractive_first=False)
    class LiteralNLI(NLI):
        def score(self,pairs):
            return [{'entailment':.99 if excerpt_text(h) in excerpt_text(p) else .01,
                'contradiction':.001,'neutral':.009 if excerpt_text(h) in excerpt_text(p) else .989} for p,h in pairs]
    service._grounder=GroundingVerifier(service.config,model=LiteralNLI())
    generator.generate=lambda question,sources,repair='':{'message':{'content':json.dumps({
        'status':'answered','claims':[{'text':'This guarantees perfect performance.',
            'source_ids':['S1'],'evidence':[{'source_id':'S1','quote':sources[0]['text']}]}]})}}
    generator.select_evidence=lambda question,candidates,required: {'message':{'content':json.dumps({
        'status':'answered','evidence_ids':[candidates[0]['evidence_id']]})}}
    result=service.ask('What retrieval method is described?')
    assert result['status']=='answered' and result['answer_mode']=='extractive'
    assert 'guarantees perfect' not in result['answer']
    assert result['evidence_checks']['passed']


def test_evidence_selection_rejects_ids_outside_the_source_pool():
    from assistant.evidence_selection import render_selection
    with pytest.raises(ValueError):
        render_selection({'status':'answered','evidence_ids':['E999']},[],[])


def test_lowercase_metric_sentences_and_long_evidence_are_retained():
    from assistant.evidence_selection import units
    class Rank:
        def scores(self,q,texts):return [1.]*len(texts),[len(t.split()) for t in texts]
    source={**SOURCE,'title':'Metric paper','text':('A long preceding sentence. '*20)+
        'nDCG@10 serves as the main metric. Other details follow.'}
    candidates=units('Which metric?', [source],Rank(),limit=100)
    assert any('nDCG@10 serves as the main metric.' in c['quote'] for c in candidates)
    assert all(len(c['quote'])<=400 for c in candidates)


def test_default_evidence_first_mode_never_generates_free_form_research_claims(prepared):
    service,papers,encoder,reranker,generator=prepared
    service.ingest(papers)
    service.config=replace(service.config,grounding_enabled=True)
    service._grounder=GroundingVerifier(service.config,model=NLI())
    def free_form(*args,**kwargs):raise AssertionError('Default mode must select source IDs instead')
    generator.generate=free_form
    generator.select_evidence=lambda question,candidates,required:{'message':{'content':json.dumps({
        'status':'answered','evidence_ids':[candidates[0]['evidence_id']]})}}
    result=service.ask('What retrieval method is described?')
    assert result['status']=='answered' and result['answer_mode']=='extractive'


def test_valid_excerpts_survive_another_selected_excerpts_rejection():
    from assistant.evidence_selection import supported_selection
    candidates=[{'evidence_id':'E1'},{'evidence_id':'E2'}]
    selected=supported_selection({'status':'answered','evidence_ids':['E1','E2']},candidates,
        {'claims':[{'passed':True},{'passed':False}]})
    assert selected=={'status':'answered','evidence_ids':['E1']}
    with pytest.raises(ValueError):
        supported_selection({'evidence_ids':['E999']},candidates,{'claims':[{'passed':True}]})


def test_unit_ranking_receives_parent_context_and_retains_each_parent():
    from assistant.evidence_selection import units
    class Rank:
        inputs=[]
        def scores(self,q,texts):
            self.inputs.extend(texts)
            return [100. if 'Parent A' in t else 1. for t in texts],[len(t.split()) for t in texts]
    first={**SOURCE,'title':'Parent A','text':'One retrieval method is sparse. Another retrieval method is dense. A third method reranks documents.'}
    second={**SOURCE,'label':'S2','title':'Parent B','chunk_id':'chunk-b','page_number':2,
        'text':'Pair classification uses average precision. This is its main metric.'}
    rank=Rank();candidates=units('Which metric?',[first,second],rank,limit=4)
    assert {c['source_id'] for c in candidates}=={'S1','S2'}
    assert all('Context:' in text and 'Excerpt:' in text for text in rank.inputs)


def test_source_span_overlap_deduplication_keeps_one_copy():
    from assistant.evidence_selection import span_overlap
    first={'paper_id':'a','page_number':1,'start_char':10,'end_char':100}
    second={'paper_id':'a','page_number':1,'start_char':20,'end_char':95}
    assert span_overlap(first,second)==1.
    assert span_overlap(first,{**second,'page_number':2})==0.


def test_focused_retry_can_recover_from_an_overcautious_selector(prepared):
    service,papers,encoder,reranker,generator=prepared
    service.ingest(papers)
    service.config=replace(service.config,grounding_enabled=True)
    service._grounder=GroundingVerifier(service.config,model=NLI())
    calls=[]
    def select(question,candidates,required):
        calls.append(len(candidates))
        selected={'status':'insufficient_evidence','evidence_ids':[]} if len(calls)==1 else {
            'status':'answered','evidence_ids':[candidates[0]['evidence_id']]}
        return {'message':{'content':json.dumps(selected)}}
    generator.select_evidence=select
    result=service.ask('What retrieval method is described?')
    assert result['status']=='answered' and len(calls)==2 and calls[1]==1
