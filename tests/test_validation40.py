import copy
import json
import pytest
from assistant.artifacts import read_jsonl
from assistant.config import PROJECT_ROOT,fingerprint
from evaluation.build_validation40 import DESTINATION,load_corpus,validate
from evaluation.summarize_grounding import comparison_paper_recall
from assistant.evidence_selection import render_selection


@pytest.fixture(scope='module')
def data():
    return read_jsonl(DESTINATION),json.loads(DESTINATION.with_suffix('.manifest.json').read_text(encoding='utf-8')),load_corpus()


def test_frozen_40_cases_cover_all_20_papers_with_real_evidence(data):
    questions,manifest,(corpus,pages,chunks)=data
    result=validate(questions,manifest,corpus,pages,chunks)
    assert result['questions']==40 and result['papers']==20 and result['answerable']==28
    assert len([q for q in questions if q['type']=='comparison'])==8


def test_freeze_hash_detects_unreviewed_question_changes(data):
    questions,manifest,corpus=data
    changed=copy.deepcopy(questions);changed[0]['question']='An unreviewed replacement question'
    with pytest.raises(ValueError,match='Frozen dataset'):
        validate(changed,manifest,*corpus)


def test_source_validation_rejects_fabricated_quote_even_with_updated_dataset_hash(data):
    questions,manifest,corpus=data
    changed=copy.deepcopy(questions)
    changed[0]['evidence_groups'][0]['alternatives'][0]['quote']='A fabricated quotation.'
    metadata={**manifest,'questions_hash':fingerprint(changed)}
    with pytest.raises(ValueError,match='PDF span'):
        validate(changed,metadata,*corpus)


def test_positive_year_filter_uses_ragtruth_submission_metadata(data):
    questions,manifest,corpus=data
    ragtruth=next(q for q in questions if q['question_id']=='validation40-single-ragtruth')
    assert ragtruth['filters']['year_min']==2023 and ragtruth['filters']['year_max']==2023


def test_comparison_coverage_counts_citations_not_every_retrieved_paper():
    row={'status':'answered','answer':'A fact [S1]','sources':[
        {'label':'S1','paper_id':'a'},{'label':'S2','paper_id':'b'}]}
    assert comparison_paper_recall(row,{'required_papers':['a','b']})==.5
    assert comparison_paper_recall({**row,'status':'insufficient_evidence'},{'required_papers':['a','b']})==0.


def test_splade_currency_control_records_the_publication_fee_as_non_answer(data):
    questions,manifest,corpus=data
    question=next(q for q in questions if q['question_id']=='validation40-missing-splade-dollar-cost')
    mentions=question['absence_audit']['related_non_answer_mentions']
    assert mentions and all('ACM ISBN' in m['quote'] for m in mentions)


def test_inconsistent_refusal_with_evidence_is_never_silently_accepted():
    with pytest.raises(ValueError,match='Invalid evidence selection'):
        render_selection({'status':'insufficient_evidence','evidence_ids':['E1']},[],[])
