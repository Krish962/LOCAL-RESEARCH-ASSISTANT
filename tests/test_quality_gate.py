from evaluation.check_grounding_quality import assess


def rows(positive_statuses,negative_statuses):
    return ([{'answerable':True,'status':s} for s in positive_statuses]+
            [{'answerable':False,'status':s} for s in negative_statuses])


def test_runtime_success_with_low_answer_coverage_does_not_pass():
    result=assess(rows(['answered']+['insufficient_evidence']*3,['no_match','insufficient_evidence']))
    assert result['runtime_failures']==0 and not result['passed'] and result['coverage']==.25


def test_negative_answer_and_runtime_failure_cannot_be_hidden_by_high_coverage():
    assert not assess(rows(['answered']*4,['answered']))['passed']
    assert not assess(rows(['answered']*4,['error']))['passed']


def test_acceptance_requires_coverage_and_correct_negative_controls():
    assert assess(rows(['answered']*3+['insufficient_evidence'],['no_match','insufficient_evidence']))['passed']
