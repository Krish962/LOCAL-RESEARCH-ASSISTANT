import pytest
from evaluation.metrics import abstention_counts,citation_ids,evidence_metrics,graded_metrics


def test_hand_checked_metric_values_and_unjudged_cases():
    hits = [{"chunk_id":"a","paper_id":"p","page_number":1,"text":"irrelevant"},
            {"chunk_id":"b","paper_id":"p","page_number":2,"text":"the verified fact"}]
    groups = [{"alternatives":[{"paper_id":"p","page_number":2,"quote":"verified fact"}]}]
    measured = evidence_metrics(hits,groups,4)
    assert measured["evidence_recall"] == 1 and measured["mrr"] == .5
    graded = graded_metrics(hits,{"a":0,"b":2},4)
    assert graded["precision"] == .25
    assert graded["ndcg"] == pytest.approx(1/1.584962500721156)
    assert graded_metrics(hits,{"a":0},4)["ndcg"] is None
    assert evidence_metrics(hits,[],4)["evidence_recall"] is None


def test_runtime_error_is_not_successful_abstention():
    assert abstention_counts(False,"error")["correct_refusal"] == 0
    assert abstention_counts(False,"error")["runtime_failure"] == 1
    assert abstention_counts(False,"no_match")["correct_refusal"] == 1
    assert citation_ids("fact [S1] and [S9]",[{"label":"S1"}])["validity"] == .5
    assert abstention_counts(False,"answered")["unsupported_answer_risk"] == 1
    assert abstention_counts(True,"answered")["unsupported_answer_risk"] == 0
