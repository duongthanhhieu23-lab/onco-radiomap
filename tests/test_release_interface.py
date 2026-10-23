"""Protect the portable interface's evidence and repetition boundaries."""
import json
import pytest
from onco_radiomap_agent.engine import read_receipt, validate_inputs


def test_legacy_receipt_cannot_cross_repetitions():
    receipt = dict(identity={}, finish_reason='stop', response=json.dumps({'claims': []}))
    job = dict(identity={},run_id=2,stage='scoped_claims')
    with pytest.raises(ValueError,match='another repetition'):
        read_receipt(job, lambda _: receipt)
    job['run_id']=1
    assert read_receipt(job,lambda _:receipt)[1] == {'claims': []}


def test_incomplete_response_is_not_repaired():
    receipt = dict(identity={},finish_reason='length',response='{}')
    with pytest.raises(ValueError,match='incomplete response'):
        read_receipt(dict(identity={},run_id=1,stage='classification'),lambda _:receipt)


def test_reference_labels_cannot_enter_prediction():
    with pytest.raises(ValueError,match='Answer field'):
        validate_inputs([{'case_id':'x','dataset_id':'d','reference_label':'T4'}],[],{})


def test_cross_dataset_facts_are_rejected():
    with pytest.raises(ValueError,match='requested dataset'):
        validate_inputs([{'case_id':'x','dataset_id':'a'}],[{'origin_dataset_id':'b'}],{})
