from copy import deepcopy
from unittest.mock import patch
import pytest
from onco_radiomap_agent import maturity_compat as m

def fixture(use='diagnostic',baseline=None):
    grade=('D' if use=='diagnostic' else 'T')+'2'
    case=dict(dataset_id='synthetic',tumor_subtype_id='tumor',radiopharmaceutical_id='drug',clinical_use=use,population_scope='human',cutoff_date='2026-01-01')
    p=dict(decision_source='audited_llm_override',llm_called=True,override_applied=True,override_audit_status='override_audit_pass',override_contract_metadata_complete=True,
        allowed_override_fields=[m.FIELD],overridden_fields=[m.FIELD],override_reason_contract={'classification_override_reason_code':'missing_field_resolved'},override_reason_code='missing_field_resolved',
        override_changes={m.FIELD:dict(rule_value=baseline,override_value=grade)},override_source_ids=['S1'],override_record_locators=['source#results'],locked_field_attempts=[])
    row=dict(case_id='synthetic',case_type='pair_snapshot',method_id='full_system',clinical_use=use,prediction_status='valid',predicted_maturity_code=grade,predicted_evidence_conclusion='inconclusive',prediction_provenance=p)
    return row,case

@pytest.mark.parametrize('use',['diagnostic','therapeutic'])
@pytest.mark.parametrize('baseline',[None,'D1'])
def test_audited_grade_preserved_and_no_mutation(use,baseline):
    if baseline:baseline=('D' if use=='diagnostic' else 'T')+'1'
    row,case=fixture(use,baseline);before=deepcopy(row)
    with patch.object(m.legacy.rules,'predict_snapshot',return_value={m.FIELD:baseline}),patch.object(m.legacy,'candidate',side_effect=AssertionError('must not call rule adapter')):
        out=m.candidate(row,case,[],None);assert out==m.candidate(row,case,[],None)
    assert row==before and out[m.FIELD]==row[m.FIELD]
    assert {k:v for k,v in out.items() if k!='prediction_provenance'}=={k:v for k,v in row.items() if k!='prediction_provenance'}
    assert out['prediction_provenance']['maturity_C2']['rule_adapter_applied'] is False

@pytest.mark.parametrize('field,value',[('decision_source','rule'),('llm_called',False),('override_applied',False),('override_audit_status','rejected'),('override_contract_metadata_complete',False),('allowed_override_fields',[]),('overridden_fields',[]),('override_reason_code','other'),('override_changes',{}),('override_source_ids',[]),('override_record_locators',[]),('locked_field_attempts',['x'])])
def test_incomplete_audit_still_fails(field,value):
    row,case=fixture();row['prediction_provenance'][field]=value
    with patch.object(m.legacy.rules,'predict_snapshot',return_value={m.FIELD:None}),pytest.raises(ValueError,match='Unaudited maturity mismatch'):
        m.candidate(row,case,[],None)

def test_prefix_mismatch_rejected():
    row,case=fixture();row[m.FIELD]='T2'
    with patch.object(m.legacy.rules,'predict_snapshot',return_value={m.FIELD:None}),pytest.raises(ValueError,match='clinical-use grade'):
        m.candidate(row,case,[],None)

def test_normal_rule_path_exactly_delegates():
    row,case=fixture();row['prediction_provenance']={};sentinel={'unchanged':'legacy output'}
    with patch.object(m.legacy.rules,'predict_snapshot',return_value={m.FIELD:'D2'}),patch.object(m.legacy,'candidate',return_value=sentinel) as fn:
        assert m.candidate(row,case,[],None) is sentinel;fn.assert_called_once()

@pytest.mark.parametrize('status',['abstain','invalid'])
def test_nonvalid_unchanged(status):
    row,case=fixture();row['prediction_status']=status;row[m.FIELD]=None
    with patch.object(m.legacy.rules,'predict_snapshot',side_effect=AssertionError('should not evaluate')):
        assert m.candidate(row,case,[],None)==row

def test_claims_scheduled_from_merged_valid_not_rule_state():
    row,_=fixture();assert m.required_claim_case_ids([row])=={'synthetic'}
    row['prediction_status']='abstain';assert not m.required_claim_case_ids([row])
