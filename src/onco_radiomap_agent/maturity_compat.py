from copy import deepcopy

from . import maturity as legacy

VERSION='V1.3.2-arbiter-compat-20260920'

FIELD='predicted_maturity_code'

def required_claim_case_ids(predictions):
    """Schedule after arbiter merge, including newly valid pairs."""
    return {r['case_id'] for r in predictions if r['case_type']=='pair_snapshot' and r['prediction_status']=='valid'}

def candidate(row,case,facts,tumors,*,ownership=True):
    if row['case_type']!='pair_snapshot' or row['prediction_status']!='valid':
        return legacy.candidate(row,case,facts,tumors,ownership=ownership)
    proposal=legacy.rules.predict_snapshot(facts,allowed_fact_ids={f['fact_id'] for f in facts},
        **{k:case[k] for k in ['dataset_id','tumor_subtype_id','radiopharmaceutical_id','clinical_use','population_scope','cutoff_date']})
    if proposal[FIELD]==row[FIELD]:
        return legacy.candidate(row,case,facts,tumors,ownership=ownership)
    p=row.get('prediction_provenance',{})
    def require(ok,reason):
        if not ok:raise ValueError('Unaudited maturity mismatch: '+reason)
    require(row.get('method_id')=='full_system','method')
    require(row.get('clinical_use')==case['clinical_use'],'query clinical use')
    allowed={('D' if case['clinical_use']=='diagnostic' else 'T')+str(i) for i in range(1,5)}
    require(row[FIELD] in allowed,'clinical-use grade')
    require(p.get('decision_source')=='audited_llm_override' and p.get('llm_called') is True,'decision origin')
    require(p.get('override_applied') is True and p.get('override_audit_status')=='override_audit_pass','audit pass')
    require(p.get('override_contract_metadata_complete') is True,'complete audit metadata')
    require(FIELD in p.get('allowed_override_fields',[]) and FIELD in p.get('overridden_fields',[]),'allowed field')
    expected=(p.get('override_reason_contract') or {}).get('classification_override_reason_code')
    require(bool(expected) and p.get('override_reason_code')==expected,'reason code')
    changes=(p.get('override_changes') or {}).get(FIELD,{})
    require('rule_value' in changes and changes['rule_value']==proposal[FIELD] and changes.get('override_value')==row[FIELD],'grade transition')
    require(bool(p.get('override_source_ids')) and bool(p.get('override_record_locators') or p.get('override_evidence_quotes')),'source trace')
    require(not p.get('locked_field_attempts'),'locked-field change')
    out=deepcopy(row)
    out['prediction_provenance']['maturity_C2']=dict(version=VERSION,arm='audited_arbiter_passthrough',
        previous_grade=row[FIELD],grade=row[FIELD],rule_baseline_grade=proposal[FIELD],
        rule_adapter_applied=False,reason='C2 repairs rule-derived grades, not independently audited arbiter grades',
        source_traces=[],study_families=[],legacy_floor_without_target_evidence=None,
        override_source_ids=p['override_source_ids'],status_policy_unchanged=True,
        secondary_source_and_independence_policy_unchanged=True,api_calls=0,expert_answers_accessed=False)
    return out
