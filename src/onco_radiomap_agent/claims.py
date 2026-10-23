from __future__ import annotations

import json

from copy import deepcopy

from . import conclusion_alignment as ca

MODEL='deepseek-v4-flash'

FIELDS=('case_id','dataset_id','case_type','clinical_use','method_id','run_id','prediction_status','predicted_tumor_subtype_id','predicted_radiopharmaceutical_id','entity_resolution_status','graph_insertable','predicted_relation_eligible','predicted_atomic_conclusion','predicted_maturity_code')

def assert_no_answers(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k != 'gold_standard_comparison' and k.startswith(('reference_','gold_','expert_','final_')):
                raise ValueError('Answer field in predictor: '+k)
            assert_no_answers(v)
    elif isinstance(value,list):
        for v in value: assert_no_answers(v)

def conclusion_request(case,context,aliases):
    sources=ca.visible_sources(context)
    payload=ca.make_payload(case,sources,aliases)
    assert_no_answers(payload)
    prompt=json.dumps(payload,ensure_ascii=False,sort_keys=True)
    identity=dict(version=ca.VERSION,prompt=prompt,system=ca.SYSTEM,model=MODEL,thinking='disabled',temperature=0.0,max_tokens=ca.MAX_TOKENS)
    return identity,sources

def apply_conclusion_only(row,raw,case,facts,sources,key):
    before=deepcopy(row)
    result=ca.apply_claims(row,raw,case,facts,sources,key)
    for field in FIELDS: assert result.get(field)==before.get(field),'Downstream changed upstream field: '+field
    assert row==before,'Input mutation'
    result['prediction_provenance']['component_experiment']='V1.3.1_C1_conclusion_only_on_reproduced_V1.2.1'
    return result
