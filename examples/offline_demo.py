"""Synthetic no-evidence example: no network, no clinical claim, no data files."""
import json
from onco_radiomap_agent.engine import run

case = dict(case_id='synthetic-pair', dataset_id='synthetic-example',
            case_type='pair_snapshot', clinical_use='therapeutic',
            tumor_subtype_id='synthetic-tumor', tumor_subtype_name='Synthetic Tumor',
            radiopharmaceutical_id='synthetic-agent', radiopharmaceutical_name='Synthetic Agent',
            population_scope='adult', cutoff_date='2022-12-31')

def no_evidence_response(job):
    raw = dict(predictions=[dict(i=0,prediction_status='abstain',maturity_code=None,evidence_conclusion=None)])
    return dict(identity=job['identity'],run_id=job['run_id'],finish_reason='stop',
                provider_model='synthetic-offline-responder',response=json.dumps(raw))

contexts = {m: {case['case_id']: ''} for m in ['naive_rag','lightrag']}
predictions = run([case],[],contexts,[],{},no_evidence_response)
assert len(predictions)==4 and all(p['prediction_status']=='abstain' for p in predictions)
print(json.dumps([dict(method=p['method_id'],status=p['prediction_status']) for p in predictions],indent=2))
