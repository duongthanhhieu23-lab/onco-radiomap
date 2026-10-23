import copy
import json
import sys
from pathlib import Path
import pytest
from onco_radiomap_agent.ownership_base import *

TUMORS = Tumors([
    dict(tumor_subtype_id='a', preferred_name='Alpha Carcinoma', aliases='["Alpha cancer"]'),
    dict(tumor_subtype_id='b', preferred_name='Beta Carcinoma', aliases='["Beta cancer"]'),
    dict(tumor_subtype_id='ac', preferred_name='Alpha Subtype Carcinoma', aliases='[]', parent_tumor_subtype_id='a')])


def example(**extra):
    q = dict(source_id='s', level='therapeutic_tumor_control', direction='supportive',
             directness='exact_pair', result_basis='observed_target_result',
             pair_quote='Patients with Alpha cancer received 177Lu-X.',
             result_quote='Tumor volume decreased 25% from baseline after treatment.',
             endpoint_family='tumor_control', regimen='177Lu-X', clinical_setting='patients with Alpha cancer')
    q.update(extra)
    c = dict(dataset_id='d', clinical_use='therapeutic', tumor_subtype_id='a',
             tumor_subtype_name='Alpha Carcinoma', radiopharmaceutical_id='r', radiopharmaceutical_name='177Lu-X')
    s = 'METHODS: '+q['pair_quote']+'\nRESULTS: '+q['result_quote']
    f = dict(source_id='s', origin_dataset_id='d', clinical_use='therapeutic',
             tumor_subtype_id='a', radiopharmaceutical_id='r', raw_source_text=s,
             atomic_review=dict(audited=True, relation_eligible=True), study_family_id='family')
    row = dict(prediction_status='valid', predicted_maturity_code='T3',
               predicted_evidence_conclusion='supportive', prediction_provenance={})
    return q, c, s, f, row


@pytest.mark.parametrize('original,source', [('P< 0.01','P < 0.01'), ('P >0.05','P > 0.05'), ('n=18','n = 18')])
def test_operator_spaces(original, source):
    assert recover_span(original, source) == source


@pytest.mark.parametrize('original,source', [('P<0.01','P>0.01'), ('P<0.01','P<0.02'),
    ('effective','not effective at all') ,('therapist','the rapist')])
def test_quote_not_fuzzy(original, source):
    # Plain substring matching alone cannot reject removed surrounding negation;
    # this test only promises no altered digits/operators or internal spaces.
    if original == 'effective':
        assert recover_span('was effective', 'was not effective') is None
    else:
        assert recover_span(original, source) is None


@pytest.mark.parametrize('setting', ['murine xenograft models', 'mouse models', 'in vitro cell lines'])
def test_explicit_nonhuman(setting):
    q,c,s,_,_=example(clinical_setting=setting)
    assert ownership_reason(q,c,s,TUMORS) == 'explicit_nonhuman_setting'


def test_animal_result_rejected():
    q,c,s,_,_=example(result_quote='Treatment prolonged survival in mouse models.')
    assert ownership_reason(q,c,s,TUMORS) == 'explicit_nonhuman_result'


def test_human_result_in_mixed_paper_kept():
    q,c,s,_,_=example(result_quote='In patients, tumor volume decreased 25% from baseline.')
    assert ownership_reason(q,c,'BACKGROUND: Mouse experiments.\n'+s,TUMORS) is None


@pytest.mark.parametrize('a,b', [('a','b'),('b','a')])
def test_other_tumor_and_name_swap(a,b):
    other='Alpha' if b=='a' else 'Beta'
    q,c,s,_,_=example(clinical_setting=f'patients with {other} cancer',pair_quote=f'Patients with {other} cancer received 177Lu-X.')
    c['tumor_subtype_id']=a
    assert ownership_reason(q,c,s,TUMORS) == 'explicit_other_tumor_population'


def test_parent_subtype_not_other_population():
    q,c,s,_,_=example(clinical_setting='Alpha Subtype Carcinoma',pair_quote='Alpha Subtype Carcinoma patients received 177Lu-X.')
    assert ownership_reason(q,c,s,TUMORS) is None


def test_histology_conflict():
    q,c,s,_,_=example(clinical_setting='urothelial carcinoma',pair_quote='Patients with urothelial carcinoma received tracer X.')
    c['tumor_subtype_name']='Clear Cell Renal Cell Carcinoma'
    assert ownership_reason(q,c,s,TUMORS) == 'explicit_incompatible_histology'


@pytest.mark.parametrize('specific', [False, True])
def test_mixed_cohort_requires_target_result(specific):
    q,c,s,_,_=example(clinical_setting='mixed cancers including Alpha cancer',
        result_quote=('Alpha cancer: ' if specific else '')+'Tumor response was observed in 5 patients.')
    assert (ownership_reason(q,c,s,TUMORS) is None) == specific


@pytest.mark.parametrize('number,other', [(131,124),(177,68)])
def test_diagnostic_isotope_role(number,other):
    q,c,s,_,_=example(regimen=f'pretherapy [{other}I]X PET before [{number}I]X therapy')
    c.update(clinical_use='diagnostic', radiopharmaceutical_name=f'{number}I-X')
    assert ownership_reason(q,c,s,TUMORS) == 'target_is_therapy_not_observed_diagnostic_tracer'


def test_same_tracer_before_therapy_kept():
    q,c,s,_,_=example(regimen='131I-X scan before 131I-X therapy')
    c.update(clinical_use='diagnostic', radiopharmaceutical_name='131I-X')
    assert ownership_reason(q,c,s,TUMORS) is None


def test_ancillary_intervention_not_drug_response():
    q,c,s,_,_=example(regimen='177Lu-X followed by rehabilitation',result_quote='The rehabilitation group showed improved swallowing.')
    assert ownership_reason(q,c,s,TUMORS) == 'outcome_of_subsequent_ancillary_intervention'


def test_real_combination_not_blanket_rejected():
    q,c,s,_,_=example(regimen='177Lu-X plus radiosensitizer',result_quote='Tumor response was improved after combination treatment.')
    assert ownership_reason(q,c,s,TUMORS) is None


@pytest.mark.parametrize('result,repair', [
    ('The transversal tumor area decreased 25% between baseline and follow-up.', True),
    ('Tumor volume did not decrease 25% between baseline and follow-up.', False),
    ('Tumor attenuation decreased 25% between baseline and follow-up.', False),
    ('Tumor volume decreased 25% and correlated with PFS at follow-up.', False),
    ('PFS was 12 months after treatment.', False)])
def test_tumor_control_not_biomarker(result,repair):
    q,c,s,f,_=example(level='imaging_biomarker_association',endpoint_family='biomarker_association',result_quote=result)
    got,edits=repair_claim(q,c,[f],{'s':s})
    assert (got['level']=='therapeutic_tumor_control') == repair


def test_negative_efficacy_not_hidden_by_feasibility():
    q,c,s,f,row=example(direction='negative',result_basis='explicit_objective_failed',result_quote='Treatment failed its prespecified efficacy objective.')
    got=apply(row,{'claims':[q]},c,[f],{'s':s},'key',TUMORS)
    assert got['predicted_evidence_conclusion']=='negative'


def test_comparator_negative_kept():
    q,c,s,f,row=example(result_quote='Sensitivity was lower than comparator X.', direction='negative')
    got=apply(row,{'claims':[q]},c,[f],{'s':s},'key',TUMORS)
    assert got['predicted_evidence_conclusion']=='negative'
    assert got['prediction_provenance']['conclusion_alignment_C1_1']['claims'][0]['endpoint_scope']['reported_metrics']==['sensitivity']


def test_protected_fields_no_input_mutation():
    q,c,s,f,row=example()
    before=copy.deepcopy([q,c,f,row])
    got=apply(row,{'claims':[q]},c,[f],{'s':s},'key',TUMORS)
    assert [q,c,f,row]==before
    assert got['predicted_maturity_code']=='T3' and got['prediction_status']=='valid'


def test_abstention_unchanged():
    q,c,s,f,row=example()
    row['prediction_status']='abstain'
    assert apply(row,{'claims':[q]},c,[f],{'s':s},'key',TUMORS)==row


def test_predictor_has_no_case_override():
    code=Path(__file__).resolve().parents[1].joinpath('src/onco_radiomap_agent/ownership_base.py').read_text(encoding='utf8')
    for forbidden in ['ER-', 'PMID:', 'final_consensus', 'dossiers_90']:
        assert forbidden not in code


def test_answer_fields_rejected():
    q,c,s,f,row=example()
    c['reference_evidence_conclusion']='supportive'
    with pytest.raises(ValueError): apply(row,{'claims':[q]},c,[f],{'s':s},'key',TUMORS)
