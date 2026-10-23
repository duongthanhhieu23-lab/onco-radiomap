import sys
from pathlib import Path
from copy import deepcopy
import pytest


from onco_radiomap_agent.maturity import design_repair, segments, rules, ownership_conflicts, Tumors, candidate,current_study_statement


def fact(text,use='therapeutic'):
    return dict(raw_source_text=text,clinical_use=use,design=dict(safety_reported=True),source_id='source',
        fact_id='fact',study_family_id='family',origin_dataset_id='test',
        relation_decision_status='clear_eligible',source_role='primary_human_result',human_subjects=True,
        oncology_context=True,result_reported=True,clinical_use_alignment=True,retracted=False,
        tumor_subtype_id='tumor',radiopharmaceutical_id='radio',applicability={'population_scope':'adult'},
        evidence_public_date='2020-01-01',record_locator='local source',result={})


@pytest.mark.parametrize('word',['non-randomized','non-randomised','not randomized','nonrandomized'])
def test_nonrandom_does_not_upgrade(word):
    f,trace=design_repair(fact(f'Title: A phase I study. Abstract: METHODS: This was a {word} dose-escalation study. RESULTS: Objective response rate was 20%.'))
    assert not f['design']['randomized'] and not f['design']['credible_comparator']
    assert rules._maturity(f)=='T2'


@pytest.mark.parametrize('text',[
    'Prospective randomized studies are needed.',
    'Future randomized studies are warranted.',
    'We will conduct a randomized controlled phase III study.',
    'This study was performed prior to expansion into a randomized phase.',
])
def test_future(text):
    f,_=design_repair(fact('Title: Pilot study. Abstract: CONCLUSIONS: '+text))
    assert not f['design']['randomized'] and not f['design']['controlled'] and not f['design']['prospective']


def test_current_random_allocation():
    f,_=design_repair(fact('Title: Phase III study. Abstract: METHODS: We randomly assigned patients to treatment A or standard care alone. RESULTS: Overall survival improved.'))
    assert f['design']['randomized'] and f['design']['controlled']
    assert rules._maturity(f)=='T4'


def test_leadin_not_parent():
    f,_=design_repair(fact('Title: Efficacy from the lead-in cohort. Abstract: INTRODUCTION: This is a phase III randomized trial. METHODS: Participants received four cycles. Objective response rate was assessed. RESULTS: Twenty-seven participants completed treatment.'))
    assert not f['design']['randomized'] and not f['design']['phase_iii']
    assert f['design']['single_arm_defined_cohort'] and rules._maturity(f)=='T2'


def test_actual_random_not_blocked_by_future_conclusion():
    f,_=design_repair(fact('Title: Phase III study. Abstract: METHODS: We randomly assigned participants to therapy A or placebo. RESULTS: Overall survival improved. CONCLUSION: Future randomized studies are needed.'))
    assert rules._maturity(f)=='T4'


@pytest.mark.parametrize('method',[
    'PSA was assessed prior to each cycle and serial CT scans were performed.',
    'Response to therapy was evaluated according to PSA response and imaging response.',
])
def test_response_protocol(method):
    f,t=design_repair(fact('Title: Treatment study. Abstract: METHODS: Forty consecutive patients were enrolled. '+method+' RESULTS: Responses were observed.'))
    assert f['design']['single_arm_defined_cohort'] and f['design']['prespecified_objective_efficacy']
    assert rules._maturity(f)=='T2'
    assert t['fields']['prespecified_objective_efficacy']['basis']=='method_defined_response_assessment'


def test_psa_number_not_predefined_efficacy():
    f,_=design_repair(fact('Title: Phase I study. Abstract: RESULTS: PSA decreased by 50% in one patient.'))
    assert not f['design']['prespecified_objective_efficacy']
    assert rules._maturity(f)=='T1'


@pytest.mark.parametrize('phrase',['changed the clinical management','change in intended management'])
def test_diagnostic_management(phrase):
    f,_=design_repair(fact('Title: Prospective multicenter study. Abstract: RESULTS: Imaging '+phrase+' in 40% of patients.','diagnostic'))
    assert f['design']['management_impact_evaluated'] and rules._maturity(f)=='D4'


def test_histopathology_and_clr():
    f,_=design_repair(fact('Title: Prospective multicenter study. Abstract: METHODS: Histopathology was the standard of truth. The endpoint was correct localization rate.','diagnostic'))
    assert f['design']['gold_standard_comparison'] and f['design']['prespecified_diagnostic_performance']
    assert rules._maturity(f)=='D3'


def test_unknown_vs_explicit_false():
    a,t=design_repair(fact('Title: Safety study.'))
    b,u=design_repair(fact('Title: Non-randomized study.'))
    assert a['design']['randomized'] is None and b['design']['randomized'] is False
    assert u['fields']['randomized']['negative']


def test_not_prospective():
    a,_=design_repair(fact('Title: A cohort study. Abstract: METHODS: This was not prospective.'))
    assert a['design']['prospective'] is False


def test_same_target_combination_is_not_wrong_use():
    case={'clinical_use':'therapeutic','radiopharmaceutical_name':'177Lu-J591','tumor_subtype_id':'tumor','tumor_subtype_name':'Prostate cancer'}
    pair='Patients with prostate cancer received 177Lu-J591 plus docetaxel.'
    result='177Lu-J591 plus docetaxel showed PSA responses in the patients.'
    claim=dict(source_id='source',pair_quote=pair,result_quote=result,clinical_setting='Prostate cancer',regimen='177Lu-J591 plus docetaxel',directness='other_use')
    tumors=Tumors([dict(tumor_subtype_id='tumor',preferred_name='Prostate cancer')])
    assert not ownership_conflicts(case,fact(pair+' '+result),{'rejected_claims':[{'claim':claim}]},tumors)


def test_quotes_exact_with_offsets():
    text='Title: Prospective study.\nAbstract: METHODS: We randomly assigned patients to A or B.\nRESULTS: Overall survival was 9.4 months.'
    _,t=design_repair(fact(text))
    for field in t['fields'].values():
        for span in field['positive']+field['negative']:
            assert text[span['start']:span['end']]==span['quote']


@pytest.mark.parametrize('target,actual',[('68Ga-PSMA-11','177Lu-PSMA-617'),('18F-tracer-X','90Y-drug-Y')])
def test_wrong_isotope_use(target,actual):
    case={'clinical_use':'therapeutic','radiopharmaceutical_name':target,'tumor_subtype_id':'tumor','tumor_subtype_name':'Prostate cancer'}
    pair=f'Patients with prostate cancer underwent {target} PET screening.'
    result=f'{actual} improved overall survival in the treatment arm.'
    claim=dict(source_id='source',pair_quote=pair,result_quote=result,clinical_setting='Prostate cancer',regimen=actual,directness='other_use')
    tumors=Tumors([dict(tumor_subtype_id='tumor',preferred_name='Prostate cancer')])
    f=fact(pair+' '+result)
    assert ownership_conflicts(case,f,{'rejected_claims':[{'claim':claim}]},tumors)
    assert not ownership_conflicts(case,f,{},tumors) # no claim != negative evidence
    assert not ownership_conflicts(case,f,{'claims':[claim],'rejected_claims':[{'claim':claim}]},tumors)
    bad=deepcopy(claim);bad['result_quote']='Invented non-verbatim result '+actual
    assert not ownership_conflicts(case,f,{'rejected_claims':[{'claim':bad}]},tumors)


def test_other_tumor_scope():
    case={'clinical_use':'therapeutic','radiopharmaceutical_name':'177Lu-drug','tumor_subtype_id':'prostate','tumor_subtype_name':'Prostate cancer'}
    text='Title: Treatment for breast cancer. Abstract: METHODS: Patients with breast cancer received treatment. RESULTS: Breast cancer tumor response was observed.'
    claim=dict(source_id='source',pair_quote='Patients with breast cancer received treatment.',result_quote='Breast cancer tumor response was observed.',clinical_setting='Breast cancer',regimen='177Lu-drug',directness='exact_pair')
    tumors=Tumors([dict(tumor_subtype_id='prostate',preferred_name='Prostate cancer'),dict(tumor_subtype_id='breast',preferred_name='Breast cancer')])
    assert ownership_conflicts(case,fact(text),{'rejected_claims':[{'claim':claim}]},tumors)


def test_unchanged_atomic_abstention():
    for row in [dict(case_type='atomic_source_record',prediction_status='valid'),dict(case_type='pair_snapshot',prediction_status='abstain')]:
        assert candidate(row,{},[],None)==row


def test_no_specific_case_or_source_patches():
    code=(Path(__file__).resolve().parents[1]/'src/onco_radiomap_agent/maturity.py').read_text(encoding='utf8')
    assert 'ER-0' not in code and 'PMID:' not in code and 'VISION' not in code


@pytest.mark.parametrize('trial',['ORBIT','DELTA','STUDY-Z'])
def test_current_design_in_purpose(trial):
    text=f'Title: Results from the {trial} multicenter study. Abstract: PURPOSE: {trial} was a prospective study designed to assess tracer X. METHODS: Management impact was evaluated. RESULTS: Management changed in 30%.'
    f,t=design_repair(fact(text,'diagnostic'))
    assert f['design']['prospective'] and rules._maturity(f)=='D4'
    assert any(s['section']=='PURPOSE' for s in t['fields']['prospective']['positive'])


@pytest.mark.parametrize('statement',[
    'This study was prospective.',
    'Our trial was a prospective investigation.',
    'We conducted a prospective multicenter study.',
])
def test_current_generic_background(statement):
    f,_=design_repair(fact('Title: A clinical investigation. Abstract: BACKGROUND: '+statement+' METHODS: PSA response was assessed.'))
    assert f['design']['prospective']


@pytest.mark.parametrize('statement',[
    'Previous studies were prospective.',
    'This study will be prospective.',
    'Our study was a review of prospective trials.',
    'OTHERTRIAL was a prospective study designed to assess tracer Y.',
])
def test_do_not_import_other_design(statement):
    f,_=design_repair(fact('Title: Pilot investigation of tracer X. Abstract: PURPOSE: '+statement))
    assert not f['design']['prospective']


def test_prospectively_evaluated_word_form():
    f,_=design_repair(fact('Title: A therapy study. Abstract: METHODS: Patients were prospectively evaluated. Objective response rate was measured.'))
    assert f['design']['prospective'] and rules._maturity(f)=='T2'


def test_defined_entry_regimen_cohort_and_response():
    text='Title: Study of tracer X. Abstract: METHODS: Patients with prostate cancer and adequate marrow function were included. Prescribed activity of therapy was 6 GBq administered in 8-week intervals. Response was assessed by PSA and week-12 PET/CT. RESULTS: One hundred patients completed one cycle of treatment.'
    f,t=design_repair(fact(text))
    assert f['design']['single_arm_defined_cohort'] and f['design']['prespecified_objective_efficacy']
    assert rules._maturity(f)=='T2'
    assert t['fields']['single_arm_defined_cohort']['basis']=='explicit_entry_common_regimen_observed_cohort'


def test_cohort_requires_actual_results():
    text='Title: Treatment protocol. Abstract: METHODS: Eligible patients will be included. Therapy will be administered at 8-week intervals. Response will be assessed by PSA.'
    f,_=design_repair(fact(text))
    assert not f['design']['single_arm_defined_cohort'] and not f['design']['prespecified_objective_efficacy']


def test_immune_response_not_objective_tumor_endpoint():
    f,_=design_repair(fact('Title: Phase I study. Abstract: METHODS: Immune response was assessed by antibody binding.'))
    assert not f['design']['prespecified_objective_efficacy']


def test_named_trial_not_same_title():
    assert not current_study_statement('ORBIT was a prospective study designed to assess tracer A.','A DELTA study of tracer B')


def test_purpose_source_spans_stay_verbatim():
    text='Title: An ORBIT multicenter study. Abstract: PURPOSE: ORBIT was a prospective study designed to evaluate imaging. METHODS: Histopathology was used.'
    _,t=design_repair(fact(text,'diagnostic'))
    for field in t['fields'].values():
        for s in field['positive']+field['negative']:assert text[s['start']:s['end']]==s['quote']
