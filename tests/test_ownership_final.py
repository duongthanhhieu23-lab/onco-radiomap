import sys
from pathlib import Path
import pytest

from onco_radiomap_agent import ownership as r3
from test_ownership_population import T
from test_ownership import example
T3=r3.Tumors(T.rows+[
    dict(tumor_subtype_id='br',preferred_name='Breast Carcinoma',aliases='["breast cancer"]'),
    dict(tumor_subtype_id='bl',preferred_name='Bladder Carcinoma',aliases='["bladder cancer"]'),
    dict(tumor_subtype_id='pan',preferred_name='Pancreatic Carcinoma',aliases='["pancreatic cancer"]')])


def test_coordinated_populations():
    assert T3.hits('prostate or breast cancer')=={'p','br'}
    assert T3.hits('breast, pancreatic, or bladder cancer')=={'br','pan','bl'}


@pytest.mark.parametrize('text,expected',[
    ('METHODS: We applied the method to specimens of patients who were preoperatively injected with tracer X.',True),
    ('METHODS: Patients were not injected with tracer X.',False),
    ('METHODS: Patients will be injected with tracer X.',False)])
def test_actual_human_administration(text,expected):
    assert r3.human_administration(text)==expected


def test_human_injection_then_specimen_imaging_is_not_nonhuman():
    q,c,s,_,_=example(clinical_setting='ex vivo surgical specimens')
    c['clinical_use']='diagnostic'
    s+='\nMETHODS: Patients were preoperatively injected with the tracer.'
    assert r3.extra_ownership_reason(q,c,s,T3) is None


def test_explicit_target_result_in_other_primary_cohort():
    q,c,s,_,_=example(clinical_setting='bladder cancer',result_quote='Prostate adenocarcinoma was observed in 12 patients; 8 had FDG uptake.')
    c['tumor_subtype_id']='pa'
    s='TITLE: A study of bladder cancer.\n'+s
    assert r3.extra_ownership_reason(q,c,s,T3) is None


def test_exvivo_therapy_calculation_still_rejected():
    q,c,s,_,_=example(clinical_setting='ex vivo patient tumor samples',result_quote='Dosimetry calculations on patient tumor samples showed high absorbed doses.')
    assert r3.extra_ownership_reason(q,c,s,T3)=='ex_vivo_human_specimen_not_clinical_intervention'
