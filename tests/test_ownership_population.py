import sys
from pathlib import Path
import pytest

from onco_radiomap_agent import ownership_population as r2
from test_ownership import example

T = r2.Tumors([
    dict(tumor_subtype_id='p',preferred_name='Prostate Carcinoma',aliases='["prostate cancer"]'),
    dict(tumor_subtype_id='pa',preferred_name='Prostate Adenocarcinoma',aliases='[]'),
    dict(tumor_subtype_id='t',preferred_name='Thyroid Gland Carcinoma',aliases='["thyroid cancer"]'),
    dict(tumor_subtype_id='tf',preferred_name='Thyroid Gland Follicular Carcinoma',aliases='[]'),
    dict(tumor_subtype_id='l',preferred_name='Hepatocellular Carcinoma',aliases='[]')])


@pytest.mark.parametrize('a,b',[('p','pa'),('t','tf')])
def test_missing_parent_not_incompatible(a,b):
    assert T.compatible(a,b)


def test_actual_distinct_organs():
    assert not T.compatible('p','t')


def test_nonprostate_not_prostate():
    assert not T.hits('advanced non-prostate cancer with high PSMA expression')
    assert T.hits('prostate cancer') == {'p'}


@pytest.mark.parametrize('result,setting',[
    ('Dosimetry calculations based on patient tumor samples showed higher dose.', 'ex vivo patient tumor samples'),
    ('Dosimetry calculations on human tissue samples showed high dose.', 'human tumor samples')])
def test_ex_vivo_is_not_human_intervention(result,setting):
    q,c,s,_,_=example(result_quote=result,clinical_setting=setting)
    assert r2.extra_ownership_reason(q,c,s,T)=='ex_vivo_human_specimen_not_clinical_intervention'


def test_actual_human_dosimetry_kept():
    q,c,s,_,_=example(result_quote='In patients after therapy, dosimetry showed favorable tumor absorbed doses.')
    assert r2.extra_ownership_reason(q,c,s,T) is None


def test_cohort_scope_omitted_from_result_quote():
    q,c,s,_,_=example(clinical_setting='adult thyroid cancer',pair_quote='Twelve patients underwent PET imaging.', result_quote='Uptake was observed in 28 of 43 lesions.')
    c.update(tumor_subtype_id='p',tumor_subtype_name='Prostate Carcinoma')
    s='TITLE: Imaging of thyroid cancer.\n'+s
    assert r2.extra_ownership_reason(q,c,s,T)=='source_confirmed_other_tumor_cohort'


def test_unverified_llm_scope_not_sufficient():
    q,c,s,_,_=example(clinical_setting='adult thyroid cancer')
    c.update(tumor_subtype_id='p',tumor_subtype_name='Prostate Carcinoma')
    assert r2.extra_ownership_reason(q,c,s,T) is None


def test_whole_pipeline_missing_ancestry():
    q,c,s,f,row=example(clinical_setting='patients with prostate cancer',pair_quote='Patients with prostate cancer received 177Lu-X.')
    c.update(tumor_subtype_id='pa',tumor_subtype_name='Prostate Adenocarcinoma')
    f['tumor_subtype_id']='pa'
    assert r2.apply(row,{'claims':[q]},c,[f],{'s':s},'key',T)['predicted_evidence_conclusion']=='supportive'
