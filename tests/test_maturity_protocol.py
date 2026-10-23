import pytest

from onco_radiomap_agent import maturity as final

from test_maturity import fact

@pytest.mark.parametrize('length',[0,100,300,600])
def test_long_inclusion_criteria_do_not_hide_future_allocation(length):
    criteria=' satisfying additional baseline imaging and clinical criteria'*(length//50)
    text='Title: Randomized controlled trial. Abstract: METHODS & DESIGN: Fifty-eight patients'+criteria+' will be randomized in a 1:1 ratio. The primary endpoint is overall survival. DISCUSSION: We will evaluate efficacy.'
    f,t=final.design_repair(fact(text))
    assert final.rules._maturity(f)=='T1'
    assert t['fields']['randomized']['basis']=='planned_protocol_without_reported_results'

def test_actual_randomized_result_is_not_protocol():
    text='Title: Randomized controlled phase III study. Abstract: METHODS: We randomly assigned patients to drug X or placebo. RESULTS: Overall survival was improved. CONCLUSION: Future studies will enroll more patients.'
    f,t=final.design_repair(fact(text))
    assert final.rules._maturity(f)=='T4'
