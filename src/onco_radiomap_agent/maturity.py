from __future__ import annotations
from . import prediction_rules as rules



import re

from copy import deepcopy

from pathlib import Path



from .ownership_base import recover_span, isotopes, ownership_reason

from .ownership import Tumors, extra_ownership_reason




VERSION = 'C2.1-current-study-design-3-protocol-guard'

SECTION = re.compile(r'(?<!\S)(Title|Abstract|Background|Introduction|Purpose|Objectives?|'
    r'Patients and Methods|Materials? and Methods|Methods? & Design|Methods?(?: and Materials)?|Experimental Design|'
    r'Results?|Findings|Conclusions?|Interpretation|Discussion|DOI)\s*:\s*', re.I)

FUTURE = re.compile(r'\b(?:will|would|future|warranted|needed|required|should be|'
    r'prior to expansion|before expansion|prior to randomi[sz]ation|before randomi[sz]ation)\b', re.I)

NEG_RANDOM = re.compile(r'\b(?:non[- ]?randomi[sz]ed|not randomi[sz]ed|without randomi[sz]ation)\b', re.I)

CONTROL = r'\b(?:randomi[sz]ed\s+controlled|controlled\s+(?:clinical\s+)?(?:trial|study)|placebo[- ]controlled|(?:control|comparator)\s+(?:arm|group|cohort))\b'

RANDOM = r'\b(?:randomi[sz]ed|randomly\s+(?:assigned|allocated))\b'

PATTERNS = {
    'prospective': r'\bprospectiv(?:e|ely)\b',
    'multicenter': r'\bmulti[- ]?cent(?:er|re)(?:ed)?\b',
    'phase_iii': r'\bphase\s+(?:iii|3)\b',
    'randomized': RANDOM,
    'controlled': CONTROL,
    'credible_comparator': CONTROL,
    'single_arm_defined_cohort': r'\b(?:single[- ]arm|consecutive\s+patients?|dose[- ]escalation|lead[- ]in\s+(?:phase|cohort))\b',
    'prespecified_diagnostic_performance': r'\b(?:sensitivity|specificity|accuracy|auc|receiver\s+operating|correct\s+locali[sz]ation\s+rate)\b',
    'gold_standard_comparison': r'\b(?:reference\s+standard|standard\s+of\s+truth|histopatholog\w*|pathologic\w*\s+confirmation|biopsy)\b',
    'head_to_head_tracer_comparison': r'\b(?:head[- ]to[- ]head|tracer\s+comparison|compared\s+with\s+[^.;]+(?:pet|spect))\b',
    'management_impact_evaluated': r'\b(?:management\s+impact|changed\s+(?:the\s+)?(?:clinical\s+)?management|change(?:s)?\s+in\s+(?:intended\s+|clinical\s+)?management|treatment\s+decisions?)\b',
    'patient_relevant_diagnostic_utility_endpoint': r'\b(?:management\s+impact|patient\s+outcome|clinical\s+utility)\b',
    'prespecified_objective_efficacy': r'\b(?:primary\s+(?:end\s*point|objective)|objective\s+response|response\s+rate|progression[- ]free|overall\s+survival)\b',
    'patient_relevant_treatment_endpoint': r'\b(?:overall\s+survival|progression[- ]free|quality\s+of\s+life|symptom\s+control)\b',
}

STRUCTURAL = {'prospective','multicenter','phase_iii','randomized','controlled','credible_comparator','single_arm_defined_cohort'}

BACKGROUND = {'BACKGROUND','INTRODUCTION','PURPOSE','OBJECTIVE','OBJECTIVES'}

END = {'CONCLUSION','CONCLUSIONS','INTERPRETATION','DISCUSSION'}

def current_study_statement(quote, title):
    """Location is not ownership: require explicit current-study syntax."""
    if FUTURE.search(quote) or re.search(r'\b(?:previous|prior study|earlier study|other studies|reported by|review|meta-analysis|retrospective)\b',quote,re.I):return False
    if re.search(r'\b(?:this|our|the present)\s+(?:study|trial|cohort)\s+(?:was|is)\b|\bwe\s+(?:conducted|performed|undertook)\b',quote,re.I):return True
    # Trial names are resolved from this source's title, never from a trial-name whitelist.
    for name in re.findall(r'\b[A-Z][A-Z0-9-]{2,}\b',title):
        if name in {'PET','SPECT','PSA','MRI','III'}:continue
        if re.search(r'\b'+re.escape(name)+r'\s+(?:was|is)\s+(?:a\s+)?(?:prospective|multicenter|randomized|phase)\b',quote,re.I):return True
    return False

def segments(text):
    """Keep original contiguous spans; no normalization of quoted evidence."""
    markers = list(SECTION.finditer(text))
    blocks = []
    if not markers: blocks = [('UNSTRUCTURED', 0, len(text))]
    elif markers[0].start(): blocks.append(('UNSTRUCTURED', 0, markers[0].start()))
    for i,m in enumerate(markers):
        section = m.group(1).upper()
        if section == 'DOI': continue
        blocks.append(('UNSTRUCTURED' if section == 'ABSTRACT' else section,
                       m.end(), markers[i+1].start() if i+1<len(markers) else len(text)))
    out = []
    for section,a,b in blocks:
        # A decimal point is not a sentence boundary.
        for m in re.finditer(r'[^\n]+?(?:[.!?](?=\s+[A-Z]|$)|$)', text[a:b]):
            raw = m.group(); left = len(raw)-len(raw.lstrip()); quote = raw.strip()
            if quote: out.append(dict(section=section,quote=quote,start=a+m.start()+left,end=a+m.start()+left+len(quote)))
    return out

def design_repair(fact):
    text = fact['raw_source_text']; segs = segments(text)
    title=' '.join(s['quote'] for s in segs if s['section']=='TITLE')
    flags = deepcopy(fact['design']); evidence = {}; rejected = {}
    for field,pattern in PATTERNS.items():
        yes=[]; no=[]; excluded=[]
        for seg in segs:
            t=seg['quote']
            if not re.search(pattern,t,re.I): continue
            why=None
            if seg['section'] in BACKGROUND and not (field in STRUCTURAL and current_study_statement(t,title)):why='not_current_design_section'
            if field in STRUCTURAL and seg['section'] in END:why='not_current_design_section'
            if FUTURE.search(t): why='future_or_recommendation'
            if why: excluded.append(dict(**seg,reason=why)); continue
            negative = (field in {'randomized','credible_comparator'} and NEG_RANDOM.search(t)) or (
                field in {'controlled','credible_comparator'} and re.search(r'\b(?:uncontrolled|non[- ]controlled|without (?:a )?(?:control|comparator))\b',t,re.I))
            if field=='prospective' and re.search(r'\b(?:not prospective|non[- ]prospective)\b',t,re.I):negative=True
            if negative: no.append(seg)
            else: yes.append(seg)
        flags[field] = True if yes else False if no else None
        evidence[field]=dict(value=flags[field],positive=yes,negative=no)
        rejected[field]=excluded

    def set_from(field, spans, reason):
        if spans:
            flags[field]=True
            evidence[field]=dict(value=True,positive=spans,negative=[],basis=reason)

    # Random allocation to explicit alternative treatment arms implies a control design.
    random_spans=evidence['randomized']['positive']
    comparison_spans=[s for s in random_spans if re.search(r'\b(?:or|versus|vs|alone|control)\b',s['quote'],re.I)]
    set_from('controlled',comparison_spans,'random_allocation_to_alternative_arms')
    set_from('credible_comparator',random_spans,'current_random_assignment')
    if not flags['credible_comparator']:
        comparator=[]
        for s in segs:
            t=s['quote']
            if s['section'] in BACKGROUND|END or FUTURE.search(t):continue
            if re.search(r'\b(?:therapy|treatment|regimen|monotherapy|combination)\b.{0,100}\b(?:versus|vs\.?|compared with)\b.{0,100}\b(?:therapy|treatment|regimen|monotherapy|combination)\b',t,re.I):comparator.append(s)
        set_from('credible_comparator',comparator,'explicit_treatment_comparator')
    if not any(flags[k] for k in ['randomized','controlled','credible_comparator']):
        early=[s for s in segs if s['section'] not in BACKGROUND|END and not FUTURE.search(s['quote']) and re.search(r'\bphase\s+(?:i|1|ii|2)\b',s['quote'],re.I)]
        set_from('single_arm_defined_cohort',early,'noncomparative_early_phase')
    # PSA alone or a retrospective PSA number is not a pre-specified efficacy endpoint.
    efficacy=[]
    for s in segs:
        t=s['quote']
        if 'METHOD' not in s['section'] and s['section']!='EXPERIMENTAL DESIGN':continue
        if FUTURE.search(t):continue
        response=bool(re.search(r'\b(?:PSA|prostate[- ]specific antigen|imaging|radiographic|tumou?r)\s+response\b|\bresponse\s+to\b',t,re.I))
        if re.search(r'\bresponse\s+(?:was|were|of)\b',t,re.I) and re.search(r'\b(?:PSA|prostate[- ]specific antigen|PET|CT|imaging|radiographic|RECIST)\b',t,re.I):response=True
        evaluation=bool(re.search(r'\b(?:assess\w*|evaluat\w*|measur\w*)\b',t,re.I))
        scheduled=bool(re.search(r'\bPSA\b',t,re.I) and re.search(r'\b(?:each cycle|serial|baseline and)\b',t,re.I) and evaluation)
        if (response and evaluation) or scheduled:efficacy.append(s)
    set_from('prespecified_objective_efficacy',efficacy,'method_defined_response_assessment')
    # Explicit entry criteria plus a common observed treatment schedule define a cohort;
    # do not infer this from a response percentage or a source merely mentioning patients.
    methods=[s for s in segs if ('METHOD' in s['section'] or s['section']=='EXPERIMENTAL DESIGN') and not FUTURE.search(s['quote'])]
    inclusion=[s for s in methods if re.search(r'\b(?:patients?|participants?|men|women)\b.{0,160}\b(?:included|enrolled|eligible)\b',s['quote'],re.I)]
    regimen=[s for s in methods if re.search(r'\b(?:administered|received|prescribed|therapy)\b',s['quote'],re.I) and re.search(r'\b(?:cycles?|intervals?|weeks?|GBq|MBq)\b',s['quote'],re.I)]
    observed=[s for s in segs if s['section'] in {'RESULT','RESULTS','FINDINGS'} and re.search(r'\b(?:\d{2,}|(?:one\s+)?hundred|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(?:patients|participants|men|women)\b',s['quote'],re.I)]
    if inclusion and regimen and observed and not any(flags[k] for k in ['randomized','controlled','credible_comparator']):
        set_from('single_arm_defined_cohort',inclusion+regimen+observed,'explicit_entry_common_regimen_observed_cohort')
    # A lead-in report is not the parent randomized trial; require current allocation in METHODS.
    lead=[s for s in segs if s['section']=='TITLE' and re.search(r'\blead[- ]in\s+(?:cohort|phase)\b',s['quote'],re.I)]
    method_random=[s for s in random_spans if 'METHOD' in s['section'] or s['section']=='EXPERIMENTAL DESIGN']
    if lead and not method_random:
        for key in ['phase_iii','randomized','controlled','credible_comparator']:
            rejected[key]+=evidence[key]['positive']
            flags[key]=False
            evidence[key]=dict(value=False,positive=[],negative=lead,basis='lead_in_not_parent_comparative_results')
        set_from('single_arm_defined_cohort',lead,'lead_in_cohort')
    for k in list(PATTERNS):
        wrong_use=(fact['clinical_use']!='diagnostic' and k in {'prespecified_diagnostic_performance','gold_standard_comparison','head_to_head_tracer_comparison','management_impact_evaluated','patient_relevant_diagnostic_utility_endpoint'}) or (fact['clinical_use']!='therapeutic' and k in {'prespecified_objective_efficacy','patient_relevant_treatment_endpoint'})
        if wrong_use:flags[k]=False;evidence[k]=dict(value=False,positive=[],negative=[],basis='different_endpoint_use')
    # A protocol's title and proposed endpoints cannot establish completed efficacy.
    # Only explicit protocol/design descriptions + planned participant allocation,
    # without an actual Results/Findings section, activate this guard.
    protocol_context=[s for s in segs if ('DESIGN' in s['section'] and 'METHOD' in s['section']) or
                      (s['section']=='TITLE' and re.search(r'\b(?:study protocol|trial protocol|rationale and design)\b',s['quote'],re.I))]
    planned=[s for s in segs if re.search(r'\b(?:patients?|participants?)\b.*\bwill\s+(?:be\s+)?(?:randomi[sz]ed|enrolled|receive|treated)\b',s['quote'],re.I)]
    actual_results=[s for s in segs if s['section'] in {'RESULT','RESULTS','FINDINGS'} and not FUTURE.search(s['quote'])]
    if protocol_context and planned and not actual_results:
        for key in PATTERNS:
            rejected[key]+=evidence[key]['positive']
            flags[key]=False
            evidence[key]=dict(value=False,positive=[],negative=protocol_context+planned,basis='planned_protocol_without_reported_results')
    changed={k:dict(before=fact['design'].get(k),after=v) for k,v in flags.items() if fact['design'].get(k)!=v}
    repaired=deepcopy(fact);repaired['design']=flags
    return repaired,dict(fields=evidence,rejected_contexts=rejected,changes=changed,
                         untouched_fields=sorted(set(flags)-set(PATTERNS)))

def ownership_conflicts(case, fact, alignment, tumors):
    """Only explicit, verbatim grounded conflicts. Missing claims never exclude."""
    source=fact['raw_source_text'];sid=fact['source_id']; conflicts=[]
    accepted=[c for c in alignment.get('claims',[]) if c.get('source_id')==sid]
    # Scope disagreement in one rejected claim does not invalidate another accepted target result.
    if not accepted:
        for row in alignment.get('rejected_claims',[]):
            c=row.get('claim',{})
            if not isinstance(c,dict) or c.get('source_id')!=sid:continue
            if not all(isinstance(c.get(k),str) and recover_span(c[k],source) for k in ['pair_quote','result_quote']):continue
            reason=ownership_reason(c,case,source,tumors) or extra_ownership_reason(c,case,source,tumors)
            if not reason and c.get('directness') in {'other_use','other_agent'}:
                # Require concrete isotope mismatch in the result, not just an LLM scope label.
                target=isotopes(case['radiopharmaceutical_name']); actual=isotopes(c['result_quote'])
                if target and actual and target.isdisjoint(actual):reason='result_belongs_to_other_isotope_or_use'
            if reason:conflicts.append(dict(reason=reason,pair_quote=c['pair_quote'],result_quote=c['result_quote'],source_id=sid))
    return conflicts

def candidate(row, case, facts, tumors, *, ownership=True):
    out=deepcopy(row)
    if row['case_type']!='pair_snapshot' or row['prediction_status']!='valid':return out
    proposal=rules.predict_snapshot(facts,allowed_fact_ids={f['fact_id'] for f in facts},
        **{k:case[k] for k in ['dataset_id','tumor_subtype_id','radiopharmaceutical_id','clinical_use','population_scope','cutoff_date']})
    assert proposal['predicted_maturity_code']==row['predicted_maturity_code'], 'Unreproduced baseline maturity'
    selected=[f for f in facts if f['fact_id'] in proposal['supporting_fact_ids']]
    alignment=row.get('prediction_provenance',{}).get('conclusion_alignment_C1_1',{})
    kept=[];traces=[]
    for f in selected:
        repaired,trace=design_repair(f)
        bad=ownership_conflicts(case,f,alignment,tumors) if ownership else []
        traces.append(dict(fact_id=f['fact_id'],source_id=f['source_id'],study_family_id=f['study_family_id'],
            locator=f.get('record_locator'),before_grade=rules._maturity(f),after_design_grade=rules._maturity(repaired),
            design=trace,ownership_conflicts=bad,included=not bool(bad)))
        if not bad:kept.append(repaired)
    families=rules._family_rows(kept) if kept else []
    grade=rules._snapshot_maturity(case['clinical_use'],families) if families else ('D1' if case['clinical_use']=='diagnostic' else 'T1')
    out['predicted_maturity_code']=grade
    out['prediction_provenance']['maturity_C2']=dict(version=VERSION,arm='design_and_ownership' if ownership else 'design_only',
        previous_grade=row['predicted_maturity_code'],grade=grade,source_traces=traces,study_families=families,
        legacy_floor_without_target_evidence=not bool(kept),status_policy_unchanged=True,
        secondary_source_and_independence_policy_unchanged=True,api_calls=0,expert_answers_accessed=False)
    assert {k:v for k,v in out.items() if k not in {'predicted_maturity_code','prediction_provenance'}}=={k:v for k,v in row.items() if k not in {'predicted_maturity_code','prediction_provenance'}}
    return out
