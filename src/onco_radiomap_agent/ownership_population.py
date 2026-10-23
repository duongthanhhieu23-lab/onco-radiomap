"""Revision 2 fixes identified by label-blind inspection of all changed outputs.

Incomplete registry ancestry does not prove distinct populations. A positive
conflict requires incompatible organs/histologies. Ex-vivo human specimens
are not human clinical interventions. A verified source cohort also scopes
claims whose individual result excerpt omits the disease name.
"""
import re
from copy import deepcopy
from . import ownership_base as base

VERSION = 'C1.1-source-ownership-2'
ORGANS = {
    'prostate': r'\bprostat\w*\b', 'thyroid': r'\bthyroid\b',
    'kidney': r'\b(?:kidney|renal)\b', 'bladder': r'\bbladder\b',
    'breast': r'\bbreast\b', 'lung': r'\b(?:lung|pulmonary)\b',
    'liver': r'\b(?:liver|hepatic|hepatocellular)\b',
    'pancreas': r'\bpancrea\w*\b', 'colorectal': r'\b(?:colon|colorectal|rectal|rectum)\b',
    'ovary': r'\bovar\w*\b', 'uterus': r'\b(?:uterine|uterus|endometrial)\b',
    'stomach': r'\b(?:gastric|stomach)\b', 'esophagus': r'\b(?:oesophag\w*|esophag\w*)\b',
    'skin': r'\b(?:skin|cutaneous)\b', 'brain': r'\b(?:brain|cerebral)\b',
}


def organs(text):
    return {k for k,p in ORGANS.items() if re.search(p, text, re.I)}


class Tumors(base.Tumors):
    def __init__(self, records):
        super().__init__(records)
        self.names = {r['tumor_subtype_id']:r['preferred_name'] for r in records}

    def hits(self, text):
        # Negative population qualifiers are not positive entity membership.
        text = re.sub(r'\bnon[- ](?:[a-z]+[- ]?){0,2}(?:cancer|carcinoma)\b', '', text, flags=re.I)
        return super().hits(text)

    def compatible(self, a, b):
        if super().compatible(a,b): return True
        left,right=organs(self.names.get(a,'')),organs(self.names.get(b,''))
        # Unresolved ancestry is NOT evidence of incompatibility.
        return not (left and right and left.isdisjoint(right))


def extra_ownership_reason(claim, case, source, tumors):
    scope=claim['clinical_setting']; result=claim['result_quote']
    if re.search(r'\bex[- ]vivo\b', scope, re.I) or re.search(r'\b(?:autoradiography|dosimetry calculations)\b.*\b(?:patient|human) (?:tumou?r |tissue )?samples\b', result, re.I):
        return 'ex_vivo_human_specimen_not_clinical_intervention'
    target=case['tumor_subtype_id']
    hits=tumors.hits(scope)
    if hits and not any(tumors.compatible(target,x) for x in hits):
        # Establish the cohort in an actual visible source section, not solely an LLM scope label.
        for seg in base.ca.segments(source):
            if ('METHOD' in seg['section'] or seg['section']=='TITLE') and hits & tumors.hits(seg['text']):
                return 'source_confirmed_other_tumor_cohort'
    return None


def apply(row, raw, case, facts, sources, key, tumors):
    out=base.apply(row,raw,case,facts,sources,key,tumors)
    if row['prediction_status']!='valid':return out
    audit=out['prediction_provenance']['conclusion_alignment_C1_1']
    kept=[]
    for c in audit['claims']:
        reason=extra_ownership_reason(c,case,sources[c['source_id']],tumors)
        if reason:audit['rejected_claims'].append(dict(claim=c,reason=reason))
        else:kept.append(c)
    audit['claims']=kept
    conclusion,reason=base.ca.aggregate_claims(kept)
    audit.update(version=VERSION,conclusion=conclusion,reason=reason,
        mixed_relative_endpoints_require_explicit_summary=(conclusion=='inconclusive' and any(c['direction']=='negative' and c['endpoint_scope']['relative_comparison'] for c in kept)))
    out['predicted_evidence_conclusion']=conclusion
    return out
