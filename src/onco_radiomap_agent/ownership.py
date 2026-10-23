"""Source-audit follow-up: coordinated cohorts, actual human administration,
and explicit target outcomes override a broad cohort label. No score access.
"""
import re
from . import ownership_base as base
from . import ownership_population as r2

VERSION='C1.1-source-ownership-3'


class Tumors(r2.Tumors):
    def hits(self,text):
        # Expand grammatical coordination, not synonyms or disease equivalences.
        word=r'(?:prostate|breast|thyroid|pancreatic|bladder|lung|renal|kidney|liver|colorectal|gastric|ovarian)'
        pattern=rf'\b{word}(?:(?:,\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+){word})+\s+(?:cancer|carcinoma)s?\b'
        expanded=text
        for m in re.finditer(pattern,text,re.I):
            expanded+=' '+' '.join(w+' cancer' for w in re.findall(word,m.group(),re.I))
        return super().hits(expanded)


def human_administration(source):
    # Actual administration, not planned/future use or a human-derived sample alone.
    for seg in base.ca.segments(source):
        text=seg['text']
        if re.search(r'\b(?:patients?|subjects?)\b.{0,75}\b(?:injected|administered)\b',text,re.I):
            if not re.search(r'\b(?:will|planned|aim|would|not)\b',text,re.I):return True
    return False


def extra_ownership_reason(claim,case,source,tumors):
    reason=r2.extra_ownership_reason(claim,case,source,tumors)
    if reason=='ex_vivo_human_specimen_not_clinical_intervention' and case['clinical_use']=='diagnostic' and human_administration(source):
        return None
    if reason=='source_confirmed_other_tumor_cohort' and tumors.target_hit(claim['result_quote'],case['tumor_subtype_id']):
        return None
    return reason


def apply(row,raw,case,facts,sources,key,tumors):
    out=base.apply(row,raw,case,facts,sources,key,tumors)
    if row['prediction_status']!='valid':return out
    audit=out['prediction_provenance']['conclusion_alignment_C1_1']
    kept=[]
    for c in audit['claims']:
        reason=extra_ownership_reason(c,case,sources[c['source_id']],tumors)
        if reason:audit['rejected_claims'].append(dict(claim=c,reason=reason))
        else:kept.append(c)
    conclusion,reason=base.ca.aggregate_claims(kept)
    audit.update(version=VERSION,claims=kept,conclusion=conclusion,reason=reason,
        mixed_relative_endpoints_require_explicit_summary=(conclusion=='inconclusive' and any(c['direction']=='negative' and c['endpoint_scope']['relative_comparison'] for c in kept)))
    out['predicted_evidence_conclusion']=conclusion
    return out
