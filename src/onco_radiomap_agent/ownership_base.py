from __future__ import annotations

import csv

import json

import re

import unicodedata

from copy import deepcopy

from .claims import ca, assert_no_answers

VERSION = 'C1.1-source-ownership-1'

ANIMAL = re.compile(r'\b(?:murine|mice|mouse|rats?|xenograft|cell lines?|in vitro)\b', re.I)

HUMAN = re.compile(r'\b(?:patients?|women|men|human subjects|clinical cohort)\b', re.I)

NEGATED = re.compile(r'\b(?:not|no|without|failed|didn.t)\b', re.I)

def textkey(s):
    s = unicodedata.normalize('NFKC', str(s)).casefold()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()

class Tumors:
    def __init__(self, records):
        self.rows = records
        self.parent = {r['tumor_subtype_id']: r.get('parent_tumor_subtype_id') for r in records}
        self.aliases = {r['tumor_subtype_id']: sorted(set(textkey(a) for a in
                        [r['preferred_name']] + json.loads(r.get('aliases') or '[]') if len(a) >= 3)) for r in records}

    @classmethod
    def load(cls, path):
        with path.open(encoding='utf-8-sig', newline='') as f:
            return cls(list(csv.DictReader(f)))

    def hits(self, s):
        value = ' ' + textkey(s) + ' '
        found = [(tid, a) for tid, aliases in self.aliases.items() for a in aliases if ' '+a+' ' in value]
        # A child alias must not trigger its lexically nested parent by itself.
        return {tid for tid, a in found if not any(a != b and ' '+a+' ' in ' '+b+' ' for _, b in found)}

    def compatible(self, a, b):
        def ancestors(x):
            result = set()
            while x and x not in result:
                result.add(x)
                x = self.parent.get(x)
            return result
        return a in ancestors(b) or b in ancestors(a)

    def target_hit(self, text, target):
        # For subgroup attribution require the actual target name/alias, not its parent.
        return target in self.hits(text)

def canonical_with_positions(s):
    """Case/whitespace matching plus operator-adjacent whitespace only."""
    chars, positions = [], []
    for i, ch in enumerate(s):
        for low in ch.casefold():
            if ch.isspace():
                if chars and chars[-1] != ' ':
                    chars.append(' '); positions.append(i)
            else:
                chars.append(low); positions.append(i)
    keep = [i for i, ch in enumerate(chars) if ch != ' ' or not (
        (i and chars[i-1] in '<=>≤≥') or (i+1 < len(chars) and chars[i+1] in '<=>≤≥'))]
    while keep and chars[keep[0]] == ' ': keep.pop(0)
    while keep and chars[keep[-1]] == ' ': keep.pop()
    return ''.join(chars[i] for i in keep), [positions[i] for i in keep]

def recover_span(quote, source):
    needle, _ = canonical_with_positions(quote)
    hay, pos = canonical_with_positions(source)
    start = hay.find(needle) if needle else -1
    return source[pos[start]:pos[start+len(needle)-1]+1] if start >= 0 else None

def isotopes(s):
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\[\]()]', ' ', s)
    values = {(a.casefold(), int(n)) for n, a in re.findall(r'(?<!\d)(\d{2,3})\s*-?\s*(Lu|Ga|Tc|In|Ac|Y|I|F|Cu|Zr)\b', s, re.I)}
    values |= {(a.casefold(), int(n)) for a, n in re.findall(r'\b(Lu|Ga|Tc|In|Ac|Y|I|F|Cu|Zr)\s*-\s*(\d{2,3})(?!\d)', s, re.I)}
    return values

def mixed_population(claim, source, tumors):
    scope = claim.get('clinical_setting', '')
    if re.search(r'\b(?:mixed (?:tumou?rs?|cancers?)|various cancers|different cancers)\b', scope, re.I):
        return True
    if re.search(r'\bincluding\b', scope, re.I) and len(tumors.hits(scope)) > 1:
        return True
    for marker in re.finditer(r'\b(?:clinical subgroups|group consisted of|mixed (?:tumou?r|cancer)|different cancers)\b', source, re.I):
        neighborhood = source[max(0, marker.start()-80):marker.end()+350]
        disease_terms = re.findall(r'\b(?:cancers?|carcinomas?|carcinoids?|sarcomas?|lymphomas?)\b', neighborhood, re.I)
        if len(tumors.hits(neighborhood)) > 1 or len(disease_terms) > 1:
            return True
    return False

def ownership_reason(claim, case, source, tumors):
    setting = claim['clinical_setting']
    result = claim['result_quote']
    pair = claim['pair_quote']
    # Explicit human result remains usable in mixed animal/human papers.
    if ANIMAL.search(result) and not HUMAN.search(result):
        return 'explicit_nonhuman_result'
    if ANIMAL.search(setting) and not HUMAN.search(result):
        return 'explicit_nonhuman_setting'
    target = case['tumor_subtype_id']
    setting_hits = tumors.hits(setting)
    quote_hits = tumors.hits(pair+' '+result)
    histology = r'\b(urothelial|squamous|small[- ]cell|clear[- ]cell|medullary|papillary|follicular)\s+(?:(?:renal|thyroid|gland|cell)\s+){0,3}(?:carcinoma|cancer)\b'
    # Explicit incompatible histologies, including generic labels absent from site-specific registry entries.
    target_histology = set(re.findall(histology, case.get('tumor_subtype_name', ''), re.I))
    setting_histology = set(re.findall(histology, setting, re.I))
    quote_histology = set(re.findall(histology, pair+' '+result, re.I))
    target_histology = {textkey(x) for x in target_histology}
    other_histology = {textkey(x) for x in setting_histology & quote_histology}
    if target_histology and other_histology and not target_histology & other_histology:
        return 'explicit_incompatible_histology'
    if (setting_hits and not any(tumors.compatible(target, x) for x in setting_hits)
            and setting_hits & quote_hits):
        return 'explicit_other_tumor_population'
    if mixed_population(claim, source, tumors):
        target_result = tumors.target_hit(result, target)
        if re.search(r'\b(?:including|overall|pooled)\b', result, re.I):
            target_result = False
        if not target_result:
            return 'mixed_cohort_without_target_result'
    # Scope contains prior radiotherapy; the reported outcome explicitly belongs to rehabilitation.
    if (case['clinical_use'] == 'therapeutic'
            and re.search(r'\brehabilitation\b', result, re.I)
            and re.search(r'\b(?:following|after|followed by|post[- ]therapy)\b', setting+' '+claim['regimen'], re.I)
            and re.search(r'\b(?:rehabilitation[- ]related|rehabilitation group)\b', result, re.I)):
        return 'outcome_of_subsequent_ancillary_intervention'
    if case['clinical_use'] == 'diagnostic':
        regimen = claim['regimen']
        parts = re.split(r'\b(?:before|prior to)\b', regimen, maxsplit=1, flags=re.I)
        target_isotopes = isotopes(case['radiopharmaceutical_name'])
        if (len(parts) == 2 and re.search(r'\b(?:therapy|treatment)\b', parts[1], re.I)
                and target_isotopes & isotopes(parts[1]) and isotopes(parts[0])
                and not target_isotopes & isotopes(parts[0])):
            return 'target_is_therapy_not_observed_diagnostic_tracer'
    return None

def repair_claim(claim, case, facts, sources):
    item = deepcopy(claim)
    edits = []
    sid = item.get('source_id')
    source = sources.get(sid, '')
    texts = [f['raw_source_text'] for f in facts if f['source_id'] == sid]
    for key in ['pair_quote', 'result_quote']:
        quote = item.get(key)
        if not isinstance(quote, str) or not 12 <= len(quote.strip()) <= 700:
            continue
        if ca.norm(quote) in ca.norm(source) and any(ca.norm(quote) in ca.norm(t) for t in texts):
            continue
        span = recover_span(quote, source)
        rawspan = next((s for t in texts if (s := recover_span(quote, t)) is not None), None)
        # Use a real contiguous visible span that is also accepted by the frozen raw validator.
        if span and rawspan and any(ca.norm(span) in ca.norm(t) for t in texts):
            item[key] = span
            edits.append(dict(kind='operator_whitespace_only', field=key, original=quote,
                              visible_span=span, raw_span=rawspan))
    q = item.get('result_quote', '')
    if (case['clinical_use'] == 'therapeutic' and item.get('level') == 'imaging_biomarker_association'
            and item.get('direction') == 'supportive' and not NEGATED.search(q)
            and re.search(r'\btumou?r (?:area|volume|size)\s+(?:was |were |had )?(?:decreased|reduced|shrank)\s+(?:by )?\d+(?:\.\d+)?\s*%', q, re.I)
            and re.search(r'\b(?:baseline|follow[- ]up|after treatment)\b', q, re.I)
            and not re.search(r'\b(?:correlat\w*|predict\w*|associated)\b', q, re.I)):
        edits.append(dict(kind='actual_tumor_shrinkage_not_biomarker', old_level=item['level'], old_endpoint=item['endpoint_family']))
        item.update(level='therapeutic_tumor_control', endpoint_family='tumor_control')
    return item, edits

def endpoint_scope(claim):
    quote = claim['result_quote']
    metrics = [x for x in ['sensitivity', 'specificity', 'AUC', 'accuracy', 'false-positive'] if x.casefold() in quote.casefold()]
    comparative = bool(re.search(r'\b(?:higher|lower|superior|inferior|better|worse)\b.*\b(?:than|compared|versus|to)\b', quote, re.I))
    return dict(reported_metrics=metrics, relative_comparison=comparative,
                evidence_scope='relative_comparator_performance' if comparative else 'reported_endpoint',
                comparison_quote=quote if comparative else None)

def apply(row, raw, case, facts, sources, key, tumors):
    assert_no_answers(case)
    assert_no_answers(raw)
    if row['prediction_status'] != 'valid': return deepcopy(row)
    # Contract validation before iterating; malformed responses remain hard failures.
    if not isinstance(raw, dict) or not isinstance(raw.get('claims'), list) or len(raw['claims']) > 6:
        raise RuntimeError('Invalid cached claims contract')
    accepted, rejected, repairs = [], [], []
    for original in raw['claims']:
        if not isinstance(original, dict):
            rejected.append(dict(claim=original, reason='not_object')); continue
        claim, edits = repair_claim(original, case, facts, sources)
        reason = None
        ok, bad = ca.validate_claims({'claims': [claim]}, case, facts, sources)
        if bad:
            rejected.extend(bad)
        else:
            reason = ownership_reason(claim, case, sources[claim['source_id']], tumors)
            if reason:
                rejected.append(dict(claim=claim, reason=reason))
            else:
                ok[0]['endpoint_scope'] = endpoint_scope(claim)
                accepted.extend(ok)
        if edits: repairs.append(dict(source_id=claim.get('source_id'), edits=edits, accepted=bool(ok and not bad and not reason)))
    conclusion, reason = ca.aggregate_claims(accepted)
    # Relative disadvantages remain in provenance AND aggregation. No score-driven suppression.
    mixed_relative = (conclusion == 'inconclusive' and any(c['direction'] == 'negative' and c['endpoint_scope']['relative_comparison'] for c in accepted))
    out = deepcopy(row)
    out['predicted_evidence_conclusion'] = conclusion
    out['prediction_provenance']['conclusion_alignment_C1_1'] = dict(
        version=VERSION, conclusion=conclusion, reason=reason, claims=accepted, rejected_claims=rejected,
        repairs=repairs, mixed_relative_endpoints_require_explicit_summary=mixed_relative,
        cache_sha256=key, expert_answers_accessed=False, api_calls=0,
        original_C1_conclusion=row['predicted_evidence_conclusion'])
    return out
