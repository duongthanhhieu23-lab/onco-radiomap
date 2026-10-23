"""Portable orchestration of the frozen V1.3.2 classification and claim stages.

Input data and retrieval contexts are supplied explicitly. A responder receives
a request job and returns a provider receipt; it may be a live client or a
strict offline receipt lookup. No reference labels enter this interface.
"""
from __future__ import annotations
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

from . import arbiter, preparation, task_contract
from .claim_context import pair_context
from .claims import apply_conclusion_only, conclusion_request, ca
from .maturity_compat import candidate, required_claim_case_ids
from .ownership import Tumors, apply as apply_ownership
from .response_io import digest, parsed_response

METHODS = preparation.METHODS
RESOURCES = Path(__file__).with_name('resources')
arbiter.PROMPT_VERSION = task_contract.VERSION
arbiter.FULL_SYSTEM_PROMPT_VERSION = task_contract.VERSION


def registries(directory: Path = RESOURCES):
    """Load the frozen names and allowed normalization IDs."""
    with (directory / 'entities_a.csv').open(encoding='utf-8-sig', newline='') as f:
        tumor_ids = {r['tumor_subtype_id'] for r in csv.DictReader(f)}
    with (directory / 'entities_b.csv').open(encoding='utf-8-sig', newline='') as f:
        radio_ids = {r['radiopharmaceutical_id'] for r in csv.DictReader(f)}
    provisional = json.loads((directory / 'provisional_radiopharmaceuticals.json').read_text('utf-8'))
    radio_ids.update(r['radiopharmaceutical_id'] for r in provisional['records'])
    return tumor_ids, radio_ids, Tumors.load(directory / 'entities_a.csv'), ca.load_aliases(directory / 'entities_b.csv')


def validate_inputs(cases, facts, contexts):
    task_contract.answer_free([cases, facts, contexts])
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Supply a nonempty dataset with unique case IDs')
    if len({c['dataset_id'] for c in cases}) != 1:
        raise ValueError('Process each dataset separately to preserve source isolation')
    dataset_id = cases[0]['dataset_id']
    if any(f['origin_dataset_id'] != dataset_id for f in facts):
        raise ValueError('Prediction facts must belong to the requested dataset')


def prepare_jobs(cases, facts, contexts, *, run_id=1, methods=METHODS):
    """Return deterministic rules and the exact frozen classification requests.

    contexts maps naive_rag/lightrag to {case_id: context_string}. Supply the
    complete dataset case pool so pair rules can access its atomic evidence.
    """
    validate_inputs(cases, facts, contexts)
    if type(run_id) is not int or run_id < 1:
        raise ValueError('run_id must be a positive integer')
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('Unknown, empty or duplicate method selection')
    rules = preparation._full_system_rows(cases, facts=facts, evidence_pool_cases=cases, run_id=run_id)
    proposals = preparation._full_system_proposals(cases, rule_rows=rules)
    jobs = []
    for method in methods:
        for case in cases:
            proposal = proposals[case['dataset_id'], case['case_id']] if method == 'full_system' else None
            if method == 'full_system' and not preparation._requires_full_system_arbiter(case, proposal):
                continue
            retrieval_method = 'lightrag' if method == 'full_system' else method
            if method == 'pure_llm':
                context = ''
            else:
                try:
                    context = contexts[retrieval_method][case['case_id']]
                except KeyError as exc:
                    raise ValueError(f'Missing {retrieval_method} context for {case["case_id"]}') from exc
            item = task_contract.decorate_item(preparation._llm_item(case, context, 0, proposal))
            task_contract.answer_free(item)
            if method == 'pure_llm' and case['case_type'] == 'pair_snapshot':
                task_contract.assert_pure_pair(item)
            prompt = f'Method: {method}. Independent repetition: {run_id}.\nClassify every input item. Use null for fields that do not apply.\n' + json.dumps({'inputs': [item]}, ensure_ascii=False)
            identity = dict(model=task_contract.MODEL, temperature=0., thinking='disabled',
                            max_tokens=1024 if method == 'full_system' else 2048,
                            system=task_contract.system_prompt(preparation, method), prompt=prompt,
                            prompt_version=task_contract.VERSION)
            jobs.append(make_job(case, method, 'classification', identity, run_id))
    return rules, jobs


def make_job(case, method, stage, identity, run_id):
    request_hash = digest(identity)
    return dict(case_id=case['case_id'], dataset_id=case['dataset_id'], method=method,
                stage=stage, run_id=run_id, identity=identity, request_sha256=request_hash,
                call_id=digest(dict(run_id=run_id, request_sha256=request_hash)))


def read_receipt(job, responder):
    receipt = responder(job)
    if receipt.get('identity') != job['identity'] or receipt.get('finish_reason') != 'stop':
        raise ValueError('Receipt identity mismatch or incomplete response')
    if receipt.get('run_id', 1) != job['run_id']:
        raise ValueError('Receipt belongs to another repetition')
    raw = parsed_response(receipt['response'], job['stage'])
    return receipt, raw


def normalize_classifications(cases, rules, jobs, responder, *, run_id=1, methods=METHODS, registry_dir=RESOURCES):
    """Normalize supplied receipts and apply the frozen audited override contract."""
    tumor_ids, radio_ids, _, _ = registries(Path(registry_dir))
    case_index = {c['case_id']: c for c in cases}
    rule_index = {p['case_id']: deepcopy(p) for p in rules}
    proposals = arbiter._full_system_proposals(cases, rule_rows=rules)
    results = {m: {} for m in methods}
    if 'full_system' in results:
        results['full_system'] = deepcopy(rule_index)
    for job in jobs:
        method, cid = job['method'], job['case_id']
        case = case_index[cid]
        receipt, raw = read_receipt(job, responder)
        raw = raw['predictions'][0]
        item = json.loads(job['identity']['prompt'].split('\n')[-1])['inputs'][0]
        proposal = proposals[case['dataset_id'], cid] if method == 'full_system' else None
        if proposal is not None and item['rule_proposal'] != proposal:
            raise ValueError('Classification proposal differs from the frozen merge contract')
        row = arbiter._normalize_llm_prediction(case, raw, method=method, run_id=run_id,
                  parse_status='fresh_parsed_attempt_1', tumor_ids=tumor_ids, radio_ids=radio_ids,
                  context_chars=len(item.get('retrieved_context', '')), rule_proposal=proposal)
        if method == 'full_system':
            row = arbiter._merge_full_system_arbiter(rule_row=rule_index[cid], normalized_row=row,
                  raw=raw, context=item.get('retrieved_context', ''), parse_status='fresh_parsed_attempt_1', proposal=proposal)
        row['prediction_provenance'].update(prompt_version=job['identity']['prompt_version'],
                  model=task_contract.MODEL, provider_model=receipt.get('provider_model'),
                  request_sha256=job['request_sha256'], current_model_response=True,
                  repetition_call_id=job.get('call_id'), cross_run_reuse=False)
        results[method][cid] = row
    return {m: [index[c['case_id']] for c in cases] for m, index in results.items()}


def finish_full_system(predictions, cases, facts, claim_contexts, responder, *, run_id=1, registry_dir=RESOURCES):
    """Apply the final source-ownership and maturity layers to valid pairs only."""
    _, _, tumors, aliases = registries(Path(registry_dir))
    case_index = {c['case_id']: c for c in cases}
    required = required_claim_case_ids(predictions)
    output = []
    for row in predictions:
        if row['case_id'] not in required:
            output.append(deepcopy(row))
            continue
        case = case_index[row['case_id']]
        context, _ = pair_context(case, facts, claim_contexts.get(case['case_id'], ''))
        identity, sources = conclusion_request(case, context, aliases)
        identity['model'] = task_contract.MODEL
        job = make_job(case, 'full_system', 'scoped_claims', identity, run_id)
        receipt, raw = read_receipt(job, responder)
        result = apply_conclusion_only(row, raw, case, facts, sources, job['request_sha256'])
        result = apply_ownership(result, raw, case, facts, sources, job['request_sha256'], tumors)
        result = candidate(result, case, facts, tumors, ownership=True)
        result['prediction_provenance'].update(component_experiment='V1.3.2_frozen_repetitions',
                    fresh_claim_provider_model=receipt.get('provider_model'), claim_repetition_call_id=job['call_id'])
        output.append(result)
    return output


def run(cases, rule_facts, contexts, claim_facts, claim_contexts, responder: Callable, *, run_id=1, methods=METHODS, registry_dir=RESOURCES):
    """Run one dataset/repetition with explicit, answer-free evidence inputs."""
    validate_inputs(cases, claim_facts, claim_contexts)
    rules, jobs = prepare_jobs(cases, rule_facts, contexts, run_id=run_id, methods=methods)
    output = normalize_classifications(cases, rules, jobs, responder, run_id=run_id, methods=methods, registry_dir=registry_dir)
    if 'full_system' in output:
        output['full_system'] = finish_full_system(output['full_system'], cases, claim_facts, claim_contexts, responder, run_id=run_id, registry_dir=registry_dir)
    return [p for method in methods for p in output[method]]
