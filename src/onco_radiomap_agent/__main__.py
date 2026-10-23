"""Command line interface; offline unless --live is explicitly selected."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from . import __version__
from .engine import METHODS, RESOURCES, prepare_jobs, run


def read(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(s) for s in path.read_text('utf-8-sig').splitlines() if s.strip()]
    return json.loads(path.read_text('utf-8-sig'))


def write(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        if jsonl:
            for row in value:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        else:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_bundle(path):
    """Resolve six explicit evidence inputs, relative to the bundle manifest."""
    path = Path(path).resolve()
    cfg = read(path)
    names = ['cases','rule_facts','naive_contexts','lightrag_contexts','claim_facts','claim_contexts']
    data = {k: read(path.parent / cfg[k]) for k in names}
    for k in ['naive_contexts','lightrag_contexts','claim_contexts']:
        if isinstance(data[k], list):
            data[k] = {r['case_id']:r['context'] for r in data[k]}
    return data


class ReceiptDirectory:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __call__(self, job):
        return read(self.directory / (job['call_id'] + '.json'))


class LiveResponder:
    """One attempt per call; preserve responses; enforce a caller-supplied count cap."""
    def __init__(self, directory, max_calls):
        if not max_calls or max_calls < 1:
            raise ValueError('--live requires --max-calls > 0')
        from openai import OpenAI
        key = os.environ.get('DEEPSEEK_API_KEY')
        if not key:
            raise ValueError('DEEPSEEK_API_KEY is required')
        self.client = OpenAI(api_key=key, base_url=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com/v1'),
                             max_retries=0, timeout=90.)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_calls, self.calls = max_calls, 0

    def __call__(self, job):
        if self.calls >= self.max_calls:
            raise RuntimeError('Call limit reached; no automatic resume')
        self.calls += 1
        i = job['identity']
        response = self.client.chat.completions.create(model=i['model'], temperature=i['temperature'],
             max_tokens=i['max_tokens'], messages=[{'role':'system','content':i['system']},{'role':'user','content':i['prompt']}],
             response_format={'type':'json_object'}, extra_body={'thinking':{'type':'disabled'}})
        receipt = dict(identity=i, run_id=job['run_id'], request_sha256=job['request_sha256'], call_id=job['call_id'],
             response=response.choices[0].message.content or '', provider_model=response.model,
             finish_reason=response.choices[0].finish_reason, usage=response.usage.model_dump() if response.usage else {})
        write(self.directory / (job['call_id']+'.json'), receipt)
        return receipt


def main():
    if not __debug__:
        raise RuntimeError('Do not run frozen research checks with python -O')
    parser = argparse.ArgumentParser(description='Onco-RadioMap V1.3.2 research code')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ['prepare','predict']:
        p = sub.add_parser(cmd)
        p.add_argument('--bundle', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--run-id', type=int, default=1)
        p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
        if cmd == 'predict':
            p.add_argument('--registries', type=Path, default=RESOURCES)
            mode = p.add_mutually_exclusive_group(required=True)
            mode.add_argument('--receipts', type=Path, help='Offline directory keyed by repetition call_id')
            mode.add_argument('--live', action='store_true', help='Call the configured provider')
            p.add_argument('--save-receipts', type=Path)
            p.add_argument('--max-calls', type=int)
    p = sub.add_parser('evaluate')
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roles',type=Path,help='JSON mapping dataset ID to dataset role')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    if args.command == 'evaluate':
        from .evaluation import evaluate_predictions
        value = evaluate_predictions(read(args.cases),read(args.predictions),dataset_roles=read(args.roles) if args.roles else None)
        write(args.output,value)
    else:
        b=load_bundle(args.bundle)
        contexts={'naive_rag':b['naive_contexts'],'lightrag':b['lightrag_contexts']}
        if args.command=='prepare':
            rules,jobs=prepare_jobs(b['cases'],b['rule_facts'],contexts,run_id=args.run_id,methods=args.methods)
            write(args.output,dict(rules=rules,classification_jobs=jobs))
        else:
            if args.live and args.save_receipts is None:
                parser.error('--live requires --save-receipts pointing to a new directory')
            responder=LiveResponder(args.save_receipts,args.max_calls) if args.live else ReceiptDirectory(args.receipts)
            value=run(b['cases'],b['rule_facts'],contexts,b['claim_facts'],b['claim_contexts'],responder,
                       run_id=args.run_id,methods=args.methods,registry_dir=args.registries)
            write(args.output,value,jsonl=True)
    print(str(args.output.resolve()))


if __name__=='__main__':main()
