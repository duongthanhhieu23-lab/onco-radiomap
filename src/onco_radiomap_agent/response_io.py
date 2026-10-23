import hashlib

import json

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def parsed_response(text,stage):
    value=json.loads(text)
    if stage=='scoped_claims':
        if not isinstance(value,dict) or not isinstance(value.get('claims'),list) or len(value['claims'])>6:
            raise ValueError('claims_envelope_invalid')
    else:
        if not isinstance(value,dict) or not isinstance(value.get('predictions'),list) or len(value['predictions'])!=1:
            raise ValueError('single_prediction_envelope_invalid')
        p=value['predictions'][0]
        if not isinstance(p,dict) or type(p.get('i')) is not int or p['i']!=0:
            raise ValueError('prediction_index_invalid')
    return value
