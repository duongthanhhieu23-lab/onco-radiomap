import math, warnings

from collections import Counter, defaultdict

import numpy as np

from sklearn.metrics import cohen_kappa_score

LABELS=['supportive','negative','inconclusive','conflicting']

def wilson(k,n):
    if n==0:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-half,mid+half]

def kappa(cm,weight='ordinary'):
    cm=np.asarray(cm,dtype=float);n=cm.sum()
    if not n:return None
    a=np.arange(4);w=1-np.eye(4) if weight=='ordinary' else abs(a[:,None]-a[None,:])**(2 if weight=='quadratic' else 1)
    den=(np.outer(cm.sum(1),cm.sum(0))/n*w).sum()
    return float(1-(cm*w).sum()/den) if den>0 else None

def analyze(matched,seed=20260920,draws=2000):
    n=len(matched);valid=np.array([r['prediction_status']=='valid' for r in matched]);nv=int(valid.sum())
    statuses=Counter(r['prediction_status'] for r in matched)
    assert set(statuses)<={'valid','invalid','abstain'}
    strata=defaultdict(list)
    for i,r in enumerate(matched):strata[r['dataset_id'],r['clinical_use']].append(i)
    rng=np.random.default_rng(seed);boot=np.zeros((draws,n),int)
    for ids in strata.values():boot[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=draws)
    result=dict(n=n,valid=nv,abstain=statuses['abstain'],invalid=statuses['invalid'],coverage=nv/n,coverage_wilson95=wilson(nv,n),tasks={})
    for task,use in [('maturity_all',None),('diagnostic','diagnostic'),('therapeutic','therapeutic'),('conclusion',None)]:
        include=np.array([use is None or r['clinical_use']==use for r in matched]);mask=valid&include
        ids=np.flatnonzero(mask);nt=int(include.sum());v=len(ids)
        y=np.array([LABELS.index(matched[i]['final_conclusion']) if task=='conclusion' else int(matched[i]['final_maturity'][1:])-1 for i in ids])
        p=np.array([LABELS.index(matched[i]['model_conclusion']) if task=='conclusion' else int(matched[i]['model_maturity'][1:])-1 for i in ids])
        exact=int((y==p).sum());cm=np.bincount((y*4+p).astype(int),minlength=16).reshape(4,4)
        rr=dict(n=nt,valid=v,exact=exact,agreement_all=exact/nt if nt else None,agreement_valid=exact/v if v else None,
            agreement_all_wilson95=wilson(exact,nt),agreement_valid_wilson95=wilson(exact,v),confusion_matrix=cm.tolist())
        if task!='conclusion':
            rr.update(within_one=int((abs(y-p)<=1).sum()),two_or_more=int((abs(y-p)>=2).sum()),overgraded=int((p>y).sum()),undergraded=int((p<y).sum()),mean_absolute_error=float(abs(y-p).mean()) if v else None)
        if task!='maturity_all':
            task_mats=np.zeros((n,16));task_mats[ids,(y*4+p).astype(int)]=1
            cms=(boot@task_mats).reshape(draws,4,4);totals=cms.sum((1,2))
            for weight in (['ordinary'] if task=='conclusion' else ['ordinary','linear','quadratic']):
                value=kappa(cm,weight);rr[weight+'_kappa']=value
                if v:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        check=cohen_kappa_score(y,p,labels=[0,1,2,3],weights=None if weight=='ordinary' else weight)
                    assert (value is None and not np.isfinite(check)) or math.isclose(value,check,abs_tol=1e-12)
                a=np.arange(4);w=1-np.eye(4) if weight=='ordinary' else abs(a[:,None]-a[None,:])**(2 if weight=='quadratic' else 1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    expected=cms.sum(2)[:,:,None]*cms.sum(1)[:,None,:]/totals[:,None,None]
                    den=(expected*w).sum((1,2));ks=np.where(den>0,1-(cms*w).sum((1,2))/den,np.nan)
                good=ks[np.isfinite(ks)]
                rr[weight+'_kappa_bootstrap95']=np.quantile(good,[.025,.975]).tolist() if len(good) else None
                rr[weight+'_kappa_defined_draws']=len(good)
        result['tasks'][task]=rr
    return result
