import math

from collections import defaultdict

import numpy as np

from .expert_metrics import LABELS

def stats(v):
    good=[x for x in v if x is not None and math.isfinite(x)]
    return dict(mean=float(np.mean(good)) if good else None,sd=float(np.std(good,ddof=1)) if len(good)>1 else (0. if good else None),defined_runs=len(good),total_runs=len(v),per_run=v)

def cluster_ci(matched):
    """Resample unique cases, carrying all five predictions for each case."""
    cases=[r for r in matched if r['run_id']==1];n=len(cases)
    ix={(r['case_id'],r['run_id']):r for r in matched};assert len(ix)==n*5
    strata=defaultdict(list)
    for i,c in enumerate(cases):strata[c['dataset_id'],c['clinical_use']].append(i)
    rng=np.random.default_rng(20260920);weights=np.zeros((2000,n),int)
    for ids in strata.values():weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    valid=np.zeros((5,n));mc=valid.copy();cc=valid.copy();pm=np.zeros((5,n),int);pc=pm.copy()
    ym=np.array([int(c['final_maturity'][1:])-1 for c in cases]);yc=np.array([LABELS.index(c['final_conclusion']) for c in cases])
    for r in range(5):
        for i,c in enumerate(cases):
            p=ix[c['case_id'],r+1];valid[r,i]=p['prediction_status']=='valid'
            if valid[r,i]:
                pm[r,i]=int(p['model_maturity'][1:])-1;pc[r,i]=LABELS.index(p['model_conclusion'])
                mc[r,i]=pm[r,i]==ym[i];cc[r,i]=pc[r,i]==yc[i]
    result={}
    for k,a in [('coverage',valid),('maturity_agreement_all',mc),('conclusion_agreement_all',cc)]:
        v=np.mean(np.einsum('bn,rn->br',weights,a)/n,axis=1)
        result[k]=dict(ci95=np.quantile(v,[.025,.975]).tolist(),defined_draws=2000)
    for task,use in [('conclusion',None),('diagnostic','diagnostic'),('therapeutic','therapeutic')]:
        selected=np.array([use is None or c['clinical_use']==use for c in cases]);truth=yc if use is None else ym;pred=pc if use is None else pm
        matrices=np.zeros((5,n,16))
        for r in range(5):
            ids=np.flatnonzero((valid[r]>0)&selected);matrices[r,ids,truth[ids]*4+pred[r,ids]]=1
        cms=np.einsum('bn,rnk->brk',weights,matrices).reshape(2000,5,4,4)
        w=1-np.eye(4) if use is None else (np.arange(4)[:,None]-np.arange(4)[None,:])**2
        with np.errstate(divide='ignore',invalid='ignore'):
            expected=cms.sum(-1)[..., :,None]*cms.sum(-2)[...,None,:]/cms.sum((-2,-1))[...,None,None]
            den=(expected*w).sum((-2,-1));k=np.where(den>0,1-(cms*w).sum((-2,-1))/den,np.nan)
            defined=np.isfinite(k).sum(1);v=np.nansum(k,axis=1)/defined
        good=v[np.isfinite(v)]
        result[task+'_kappa']=dict(ci95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,defined_draws=len(good),all_five_defined_draws=int((defined==5).sum()))
    return result
