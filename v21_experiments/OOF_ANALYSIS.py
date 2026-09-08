"""Post-hoc OOF analysis. NumPy + pandas only; no training or image reads."""
from pathlib import Path
import argparse, json, math, hashlib
import numpy as np
import pandas as pd

METRICS = ['accuracy','balanced_accuracy','macro_f1']
def metrics(cm):
    cm=np.asarray(cm,dtype=float); tp=np.diagonal(cm,axis1=-2,axis2=-1)
    actual=cm.sum(axis=-1); pred=cm.sum(axis=-2)
    recall=np.divide(tp,actual,out=np.zeros_like(tp),where=actual>0)
    f1=np.divide(2*tp,actual+pred,out=np.zeros_like(tp),where=(actual+pred)>0)
    return np.stack([tp.sum(axis=-1)/cm.sum(axis=(-2,-1)),recall.mean(axis=-1),f1.mean(axis=-1)],axis=-1)

def exact_mcnemar(y,a,b):
    a=np.asarray(a)==y;b=np.asarray(b)==y
    ab=int((a & ~b).sum()); ba=int((~a & b).sum()); n=ab+ba
    if not n:p=1.
    else:
        k=min(ab,ba)
        logs=np.array([math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)])
        hi=float(logs.max());p=min(1.,2*math.exp(hi)*float(np.exp(logs-hi).sum()))
    return dict(a_correct_b_wrong=ab,a_wrong_b_correct=ba,discordant=n,p_exact_two_sided=p)

def validate(df):
    required={'fold','method','path','patient_id','y_true','y_pred'}
    if not required.issubset(df):raise ValueError('Required columns: '+str(sorted(required)))
    if df[list(required)].isna().any().any():raise ValueError('Missing OOF identifiers or labels')
    for c in ['patient_id','path','method']:
        df[c]=df[c].astype(str)
        if df[c].str.strip().eq('').any():raise ValueError('Empty '+c)
    for c in ['fold','y_true','y_pred']:
        v=pd.to_numeric(df[c],errors='raise')
        if not np.equal(v,np.floor(v)).all():raise ValueError('Noninteger '+c)
        df[c]=v.astype(int)
    if df.duplicated(['method','path']).any():raise ValueError('Repeated method/path records (not a content-dedup audit)')
    for c in ['fold','y_true']:
        if df.groupby('patient_id')[c].nunique().max()!=1:raise ValueError('An identifier crosses folds or class labels')
    if df.groupby('path')['patient_id'].nunique().max()!=1:raise ValueError('Path assigned to multiple IDs')
    labels=sorted(df.y_true.unique().tolist())
    if labels!=list(range(len(labels))):raise ValueError('Expected zero-based class codes')
    if not set(df.y_pred).issubset(labels):raise ValueError('Unknown predicted label')
    methods=list(dict.fromkeys(df.method)); ref=df[df.method==methods[0]].sort_values('path')
    for m in methods:
        r=df[df.method==m].sort_values('path')
        if not r[['path','patient_id','fold','y_true']].reset_index(drop=True).equals(ref[['path','patient_id','fold','y_true']].reset_index(drop=True)):
            raise ValueError('Methods do not contain exactly aligned cases: '+m)
    return df,methods,len(labels)

def run(a):
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory; no overwrites')
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.oof,dtype={'patient_id':str,'path':str,'method':str})
    df,methods,k=validate(df)
    if a.a not in methods or a.b not in methods:raise ValueError('Comparison methods absent: '+str(methods))
    if a.resamples<100:raise ValueError('Use at least 100 resamples')
    ids=sorted(df.patient_id.unique()); idmap={x:i for i,x in enumerate(ids)}
    ref=df[df.method==methods[0]].sort_values('path')
    truth=ref.y_true.to_numpy(); gi=ref.patient_id.map(idmap).to_numpy()
    idtruth=ref.groupby('patient_id').y_true.first().reindex(ids).to_numpy()
    rng=np.random.default_rng(a.seed); weights=np.zeros((a.resamples,len(ids)),dtype=int)
    for c in range(k):
        ix=np.where(idtruth==c)[0]
        weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=a.resamples)
    pcols=['p_'+str(c) for c in range(k)];pools=['majority_vote']
    if any(c.startswith('p_') for c in df.columns):
        if not set(pcols).issubset(df):raise ValueError('Incomplete probability columns')
        pp=df[pcols].to_numpy(float)
        if not np.isfinite(pp).all() or (pp<0).any() or not np.allclose(pp.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
        if not np.allclose(pp[np.arange(len(df)),df.y_pred],pp.max(1),atol=1e-6):raise ValueError('Stored labels disagree with probability argmax')
        pools.append('mean_probability')
    modes=['slice_clustered']+pools;cubes={};boot={};preds={};summaries=[];patientrows=[];foldrows=[]
    for m in methods:
        r=df[df.method==m].sort_values('path');pred=r.y_pred.to_numpy();preds[m]=pred
        cube=np.zeros((len(ids),k,k),dtype=int);np.add.at(cube,(gi,truth,pred),1);cubes[(m,'slice_clustered')]=cube
        for pool in pools:
            scores=cube.sum(axis=1) if pool=='majority_vote' else r.groupby('patient_id')[pcols].mean().reindex(ids).to_numpy()
            winners=scores.argmax(axis=1);ties=np.sum(np.isclose(scores,scores.max(1,keepdims=True),atol=1e-12,rtol=0),axis=1)>1
            pc=np.zeros_like(cube);pc[np.arange(len(ids)),idtruth,winners]=1;cubes[(m,pool)]=pc
            for i,pid in enumerate(ids):patientrows.append(dict(method=m,pooling=pool,patient_id=pid,fold=int(ref[ref.patient_id==pid].fold.iloc[0]),y_true=int(idtruth[i]),y_pred=int(winners[i]),slices=int(cube[i].sum()),tie=bool(ties[i])))
        for mode in modes:
            cc=cubes[(m,mode)];point=metrics(cc.sum(0));bs=metrics((weights@cc.reshape(len(ids),-1)).reshape(a.resamples,k,k));boot[(m,mode)]=bs
            for j,metric in enumerate(METRICS):
                lo,hi=np.quantile(bs[:,j],[.025,.975]);summaries.append(dict(method=m,unit=mode,metric=metric,estimate=float(point[j]),ci95_low=float(lo),ci95_high=float(hi),identifiers=len(ids),slices=len(ref)))
        for f in sorted(ref.fold.unique()):
            mask=ref.fold.to_numpy()==f;cm=np.zeros((k,k),int);np.add.at(cm,(truth[mask],pred[mask]),1);foldrows.append(dict(method=m,fold=int(f),**dict(zip(METRICS,metrics(cm)))))
    comparisons=[]
    for mode in modes:
        delta=boot[(a.a,mode)]-boot[(a.b,mode)];observed=metrics(cubes[(a.a,mode)].sum(0))-metrics(cubes[(a.b,mode)].sum(0))
        for j,metric in enumerate(METRICS):
            lo,hi=np.quantile(delta[:,j],[.025,.975]);comparisons.append(dict(comparison=a.a+' minus '+a.b,unit=mode,metric=metric,difference=float(observed[j]),ci95_low=float(lo),ci95_high=float(hi)))
    tests=[dict(unit='slice_IID_DIAGNOSTIC_ONLY',**exact_mcnemar(truth,preds[a.a],preds[a.b]))]
    for pool in pools:
        pa=cubes[(a.a,pool)].sum(1).argmax(1);pb=cubes[(a.b,pool)].sum(1).argmax(1);tests.append(dict(unit='identifier_'+pool,**exact_mcnemar(idtruth,pa,pb)))
    for name,rows_ in [('metric_intervals',summaries),('paired_differences',comparisons),('mcnemar',tests),('patient_predictions',patientrows),('fold_metrics',foldrows)]:pd.DataFrame(rows_).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame(foldrows).groupby('method')[METRICS].agg(['mean','std']).to_csv(out/'fold_mean_sd.csv')
    info=dict(methods=methods,slices=len(ref),identifiers=len(ids),resamples=a.resamples,seed=a.seed,classes=k,comparison=[a.a,a.b],input_sha256=hashlib.sha256(Path(a.oof).read_bytes()).hexdigest(),mean_probability_available='mean_probability' in pools,interpretation=['POST-HOC exploratory analysis; no new independent cohort.','Values are fractions, not percentages. CI for pooled metrics is NOT CI for fold mean.','Bootstrap: class-stratified released-ID clusters; pointwise percentile 95% intervals. Pairwise draws shared across methods.','Slice McNemar violates within-ID independence: diagnostic only, not evidence of patient-level significance.','Identifier aggregation uses deterministic lowest-class-index tie break, with ties explicitly counted. Do not select the best pooling rule after viewing results.','Patient means released identifier, not verified distinct natural person. Outer models share training cases: these intervals condition on fitted OOF predictions and omit retraining/selection variability.','Single focal pair only; no unadjusted all-pairs testing. Do not infer superiority across all metrics from one p-value.'])
    (out/'analysis.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('COMPLETE:',out,'|',len(ids),'IDs,',len(ref),'slices,',len(methods),'methods; mean probabilities:',info['mean_probability_available'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oof',required=True);p.add_argument('--out',required=True);p.add_argument('--a',default='proposed');p.add_argument('--b',default='linear_probe');p.add_argument('--resamples',type=int,default=2000);p.add_argument('--seed',type=int,default=42);run(p.parse_args())
if __name__=='__main__':main()
