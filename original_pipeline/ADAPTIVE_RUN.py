"""Separated validation development and final evaluation for the supplied article2.
Run from the project root. Never invokes RUN_ALL or the old ablation scripts.
Supports dynamic folder/CSV manifests through ADAPTIVE_DATA. No imbalance or dedup stage.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REQUIRED = ['code_final_fixed_v5.py', 'code_final_fixed_v4.py', 'splits_v5.py', 'preprocess_v5.py', 'ADAPTIVE_DATA.py', 'ADAPTIVE_ADAPTER.py', 'ADAPTIVE_PREPROCESS.py']

class ProtocolError(RuntimeError): pass

def clean(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    return x

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def verify(files):
    for path, expected in files.items():
        if not Path(path).is_file() or digest(path)!=expected:
            raise ProtocolError('Frozen file changed/missing: '+path)

def guard(role, final=False):
    if role not in ('train','valid','test'): raise ProtocolError('Unknown role: '+str(role))
    if role=='test' and not final: raise ProtocolError('Test feature access blocked during development.')

def env():
    from importlib.metadata import version, PackageNotFoundError
    result={'python':sys.version}
    for name in ('numpy','scipy','scikit-learn','torch','torchvision','xgboost','lightgbm','catboost'):
        try: result[name]=version(name)
        except PackageNotFoundError: result[name]=None
    return result

def check():
    import ast
    missing=[f for f in REQUIRED if not (ROOT/f).is_file()]
    if missing: raise ProtocolError('Missing original files: '+', '.join(missing))
    for f in REQUIRED+['ADAPTIVE_RUN.py']:
        ast.parse((ROOT/f).read_text(encoding='utf-8-sig'), filename=f)
    print('Sources exist and parse. Not an end-to-end training test.')

def load_modules(run, final=False):
    check()
    import numpy as np
    import code_final_fixed_v5 as V
    from ADAPTIVE_ADAPTER import install
    install(V,run,final)
    return np,V

def cfg_for(V, bed, folder, split_dir, pca, c):
    return V.make_cfg(bed,run_dir=str(folder),split_dir=str(split_dir),
        pca_var=pca,svm_c=c,tta=1,baseline_tta=1,decision_rule='argmax',
        calibrate=False,use_hard_weighting=False,hard_weight_gamma=0.0,
        hard_weight_fn_penalty=1.0,refit_on_train_valid=False,
        strict_tta=True,prefer_existing_caches=False,extract_missing=True,
        allow_partial_baselines=False,bootstrap_n=2000)

def audit_groups(V,cfg):
    import itertools
    bed=V.load_bed(cfg.bed,cfg.split_dir,cfg.normal_class)
    for a,b in itertools.combinations(('train','valid','test'),2):
        if set(bed.split(a).files)&set(bed.split(b).files):
            raise ProtocolError('Shared file names across '+a+'/'+b)
        if set(map(str,bed.groups(a)))&set(map(str,bed.groups(b))):
            raise ProtocolError('Shared group identifiers across '+a+'/'+b)
    return bed

def head_grid(bl,cs):
    kw=dict(bl.head_kwargs)
    if bl.head=='svm': return [dict(kw,C=c,gamma='scale',class_weight='balanced') for c in cs]
    if bl.head=='knn': return [dict(kw,n_neighbors=k) for k in (3,5,9)]
    if bl.head=='rf': return [dict(kw,min_samples_leaf=k) for k in (1,3,5)]
    if bl.head=='logreg': return [dict(kw,C=c,class_weight='balanced') for c in cs]
    return [kw]

def develop(a,run):
    if run.exists() and any(run.iterdir()): raise ProtocolError('Run directory must be new/empty. Preserve old runs.')
    np,V=load_modules(run)
    import ADAPTIVE_DATA as D
    D.check_content(D.load((ROOT/a.manifest).resolve()),('train','valid'))
    run.mkdir(parents=True,exist_ok=True)
    cs=[0.1,1.5,10.0]; pcas=[0.0,0.95]
    trials=[]; best=None
    for i,(pca,c) in enumerate((p,c) for p in pcas for c in cs):
        cfg=cfg_for(V,'dataset',run/('candidate_%02d'%i),(ROOT/a.manifest).resolve(),pca,c)
        bed=audit_groups(V,cfg); store=V.FeatureStore(cfg,bed)
        led=V.S.SelectionLedger(cfg.run_dir,strict=True)
        res=V.select_on_validation(cfg,bed,store,led)
        expected=len(cfg.specs)*(len(cfg.learners)+1)
        if len(res.grid)!=expected: raise ProtocolError('One or more candidate learners failed; refuse incomplete selection.')
        V.save_selection(cfg,res)
        V.write_csv(str(Path(cfg.run_dir)/'selection_grid.csv'),
                    [{k:v for k,v in r.items() if k!='rule_scores'} for r in res.grid])
        row={'config':cfg.to_json(),'selection':res.to_json(),
             'validation_score':float(res.valid_scores['valid_bal_acc'])}
        trials.append(row)
        if best is None or row['validation_score']>best['validation_score']: best=row
    cfg=V.Cfg5(**best['config']); bed=audit_groups(V,cfg); store=V.FeatureStore(cfg,bed)
    chosen={}; baseline_rows=[]
    for key,bl in V.BASELINES.items():
        X,y=store.spec_features('train',bl.spec,1)
        Xv,yv=store.spec_features('valid',bl.spec,1)
        rep,A=V.fit_representation(X,bl.standardize,bl.pca_var,dims=V.spec_dims(bl.spec),seed=cfg.random_state)
        B=rep.transform(Xv); candidates=[]
        for kw in head_grid(bl,cs):
            probs,secs=V.fit_baseline_head(bl.head,A,y,B,bed.n_classes,cfg.random_state,kw,cfg,X_va=B,y_va=yv)
            score=float(V.score_block(bed,yv,probs.argmax(1))['bal_acc'])
            row={'key':key,'kwargs':kw,'valid_bal_acc':score,'fit_seconds':float(secs)}
            candidates.append(row); baseline_rows.append(row)
            print('VALID baseline',key,score,flush=True)
        chosen[key]=max(candidates,key=lambda r:r['valid_bal_acc'])
    write(run/'development_trials.json',trials)
    V.write_csv(str(run/'baseline_validation_grid.csv'),baseline_rows)
    write(run/'frozen_selection.json',{'proposed':best,'baselines':chosen,
        'manifest_digests':{r:bed.split(r).digest() for r in ('train','valid','test')},
        'protocol':{'tta':1,'decision_rule':'argmax','confidence_weighting':False,
                    'svm_C_grid':cs,'proposed_PCA_grid':pcas,
                    'selection_metric':'slice-level validation balanced accuracy',
                    'baseline_status':'adaptations/controls, not verified published-method reproductions'}})
    files=[ROOT/f for f in REQUIRED+['ADAPTIVE_RUN.py']]
    files+=list(Path(cfg.split_dir).glob('*.json'))
    files+=[p for p in run.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npy')]
    write(run/'freeze.json',{'files':{str(p.resolve()):digest(p) for p in files},'environment':env(),
                           'warning':'No test features evaluated through this runner. Historical independence is not established.'})
    print('DEVELOPMENT COMPLETE. Review cohort history before any final evaluation. Outputs:',run)

def holm(ps):
    result=[0.0]*len(ps); prior=0.0
    for rank,i in enumerate(sorted(range(len(ps)),key=lambda i:ps[i])):
        prior=max(prior,min(1.0,(len(ps)-rank)*ps[i])); result[i]=prior
    return result

def paired(y,a,b,groups,n=2000):
    import numpy as np
    y,a,b,groups=map(np.asarray,(y,a,b,groups))
    if not(len(y)==len(a)==len(b)==len(groups)) or not len(y): raise ProtocolError('Misaligned predictions.')
    u,ix=np.unique(groups,return_inverse=True)
    if len(u)<2: raise ProtocolError('Fewer than two independent groups.')
    delta=(a==y).astype(float)-(b==y).astype(float)
    sums=np.bincount(ix,weights=delta); sizes=np.bincount(ix); rng=np.random.default_rng(42)
    boots=[]; extreme=0
    for _ in range(n):
        draw=rng.integers(0,len(u),len(u)); boots.append(float(sums[draw].sum()/sizes[draw].sum()))
        null=(sums*rng.choice([-1,1],len(u))).sum()
        extreme+=abs(null)>=abs(sums.sum())-1e-12
    lo,hi=np.quantile(boots,[.025,.975])
    return {'delta_accuracy':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),
            'p_raw':float((extreme+1)/(n+1)),'n_groups':int(len(u))}

def final(a,run):
    freeze=read(run/'freeze.json'); verify(freeze['files'])
    if env()!=freeze['environment']: raise ProtocolError('Package/Python versions changed after freeze.')
    if not a.cohort_history.strip(): raise ProtocolError('Cohort-history statement required.')
    np,V=load_modules(run,True); choice=read(run/'frozen_selection.json')
    cfg=V.Cfg5(**choice['proposed']['config']); bed=audit_groups(V,cfg)
    import ADAPTIVE_DATA as D
    D.check_content(D.load(cfg.split_dir),('train','valid','test'))
    for role,d in choice['manifest_digests'].items():
        if bed.split(role).digest()!=d: raise ProtocolError('Manifest changed after freeze.')
    # Exclusive file creation: a crashed final is not silently restarted.
    with (run/'FINAL_STARTED.json').open('x',encoding='utf-8') as f:
        json.dump({'mode':a.mode,'cohort_history':a.cohort_history,'time':time.time(),
                   'warning':'Keep this record after failure. Do not delete it to reset history.'},f,indent=2)
    res=V.SelectionResult(**choice['proposed']['selection'])
    store=V.FeatureStore(cfg,bed)
    # A new ledger in final_records records this evaluator separately from old v5 runs.
    # FINAL_STARTED remains the guard for retries; neither ledger proves historical independence.
    led=V.S.SelectionLedger(str(run/'final_records'),strict=True)
    led.record('frozen_selection_sha256',digest(run/'frozen_selection.json'),evidence='Development-only artifact')
    led.freeze(note='All proposed and baseline choices were frozen before this evaluation.')
    primary=V.report_on_test(cfg,bed,store,led,res)
    # Extra rules calculated by the old reporter are not used for selection or exported as findings.
    primary.pop('rules_on_test',None)
    if 'ci_cluster' in primary: primary['ci_cluster']['unit']=bed.group_unit
    write(run/'partial_proposed.json',V.S._jsonable(primary))
    for key,selection in choice['baselines'].items():
        V.BASELINES[key]=copy.deepcopy(V.BASELINES[key])
        V.BASELINES[key].head_kwargs=selection['kwargs']; V.BASELINES[key].tta=1
        V.BASELINES[key].reported=None
        V.BASELINES[key].notes='Validation-tuned adaptation/control. Published-method reproduction NOT verified.'
    rows=V.run_baselines(cfg,bed,store,led,res,keys=list(choice['baselines']))
    if any(r.get('status')!='ok' for r in rows): raise ProtocolError('Incomplete baseline evaluation. Preserve partial artifacts.')
    comps=[]; y=np.asarray(primary['slice_predictions']['y_true']); pred=np.asarray(primary['slice_predictions']['y_pred'])
    groups=np.asarray(bed.groups('test'))
    from sklearn.metrics import confusion_matrix
    primary['confusion_matrix']=confusion_matrix(y,pred,labels=range(bed.n_classes)).tolist()
    for row in rows:
        row.pop('under_sealed_rule',None)
        if 'ci_cluster' in row: row['ci_cluster']['unit']=bed.group_unit
        yb=np.asarray(row['slice_predictions']['y_true']); pb=np.asarray(row['slice_predictions']['y_pred'])
        if not np.array_equal(y,yb): raise ProtocolError('Baseline labels not aligned.')
        row['confusion_matrix']=confusion_matrix(y,pb,labels=range(bed.n_classes)).tolist()
        row['slice_predictions']['files']=bed.split('test').files
        row['slice_predictions']['groups']=[str(g) for g in groups]
        comps.append(dict(comparison='proposed minus '+row['key'],**paired(y,pred,pb,groups,cfg.bootstrap_n)))
    for r,p in zip(comps,holm([r['p_raw'] for r in comps])): r['p_holm']=p
    # Preserve predeclared order; do not rank by test accuracy.
    compact=[{'method':'proposed',**{k:primary[k] for k in ('test_acc','test_bal_acc','test_f1')}}]
    compact += [{'method':r['key'],**{k:r[k] for k in ('test_acc','test_bal_acc','test_f1')}} for r in rows]
    V.write_csv(str(run/'results_final.csv'),compact)
    V.write_csv(str(run/'comparisons_paired.csv'),comps)
    write(run/'final_results.json',V.S._jsonable({'mode':a.mode,'cohort_history':a.cohort_history,
        'group_unit':bed.group_unit,'class_names':list(bed.class_names),'proposed':primary,
        'baselines':rows,'comparisons':comps,
        'caveats':['Independent group IDs must be validated; image hashes do not prove patient independence.',
                   'Baselines are adaptations, not verified reproductions.',
                   'Paired CI is pointwise; Holm adjusts only the predeclared p-value family.',
                   'Historical independence is not established by a mode flag.']}))
    print('FINAL COMPLETE:',run)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    sub.add_parser('check')
    q=sub.add_parser('prepare',help='Inspect raster files and create a dynamic manifest; no training.')
    q.add_argument('--data',required=True); q.add_argument('--csv',default=None)
    q.add_argument('--layout',choices=['auto','flat','split'],default='auto')
    q.add_argument('--out',required=True); q.add_argument('--val-fraction',type=float,default=.2)
    q.add_argument('--test-fraction',type=float,default=.2); q.add_argument('--seed',type=int,default=42)
    q.add_argument('--normal-class',default=None)
    d=sub.add_parser('develop'); d.add_argument('--manifest',required=True); d.add_argument('--run',required=True)
    f=sub.add_parser('final'); f.add_argument('--run',required=True)
    f.add_argument('--mode',choices=['exploratory','independent'],required=True)
    f.add_argument('--cohort-history',required=True)
    a=p.parse_args()
    if a.stage=='check': check(); return
    if a.stage=='prepare':
        from ADAPTIVE_DATA import prepare
        prepare(a.data,(ROOT/a.out).resolve(),a.csv,a.layout,a.val_fraction,a.test_fraction,a.seed,a.normal_class)
        return
    run=(ROOT/a.run).resolve()
    if a.stage=='develop': develop(a,run)
    else: final(a,run)

if __name__=='__main__':
    try: main()
    except Exception:
        import traceback; traceback.print_exc()
        print('STOP. Preserve all artifacts and logs; do not reset test history.',file=sys.stderr)
        sys.exit(1)
