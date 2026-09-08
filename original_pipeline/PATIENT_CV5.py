"""Five patient-disjoint outer folds, one grouped validation holdout per fold.
Not full nested cross-validation. Existing ADAPTIVE_RUN.py supplies all learners.
All five development runs must be sealed before any outer test is evaluated.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
VERSION='patient-cv5-v1'


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def verify(files):
    for p,h in files.items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Changed/missing frozen file: '+p)


def csv_write(path,rows,fields):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def grouped_folds(rows,seed=42,valid_fraction=.2):
    if not 0<valid_fraction<1:raise ValueError('Validation fraction must be between 0 and 1.')
    groups={};seen=set()
    for r in rows:
        g=str(r.get('group') or '').strip()
        if not g:raise ValueError('Every image needs a real patient group ID.')
        if r['path'] in seen:raise ValueError('A path is repeated.')
        seen.add(r['path'])
        if g in groups and groups[g]!=r['label']:raise ValueError('One patient has conflicting class labels.')
        groups[g]=r['label']
    classes=sorted(set(groups.values()))
    if len(classes)<2:raise ValueError('At least two classes required.')
    outer={}
    for ci,c in enumerate(classes):
        ids=sorted(g for g,label in groups.items() if label==c)
        if len(ids)<5:raise ValueError('Each class needs at least five patients: '+c)
        random.Random(seed+1009*ci).shuffle(ids)
        for i,g in enumerate(ids):outer[g]=i%5
    result=[]
    for fold in range(5):
        role={g:'test' if outer[g]==fold else 'train' for g in groups}
        for ci,c in enumerate(classes):
            ids=sorted(g for g in groups if groups[g]==c and role[g]=='train')
            random.Random(seed+100003+fold*1009+ci).shuffle(ids)
            if len(ids)<2:raise ValueError('Not enough development patients for '+c)
            count=max(1,min(len(ids)-1,round(len(ids)*valid_fraction)))
            for g in ids[:count]:role[g]='valid'
        split=[dict(r,group=str(r['group']).strip(),split=role[str(r['group']).strip()]) for r in rows]
        sets={s:{r['group'] for r in split if r['split']==s} for s in ('train','valid','test')}
        assert not(sets['train']&sets['valid'] or sets['train']&sets['test'] or sets['valid']&sets['test'])
        result.append(split)
    for g in groups:assert sum(any(r['group']==g and r['split']=='test' for r in fold) for fold in result)==1
    return result


def source_snapshot():
    import ADAPTIVE_RUN as R
    R.check()
    paths=[ROOT/f for f in R.REQUIRED+['ADAPTIVE_RUN.py','PATIENT_CV5.py']]
    return {str(p.resolve()):sha(p) for p in paths}


def plan(args):
    import ADAPTIVE_DATA as D
    import ADAPTIVE_RUN as R
    data=Path(args.data).resolve();out=Path(args.out).resolve()
    if not data.is_dir():raise ValueError('Data root not found.')
    if out.exists() and any(out.iterdir()):raise ValueError('Plan directory must be new and empty.')
    if (data/'CONVERSION_FAILED.txt').exists():raise ValueError('Input conversion is incomplete; inspect its error record.')
    rows=D.csv_rows(data,args.csv)
    if any(r['split']!='train' for r in rows) and not args.repartition_existing:
        raise ValueError('CSV has existing valid/test assignments. Do not silently repartition an established test. See --repartition-existing.')
    if args.normal_class is not None and args.normal_class not in {r['label'] for r in rows}:
        raise ValueError('Normal class is not present.')
    folds=grouped_folds(rows,args.seed,args.valid_fraction)
    source_files=source_snapshot();out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir();entries=[];reports=[]
    for i,split in enumerate(folds,1):
        name='fold_%02d'%i;folder=out/'manifests'/name;folder.mkdir(parents=True)
        input_csv=out/(name+'_assignments.csv')
        csv_write(input_csv,split,['path','label','split','group'])
        D.prepare(data,folder,csv_path=input_csv,seed=args.seed,normal_class=args.normal_class)
        summary=read(folder/'dataset_summary.json')
        entries.append({'fold':i,'manifest':str(folder),'run':str(out/'runs'/name)})
        for role in ('train','valid','test'):
            reports.append({'fold':i,'split':role,'patients':summary['group_counts'][role],
                            'images':sum(summary['counts'][role].values())})
    csv_write(out/'fold_counts.csv',reports,['fold','split','patients','images'])
    description={'version':VERSION,'folds':entries,'seed':args.seed,'valid_fraction_of_outer_training':args.valid_fraction,
                 'input_rows':len(rows),'patients':len({r['group'] for r in rows}),
                 'classes':sorted({r['label'] for r in rows}),
                 'outer_split':'stratified by patient class; deterministic shuffled round-robin across 5 folds',
                 'inner_split':'one stratified patient validation holdout, NOT inner cross-validation',
                 'selection_metric':'slice-level validation balanced accuracy',
                 'reporting':'Each fold metrics plus unweighted fold mean and sample SD; pooled out-of-fold slice metrics. No average p-values or global significance from dependent folds.',
                 'baseline_status':'existing validation-tuned adaptations/controls; no new paper reproductions',
                 'prior_assignments_repartitioned':bool(args.repartition_existing),
                 'independence_warning':'Patient-disjoint folds do not erase historical development exposure. Results are exploratory internal CV, not external independent validation.'}
    write(out/'plan.json',description)
    source_files.update({str(p.resolve()):sha(p) for p in out.rglob('*') if p.is_file()})
    write(out/'plan_seal.json',{'files':source_files,'environment':R.env()})
    print('PLAN COMPLETE:',out,'\nNo model trained and no test performance evaluated.')


def open_plan(path):
    import ADAPTIVE_RUN as R
    out=Path(path).resolve();s=read(out/'plan_seal.json');verify(s['files'])
    if R.env()!=s['environment']:raise ValueError('Package/Python environment changed since plan creation.')
    return out,read(out/'plan.json')


def child(out,name,arguments):
    log=out/'logs'/(name+'.log')
    if log.exists():raise ValueError('Log already exists; previous attempt must be reviewed: '+str(log))
    cmd=[sys.executable,'-u',str(ROOT/'ADAPTIVE_RUN.py')]+list(arguments)
    print('RUN:',subprocess.list2cmdline(cmd),'\nLog:',log,flush=True)
    # Windows user runs this in the foreground; the assistant never starts this training itself.
    with log.open('x',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);f.write(line);f.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code:raise RuntimeError('Child failed; preserve artifacts and log. No automatic retry: '+str(log))


def develop(args):
    import ADAPTIVE_RUN as R
    out,p=open_plan(args.plan)
    if (out/'CV_FINAL_STARTED.json').exists():raise ValueError('Test evaluation has started. Development cannot be resumed/changed.')
    if (out/'DEVELOPMENT_SEAL.json').exists():
        verify(read(out/'DEVELOPMENT_SEAL.json')['files']);print('All five developments already sealed.');return
    for fold in p['folds']:
        run=Path(fold['run'])
        if (run/'FINAL_STARTED.json').exists():raise ValueError('A fold test has already been exposed before the CV seal.')
        if (run/'freeze.json').exists():
            fr=read(run/'freeze.json');verify(fr['files'])
            if fr['environment']!=R.env():raise ValueError('Frozen fold environment mismatch.')
            print('Preserving completed development fold',fold['fold']);continue
        if run.exists() and any(run.iterdir()):raise ValueError('Partial development found; preserve it and inspect the log. No silent restart: '+str(run))
        child(out,'develop_%02d'%fold['fold'],['develop','--manifest',fold['manifest'],'--run',str(run)])
        if not (run/'freeze.json').is_file():raise ValueError('Development did not produce its seal.')
    files={}
    for fold in p['folds']:
        run=Path(fold['run']);fr=read(run/'freeze.json');verify(fr['files'])
        files.update(fr['files']);files[str((run/'freeze.json').resolve())]=sha(run/'freeze.json')
    files[str((out/'plan_seal.json').resolve())]=sha(out/'plan_seal.json')
    write(out/'DEVELOPMENT_SEAL.json',{'files':files,'note':'All five validation selections were completed before outer test evaluation through this wrapper.'})
    print('ALL FIVE DEVELOPMENTS SEALED. Review history, then run evaluate explicitly.')


def evaluate(args):
    out,p=open_plan(args.plan);seal=read(out/'DEVELOPMENT_SEAL.json');verify(seal['files'])
    history=args.cohort_history.strip()
    if not history:raise ValueError('Cohort-history statement is required.')
    marker=out/'CV_FINAL_STARTED.json'
    if marker.exists():
        old=read(marker)
        if old['cohort_history']!=history:raise ValueError('Cohort history cannot change after evaluation starts.')
    else:
        if any((Path(f['run'])/'FINAL_STARTED.json').exists() for f in p['folds']):
            raise ValueError('A fold was evaluated outside this wrapper. Preserve results and review exposure history.')
        with marker.open('x',encoding='utf-8') as f:json.dump({'mode':'exploratory','cohort_history':history},f,indent=2)
    for fold in p['folds']:
        run=Path(fold['run']);done=run/'CV_RESULT_SEAL.json'
        if done.exists():verify(read(done)['files']);print('Preserving completed test fold',fold['fold']);continue
        if (run/'FINAL_STARTED.json').exists() or (run/'final_results.json').exists():
            raise ValueError('Previously started/unsealed fold test; do NOT reset or retry. Inspect artifacts: '+str(run))
        child(out,'evaluate_%02d'%fold['fold'],['final','--run',str(run),'--mode','exploratory','--cohort-history',history])
        paths=[run/name for name in ('final_results.json','results_final.csv','comparisons_paired.csv','FINAL_STARTED.json')]
        if not all(x.is_file() for x in paths):raise ValueError('Missing final artifacts.')
        write(done,{'files':{str(x.resolve()):sha(x) for x in paths}})
    summarize(args)


def scores(y,yp,nclasses):
    import numpy as np
    y=np.asarray(y,dtype=int);yp=np.asarray(yp,dtype=int)
    if len(y)==0 or len(y)!=len(yp):raise ValueError('Missing/misaligned predictions.')
    if min(y.min(),yp.min())<0 or max(y.max(),yp.max())>=nclasses:raise ValueError('Invalid class index.')
    cm=np.zeros((nclasses,nclasses),dtype=int);np.add.at(cm,(y,yp),1)
    support=cm.sum(1)
    if (support==0).any():raise ValueError('A class is absent in outer test.')
    rec=np.diag(cm)/support
    prec=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(nclasses),where=cm.sum(0)!=0)
    f1=np.divide(2*prec*rec,prec+rec,out=np.zeros(nclasses),where=(prec+rec)!=0)
    return {'accuracy':float(np.trace(cm)/len(y)),'balanced_accuracy':float(rec.mean()),'macro_f1':float(f1.mean())}


def summarize(args):
    import numpy as np
    out,p=open_plan(args.plan);verify(read(out/'DEVELOPMENT_SEAL.json')['files'])
    fold_rows=[];oof=[];method_order=None;expected_paths=set();patient_outer={}
    for fold in p['folds']:
        run=Path(fold['run']);verify(read(run/'CV_RESULT_SEAL.json')['files'])
        j=read(run/'final_results.json');d=read(Path(fold['manifest'])/'dataset.json')
        if j['class_names']!=p['classes']:raise ValueError('Class order mismatch.')
        tests=[r for r in d['rows'] if r['split']=='test'];files=[r['path'] for r in tests]
        if expected_paths.intersection(files):raise ValueError('Repeated outer-test images.')
        expected_paths.update(files)
        for r in tests:
            if r['group'] in patient_outer and patient_outer[r['group']]!=fold['fold']:
                raise ValueError('Patient appears in multiple outer-test folds.')
            patient_outer[r['group']]=fold['fold']
        models=[('proposed',j['proposed'])]+[(b['key'],b) for b in j['baselines']]
        keys=[k for k,v in models]
        if method_order is None:method_order=keys
        if method_order!=keys:raise ValueError('Inconsistent baseline coverage/order.')
        expected_y=[p['classes'].index(r['label']) for r in tests]
        for name,result in models:
            pred=result['slice_predictions'];y=pred['y_true'];yp=pred['y_pred']
            if pred.get('files')!=files or y!=expected_y:raise ValueError('Predictions are not aligned with manifest.')
            m=scores(y,yp,len(p['classes']))
            for k,original in [('accuracy','test_acc'),('balanced_accuracy','test_bal_acc'),('macro_f1','test_f1')]:
                if abs(m[k]-result[original])>1e-10:raise ValueError('Reported metric does not match predictions.')
            fold_rows.append({'fold':fold['fold'],'method':name,'images':len(y),'patients':len({r['group'] for r in tests}),**m})
            for r,yt,ypred in zip(tests,y,yp):
                oof.append({'fold':fold['fold'],'method':name,'path':r['path'],'patient_id':r['group'],'y_true':yt,'y_pred':ypred})
    if len(expected_paths)!=p['input_rows'] or len(patient_outer)!=p['patients']:raise ValueError('Incomplete out-of-fold coverage.')
    summary=[]
    for method in method_order:
        fr=[r for r in fold_rows if r['method']==method];obs=[r for r in oof if r['method']==method]
        if len(fr)!=5 or len(obs)!=p['input_rows']:raise ValueError('Missing folds/predictions.')
        row={'method':method}
        pooled=scores([r['y_true'] for r in obs],[r['y_pred'] for r in obs],len(p['classes']))
        for metric in ('accuracy','balanced_accuracy','macro_f1'):
            vals=[r[metric] for r in fr]
            row[metric+'_mean']=float(np.mean(vals));row[metric+'_sd']=float(np.std(vals,ddof=1));row[metric+'_pooled_oof']=pooled[metric]
        summary.append(row)
    csv_write(out/'cv_fold_metrics.csv',fold_rows,list(fold_rows[0]))
    csv_write(out/'cv_summary.csv',summary,list(summary[0]))
    csv_write(out/'cv_oof_predictions.csv',oof,list(oof[0]))
    write(out/'cv_report.json',{'protocol':p,'summary':summary,
          'warnings':['All metric values are fractions, not percentage units.',
                      'Sample SD across five overlapping-training folds is descriptive; it is not a standard error or confidence interval.',
                      'Pooled OOF metrics are slice-level. No pooled significance test is supplied because fitted folds share training data.',
                      'Model/hyperparameter winners may legitimately differ across folds; do not choose a deployable winner by test-fold score.',
                      'Patient IDs and image paths in the OOF file may be sensitive; inspect before sharing publicly.']})
    print('CV COMPLETE:',out/'cv_summary.csv')


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='stage',required=True)
    q=sub.add_parser('plan');q.add_argument('--data',required=True);q.add_argument('--csv',required=True);q.add_argument('--out',required=True)
    q.add_argument('--seed',type=int,default=42);q.add_argument('--valid-fraction',type=float,default=.2);q.add_argument('--normal-class',default=None)
    q.add_argument('--repartition-existing',action='store_true',help='Explicitly discard existing CSV split assignments for exploratory CV. Does NOT reset historical test exposure.')
    for stage in ('develop','evaluate','summarize'):
        q=sub.add_parser(stage);q.add_argument('--plan',required=True)
        if stage=='evaluate':q.add_argument('--cohort-history',required=True)
    a=ap.parse_args();{'plan':plan,'develop':develop,'evaluate':evaluate,'summarize':summarize}[a.stage](a)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;traceback.print_exc();print('STOP. Keep all logs and seals. Do not delete markers to restart tests.',file=sys.stderr);sys.exit(1)
