"""Additional exploratory controls, isolated from all completed runs.
init -> develop (all folds) -> evaluate. No original file is modified.
"""
from pathlib import Path
import argparse, time, json, os, warnings
import numpy as np
import pandas as pd
from EXPERIMENT_COMMON import *
from OOF_ANALYSIS import metrics

def init(a):
    out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Output must be new/empty')
    if bool(a.plan)==bool(a.manifest):raise ValueError('Use --plan OR --manifest with --run')
    if a.plan:
        base=Path(a.plan).resolve();plan=read(base/'plan.json');fs=[]
        if len(plan['folds'])!=5:raise ValueError('Expected same five existing folds')
        for e in plan['folds']:
            n=e['fold'];mp=base/'manifests'/('fold_%02d'%n)/'dataset.json';rp=base/'runs'/('fold_%02d'%n)
            fs.append(dict(fold=n,manifest=str(mp),run=str(rp),frozen_selection=str(rp/'frozen_selection.json')))
    else:
        if not a.run:raise ValueError('--run required with --manifest')
        mp=Path(a.manifest).resolve()/'dataset.json';rp=Path(a.run).resolve();fs=[dict(fold=1,manifest=str(mp),run=str(rp),frozen_selection=str(rp/'frozen_selection.json'))]
    if any(out==Path(f['run']) or Path(f['run']) in out.parents or out==Path(f['manifest']).parent for f in fs):raise ValueError('Use a separate output directory')
    classes=validate_manifests(fs,grouped=bool(a.plan));out.mkdir(parents=True,exist_ok=True)
    if a.epochs<1 or a.batch<1 or a.patience<1:raise ValueError('Invalid training settings')
    bindings={}; cache_status=[]; oldlp=[]
    for f in fs:
        for p in [f['manifest'],f['frozen_selection']]:bindings[p]=sha(p)
        doc=read(f['manifest']);oldlp.append(dict(fold=f['fold'],selection=read(f['frozen_selection'])['baselines'].get('linear_probe')))
        if a.kind=='lp':
            for role in ['train','valid','test']:
                for xp,yp,dim in cache_files(doc,role,f['run']):
                    if not xp.is_file() or not yp.is_file():raise ValueError('Missing exact cache: '+str(xp)+'. Preserve old runs. Restore their native_cache before proceeding.')
                    bindings[str(xp)]=sha(xp);bindings[str(yp)]=sha(yp);cache_status.append(str(xp))
    if a.kind=='e2e':
        import torch
        from VISION_HELPERS import weight_enum
        w=weight_enum('resnet18');wp=Path(torch.hub.get_dir())/'checkpoints'/w.url.rsplit('/',1)[-1]
        if not wp.is_file():raise ValueError('ResNet18 weights missing. Run BENCHMARK_TIMING.py weights --backbones resnet18 --download before init.')
        bindings[str(wp.resolve())]=sha(wp)
    for name in ['NEW_CONTROLS.py','EXPERIMENT_COMMON.py','VISION_HELPERS.py','OOF_ANALYSIS.py']:
        p=Path(__file__).resolve().parent/name;bindings[str(p)]=sha(p)
    cfg=dict(version='v21-additional-controls-1',kind=a.kind,folds=fs,grouped=bool(a.plan),classes=classes,environment=environment(),bindings=bindings,
      seed=42,C_grid=[.01,.1,1.,10.],pca=[0.,.95],class_weight='balanced',max_iter=3000,solver='lbfgs',
      epochs=a.epochs,batch=a.batch,patience=a.patience,lr_grid=[.0001,.001],weight_decay=.0001,optimizer='AdamW',architecture='resnet18',
      device=a.device,preprocessing=PROVENANCE,selection='slice validation balanced accuracy; argmax; first candidate wins exact ties',
      original_linear_probe=oldlp,history='Additional analyses planned AFTER inspection of original test outcomes. Exploratory internal evaluation only; not independent confirmatory testing.')
    verify_inputs(cfg,images=True)
    write(out/'CONFIG.json',cfg);print('INITIALIZED',out,'| old Linear Probe settings:',oldlp)

def lp_develop(cfg,f,attempt):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    import joblib
    d=read(f['manifest']);timing=[];start=time.perf_counter();tr,y=load_features(d,'train',f['run']);va,yv=load_features(d,'valid',f['run']);loadtime=time.perf_counter()-start
    choices=[]
    for variance in cfg['pca']:
        t=time.perf_counter();scaler=StandardScaler();z=scaler.fit_transform(tr);v=scaler.transform(va)
        pca=PCA(n_components=variance,svd_solver='full',random_state=cfg['seed']) if variance else None
        if pca is not None:z=pca.fit_transform(z);v=pca.transform(v)
        z=np.asarray(z,dtype=np.float32);v=np.asarray(v,dtype=np.float32);representation=time.perf_counter()-t
        best=-np.inf;method='lp_tuned_pca95' if variance else 'lp_tuned_raw'
        for c in cfg['C_grid']:
            model=LogisticRegression(C=c,max_iter=cfg['max_iter'],solver=cfg['solver'],class_weight='balanced',random_state=cfg['seed'])
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always');t=time.perf_counter();model.fit(z,y);fitsec=time.perf_counter()-t
            if any(issubclass(w.category,ConvergenceWarning) for w in ww):raise ValueError('LR did not converge. Do not test partial selection; review numerical settings for ALL folds.')
            t=time.perf_counter();pred=model.predict(v);evalsec=time.perf_counter()-t;cm=np.zeros((len(d['classes']),)*2,int);np.add.at(cm,(yv,pred),1);ba=float(metrics(cm)[1])
            timing.append(dict(method=method,C=c,valid_balanced_accuracy=ba,representation_seconds=representation,fit_seconds=fitsec,validation_seconds=evalsec,components=z.shape[1]))
            print('fold',f['fold'],method,'C',c,'valid BA',round(ba,6),flush=True)
            if ba>best:
                best=ba;path=attempt/(method+'.joblib');joblib.dump((scaler,pca,model),path);choice=dict(method=method,C=c,pca=variance,components=z.shape[1],validation_ba=ba,model=str(path.resolve()))
        choices.append(choice)
    pd.DataFrame(timing).to_csv(attempt/'validation_trials.csv',index=False)
    return dict(choices=choices,cache_read_seconds=loadtime,representation_note='Train-fitted standardization, then optional full-SVD PCA-95; shared C grid. No train+valid refit.',artifacts={str(p.resolve()):sha(p) for p in attempt.iterdir() if p.is_file()})

def e2e_develop(cfg,f,attempt):
    import torch
    from VISION_HELPERS import seed_all,device_for,loader,model_for,sync,infer
    d=read(f['manifest']);device=device_for(cfg['device']);k=len(d['classes']);log=[];choices=[];best=-np.inf
    counts=np.bincount(labels(d,rows(d,'train')),minlength=k);cw=torch.tensor(counts.sum()/(k*counts),dtype=torch.float32,device=device)
    for lr in cfg['lr_grid']:
        seed_all(cfg['seed']);train=loader(d,'train',cfg['batch'],True,cfg['seed'],device);valid=loader(d,'valid',cfg['batch'],False,cfg['seed'],device)
        net=model_for('resnet18',k,device);opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=cfg['weight_decay']);criterion=torch.nn.CrossEntropyLoss(weight=cw)
        localbest=-np.inf;stale=0
        for epoch in range(1,cfg['epochs']+1):
            net.train();sync(device);t=time.perf_counter();loss_sum=0.;n=0
            for x,y in train:
                x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True);loss=criterion(net(x),y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();opt.step();loss_sum+=float(loss.detach())*len(y);n+=len(y)
            sync(device);trainsec=time.perf_counter()-t;t=time.perf_counter();p,y=infer(net,valid,device);valsec=time.perf_counter()-t
            cm=np.zeros((k,k),int);np.add.at(cm,(y,p.argmax(1)),1);ba=float(metrics(cm)[1])
            log.append(dict(lr=lr,epoch=epoch,train_loss=loss_sum/n,valid_balanced_accuracy=ba,train_seconds=trainsec,validation_seconds=valsec))
            pd.DataFrame(log).to_csv(attempt/'epoch_log.csv',index=False);print('fold',f['fold'],'lr',lr,'epoch',epoch,'valid BA',round(ba,6),flush=True)
            if ba>localbest:localbest=ba;stale=0
            else:stale+=1
            if ba>best:
                best=ba;path=attempt/'resnet18_finetuned.pt';torch.save({kk:vv.detach().cpu() for kk,vv in net.state_dict().items()},path)
                choice=dict(method='resnet18_finetuned',lr=lr,epoch=epoch,validation_ba=ba,model=str(path.resolve()))
            if stale>=cfg['patience']:break
        del net,opt
        if device.type=='cuda':torch.cuda.empty_cache()
    return dict(choices=[choice],artifacts={str(p.resolve()):sha(p) for p in attempt.iterdir() if p.is_file()},hardware={'device':str(device),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None})

def develop(a):
    out=Path(a.out).resolve();cfg=read(out/'CONFIG.json')
    with locked(out):
        verify_inputs(cfg,images=True)
        if any(out.glob('fold_*/FINAL_STARTED.json')):raise ValueError('Test evaluation already started; no further development allowed')
        if (out/'DEVELOPMENT_SEAL.json').exists():verify_seal(out);print('All development already sealed');return
        for f in cfg['folds']:
            folder=out/('fold_%02d'%f['fold']);folder.mkdir(exist_ok=True)
            if (folder/'DEVELOPED.json').exists():
                for p,h in read(folder/'DEVELOPED.json')['artifacts'].items():
                    if sha(p)!=h:raise ValueError('Completed artifact changed')
                print('[KEEP] completed development fold',f['fold']);continue
            attempt=folder/('attempt_%02d'%(len(list(folder.glob('attempt_*')))+1));attempt.mkdir();write(attempt/'STARTED.json',{'time':time.time(),'note':'Interrupted attempts are preserved; only undeveloped folds can restart before testing.'})
            t=time.perf_counter();result=(lp_develop if cfg['kind']=='lp' else e2e_develop)(cfg,f,attempt);result['development_wall_seconds']=time.perf_counter()-t
            write(folder/'DEVELOPED.json',result)
        seal_development(out,cfg);print('ALL FOLDS FROZEN. Now run evaluate once.')

def evaluate(a):
    out=Path(a.out).resolve();cfg=read(out/'CONFIG.json')
    with locked(out):
        verify_inputs(cfg,images=True);verify_seal(out)
        for f in cfg['folds']:
            folder=out/('fold_%02d'%f['fold']);done=folder/'FINAL_DONE.json'
            if done.exists():
                if sha(folder/'oof.csv')!=read(done)['oof_sha256']:raise ValueError('Final output changed')
                print('[KEEP] completed final fold',f['fold']);continue
            marker=folder/'FINAL_STARTED.json'
            if marker.exists():raise ValueError('Interrupted final evaluation in '+str(folder)+'. Preserve files; review before any rerun. Do not delete FINAL_STARTED.')
            choices=read(folder/'DEVELOPED.json')['choices'];d=read(f['manifest']);outputs=[]
            with marker.open('x',encoding='utf-8') as fp:json.dump({'time':time.time(),'history':cfg['history']},fp)
            t=time.perf_counter()
            if cfg['kind']=='lp':
                import joblib
                x,y=load_features(d,'test',f['run'])
                for c in choices:
                    sc,pc,clf=joblib.load(c['model']);z=sc.transform(x)
                    if pc is not None:z=pc.transform(z)
                    outputs+=prediction_rows(d,f['fold'],c['method'],clf.predict_proba(np.asarray(z,dtype=np.float32)))
            else:
                import torch
                from VISION_HELPERS import model_for,device_for,loader,infer
                device=device_for(cfg['device']);net=model_for('resnet18',len(d['classes']),device,pretrained=False)
                net.load_state_dict(torch.load(choices[0]['model'],map_location=device,weights_only=True));p,y=infer(net,loader(d,'test',cfg['batch'],False,cfg['seed'],device),device)
                outputs+=prediction_rows(d,f['fold'],choices[0]['method'],p)
                del net
                if device.type=='cuda':torch.cuda.empty_cache()
            pd.DataFrame(outputs).to_csv(folder/'oof.csv',index=False);write(done,{'evaluation_wall_seconds':time.perf_counter()-t,'oof_sha256':sha(folder/'oof.csv')})
        dfs=[pd.read_csv(out/('fold_%02d'%f['fold'])/'oof.csv',dtype={'patient_id':str}) for f in cfg['folds']];df=pd.concat(dfs,ignore_index=True)
        df.to_csv(out/'new_controls_oof.csv',index=False);fr=[]
        for (fold,method),r in df.groupby(['fold','method']):
            cm=np.zeros((len(cfg['classes']),)*2,int);np.add.at(cm,(r.y_true,r.y_pred),1)
            fr.append(dict(fold=int(fold),method=method,**dict(zip(['accuracy','balanced_accuracy','macro_f1'],metrics(cm)))))
        pd.DataFrame(fr).to_csv(out/'new_controls_fold_metrics.csv',index=False)
        pd.DataFrame(fr).groupby('method')[['accuracy','balanced_accuracy','macro_f1']].agg(['mean','std']).to_csv(out/'new_controls_summary.csv')
        timings=[]
        for f in cfg['folds']:
            folder=out/('fold_%02d'%f['fold']);dev=read(folder/'DEVELOPED.json')
            row=dict(fold=f['fold'],kind=cfg['kind'],development_wall_seconds=dev['development_wall_seconds'],evaluation_wall_seconds=read(folder/'FINAL_DONE.json')['evaluation_wall_seconds'])
            logname='validation_trials.csv' if cfg['kind']=='lp' else 'epoch_log.csv'
            logpath=next(Path(p) for p in dev['artifacts'] if Path(p).name==logname);log=pd.read_csv(logpath)
            if cfg['kind']=='lp':row.update(mean_fit_seconds=float(log.fit_seconds.mean()),actual_model_fits=len(log),sum_fit_seconds=float(log.fit_seconds.sum()),cache_read_seconds=dev['cache_read_seconds'])
            else:row.update(actual_epochs=len(log),mean_epoch_train_seconds=float(log.train_seconds.mean()),sum_train_seconds=float(log.train_seconds.sum()))
            timings.append(row)
        pd.DataFrame(timings).to_csv(out/'new_controls_timing.csv',index=False)
        print('COMPLETE. Original results unchanged:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('init');q.add_argument('--kind',choices=['lp','e2e'],required=True);q.add_argument('--plan');q.add_argument('--manifest');q.add_argument('--run');q.add_argument('--out',required=True);q.add_argument('--epochs',type=int,default=20);q.add_argument('--patience',type=int,default=5);q.add_argument('--batch',type=int,default=16);q.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    for cmd in ['develop','evaluate']:s.add_parser(cmd).add_argument('--out',required=True)
    a=p.parse_args();{'init':init,'develop':develop,'evaluate':evaluate}[a.cmd](a)
if __name__=='__main__':main()
