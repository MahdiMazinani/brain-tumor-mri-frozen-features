"""Recover recorded timing fields honestly or prospectively measure new runs."""
from pathlib import Path
import argparse,json,time,subprocess,sys
import pandas as pd
from EXPERIMENT_COMMON import read,write,sha,environment,rows

def fresh(path):
    p=Path(path).resolve()
    if p.exists() and any(p.iterdir()):raise ValueError('Choose a new output directory')
    p.mkdir(parents=True,exist_ok=True);return p

def audit(a):
    out=fresh(a.out);records=[];settings=[];sources=[]
    for p in sorted(Path(a.root).rglob('selection_grid.csv')):
        df=pd.read_csv(p);sources.append({'path':str(p.resolve()),'sha256':sha(p)})
        if 'fit_seconds' in df:
            for i,r in df.iterrows():records.append(dict(source=str(p),row=int(i),spec=r.get('spec'),learner=r.get('learner'),recorded_fit_seconds=r.get('fit_seconds')))
    for p in sorted(Path(a.root).rglob('frozen_selection.json')):
        j=read(p);settings.append(dict(source=str(p),linear_probe=j.get('baselines',{}).get('linear_probe')))
    pd.DataFrame(records,columns=['source','row','spec','learner','recorded_fit_seconds']).to_csv(out/'historical_recorded_fit_fields.csv',index=False)
    write(out/'timing_audit.json',{'sources':sources,'linear_probe_settings':settings,'historical_extraction_seconds':None,'historical_total_sweep_seconds':None,'historical_mean_unique_fit_seconds':None,'reason':'Recorded selection-grid times may include reused tree fits and ensemble member sums; they are NOT additive and do not identify actual unique training operations. File modification times cannot recover wall-clock durations. Missing historical durations remain null. Inspect original logs separately if timestamped start/end records exist.'})
    print('Audit complete; no unrecorded historical time invented:',out)

def weights(a):
    import torch
    from VISION_HELPERS import weight_enum
    for bb in a.backbones.split(','):
        w=weight_enum(bb);p=Path(torch.hub.get_dir())/'checkpoints'/w.url.rsplit('/',1)[-1]
        if not p.exists() and a.download:w.get_state_dict(progress=True,check_hash=True)
        print(bb,'READY' if p.exists() else 'MISSING',p)

def extraction(a):
    import torch,numpy as np
    from VISION_HELPERS import device_for,model_for,loader,sync,seed_all
    out=fresh(a.out);mp=Path(a.manifest)/'dataset.json';d=read(mp);device=device_for(a.device);seed_all(42)
    if a.repeats<1 or a.batch<1:raise ValueError('Positive repeat/batch required')
    for r in d['rows']:
        if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']:raise ValueError('Image changed: '+r['path'])
    log=[]
    for rep in range(1,a.repeats+1):
        for bb in a.backbones.split(','):
            sync(device);t=time.perf_counter();net=model_for(bb,len(d['classes']),device,frozen=True);sync(device);setup=time.perf_counter()-t
            for role in ['train','valid','test']:
                dl=loader(d,role,a.batch,False,42,device);n=0;checksum=0.;sync(device);t=time.perf_counter()
                with torch.inference_mode():
                    for x,y in dl:
                        z=net(x.to(device)).flatten(1).float().cpu().numpy()
                        if not np.isfinite(z).all():raise ValueError('Nonfinite embeddings')
                        n+=len(z);checksum+=float(np.asarray(z,dtype=np.float64).sum())
                sync(device);seconds=time.perf_counter()-t
                log.append(dict(repeat=rep,backbone=bb,split=role,images=n,seconds=seconds,images_per_second=n/seconds,model_setup_seconds=setup,feature_checksum=checksum))
                pd.DataFrame(log).to_csv(out/'extraction_timing.csv',index=False);print(bb,role,round(seconds,2),'s',flush=True)
            del net
            if device.type=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(log).groupby('repeat').seconds.sum().to_csv(out/'dataset_all_backbones_seconds.csv')
    write(out/'protocol.json',dict(environment=environment(),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,batch=a.batch,repeats=a.repeats,manifest_sha256=sha(mp),backbones=a.backbones,scope='NEW benchmark, not historical run timing. Forward + image loading/transform + CPU feature transfer; excludes model setup and array writes. No stored feature cache used or written. No accuracy calculation. No warmup; first-forward overhead is included. OS/driver caching not reset. No deduplication or resampling. Sum across roles/backbones is for this supplied dataset once, not a five-fold historical total.'))

def grid(a):
    out=fresh(a.out);project=Path(a.project).resolve();script=project/'ADAPTIVE_RUN.py'
    if not script.is_file():raise ValueError('ADAPTIVE_RUN.py not found in --project')
    cmd=[sys.executable,'-u',str(script),'develop','--manifest',str(Path(a.manifest).resolve()),'--run',str(out/'fresh_grid_run')]
    write(out/'STARTED.json',{'command':cmd,'script_sha256':sha(script),'environment':environment(),'time':time.time()})
    t=time.perf_counter()
    with (out/'grid.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(cmd,cwd=project,stdout=log,stderr=subprocess.STDOUT)
        rc=process.wait()
    write(out/'grid_wall_time.json',{'returncode':rc,'wall_seconds':time.perf_counter()-t,'scope':'NEW full development invocation, including extraction/cache writes, model selection and IO. New initially empty native cache. Not historical timing; not just model.fit. No final evaluation requested.'})
    if rc:raise RuntimeError('Fresh grid failed; preserve grid.log. Do not retry into the same folder.')
    print('NEW grid wall time recorded:',out)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('audit');q.add_argument('--root',required=True);q.add_argument('--out',required=True)
    q=s.add_parser('weights');q.add_argument('--download',action='store_true');q.add_argument('--backbones',default='resnet18,densenet121,vit_b_16')
    q=s.add_parser('extraction');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--device',default='cuda',choices=['cuda','cpu','auto']);q.add_argument('--batch',type=int,default=16);q.add_argument('--repeats',type=int,default=3);q.add_argument('--backbones',default='resnet18,densenet121,vit_b_16')
    q=s.add_parser('grid');q.add_argument('--project',default='.');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();{'audit':audit,'weights':weights,'extraction':extraction,'grid':grid}[a.cmd](a)
if __name__=='__main__':main()
