"""Read-only original inputs; isolated, hashed experimental outputs."""
from pathlib import Path
import json, hashlib, os, sys, platform, importlib.metadata, time
import numpy as np

BACKBONES={'resnet18':512,'densenet121':1024,'vit_b_16':768}
PROVENANCE={'img_size':224,'resize':'square, bilinear, antialias=True','channels':'PIL convert RGB','mean':[.485,.456,.406],'std':[.229,.224,.225],'tta':['identity'],'channel_audit':'not performed on this dataset'}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def environment():
    d={'python':sys.version,'platform':platform.platform()}
    for p in ['numpy','pandas','scipy','scikit-learn','joblib','torch','torchvision','Pillow']:
        try:d[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:d[p]=None
    return d

def rows(doc,role):return [r for r in doc['rows'] if r['split']==role]
def labels(doc,rs):return np.asarray([doc['classes'].index(r['label']) for r in rs],dtype=np.int64)
def digest_rows(rs):return hashlib.sha256(json.dumps(rs,sort_keys=True).encode()).hexdigest()
def cache_files(doc,role,run):
    rs=rows(doc,role);md=digest_rows(rs)
    pre=hashlib.sha256(json.dumps(PROVENANCE,sort_keys=True).encode()).hexdigest();files=[]
    for bb,dim in BACKBONES.items():
        token=hashlib.sha256((md+bb+pre).encode()).hexdigest()
        xp=Path(run)/'native_cache'/(bb+'_'+role+'_'+token+'.npy')
        yp=xp.with_name(xp.stem+'_labels.npy');files.append((xp,yp,dim))
    return files

def load_features(doc,role,run):
    yy=labels(doc,rows(doc,role));parts=[]
    for xp,yp,dim in cache_files(doc,role,run):
        if not xp.is_file() or not yp.is_file():raise ValueError('Exact existing cache missing: '+str(xp)+'. No legacy file will be guessed or overwritten.')
        x=np.load(xp,allow_pickle=False);y=np.load(yp,allow_pickle=False)
        if x.shape!=(len(yy),dim) or not np.array_equal(y,yy) or not np.isfinite(x).all():raise ValueError('Invalid cache: '+str(xp))
        parts.append(np.asarray(x,dtype=np.float32))
    return np.concatenate(parts,axis=1),yy

def validate_manifests(folds,grouped=True):
    outer={};classref=None;allpaths=None
    for f in folds:
        d=read(f['manifest']);cs=d['classes']
        if classref is None:classref=cs
        if cs!=classref:raise ValueError('Class order differs across folds')
        rs=d['rows'];paths=[r['path'] for r in rs]
        if len(set(paths))!=len(paths):raise ValueError('Duplicate manifest path')
        if any(r.get('group') is None or not str(r['group']).strip() for r in rs):raise ValueError('Missing group')
        if set(r['split'] for r in rs)!={'train','valid','test'}:raise ValueError('Unknown/missing split')
        if grouped and d['group_unit']!='supplied_group':raise ValueError('Plan does not use supplied patient IDs')
        if allpaths is None:allpaths=set(paths)
        if grouped and set(paths)!=allpaths:raise ValueError('Different outer-fold cohorts')
        seen={}
        for r in rs:
            old=seen.setdefault(str(r['group']),(r['split'],r['label']))
            if old!=(r['split'],r['label']):raise ValueError('Group crosses splits/classes')
        for role in ['train','valid','test']:
            if set(r['label'] for r in rows(d,role))!=set(cs):raise ValueError('Missing class in '+role)
        for pid in set(str(r['group']) for r in rows(d,'test')):
            if pid in outer:raise ValueError('ID appears in multiple outer test folds')
            outer[pid]=f['fold']
        if f.get('frozen_selection'):
            frozen=read(f['frozen_selection'])
            if any(digest_rows(rows(d,r))!=dg for r,dg in frozen['manifest_digests'].items()):raise ValueError('Manifest differs from completed run')
    if grouped:
        allids=set(str(r['group']) for r in read(folds[0]['manifest'])['rows'])
        if set(outer)!=allids:raise ValueError('Not every ID appears exactly once in outer tests')
    return classref

def verify_inputs(cfg,images=False):
    for p,h in cfg['bindings'].items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Bound input changed/missing: '+p)
    if environment()!=cfg['environment']:raise ValueError('Environment changed since init; do not silently resume')
    if images:
        seen=set()
        for f in cfg['folds']:
            for r in read(f['manifest'])['rows']:
                if r['path'] not in seen:
                    if not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']:raise ValueError('Image changed or missing: '+r['path'])
                    seen.add(r['path'])

def locked(out):
    class Lock:
        def __enter__(self):
            self.p=Path(out)/'ACTIVE.lock'
            try:
                with self.p.open('x') as f:f.write(str(os.getpid()))
            except FileExistsError:raise ValueError('ACTIVE.lock exists. Confirm no process is running before manually clearing only this lock.')
        def __exit__(self,*a):self.p.unlink(missing_ok=True)
    return Lock()

def prediction_rows(doc,fold,method,probs):
    rs=rows(doc,'test');y=labels(doc,rs);p=np.asarray(probs)
    if p.shape!=(len(rs),len(doc['classes'])) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid output probabilities')
    return [dict(fold=fold,method=method,path=r['path'],patient_id=str(r['group']),y_true=int(y[i]),y_pred=int(p[i].argmax()),**{'p_'+str(c):float(p[i,c]) for c in range(p.shape[1])}) for i,r in enumerate(rs)]

def seal_development(out,cfg):
    files={str(Path(out,'CONFIG.json').resolve()):sha(Path(out,'CONFIG.json'))}
    for f in cfg['folds']:
        folder=Path(out)/('fold_%02d'%f['fold'])
        if not (folder/'DEVELOPED.json').is_file():raise ValueError('All folds must finish development before any outer testing')
        info=read(folder/'DEVELOPED.json')
        files[str((folder/'DEVELOPED.json').resolve())]=sha(folder/'DEVELOPED.json')
        for p,h in info['artifacts'].items():
            if sha(p)!=h:raise ValueError('Development artifact changed')
            files[p]=h
    write(Path(out)/'DEVELOPMENT_SEAL.json',files)

def verify_seal(out):
    for p,h in read(Path(out)/'DEVELOPMENT_SEAL.json').items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Sealed artifact changed: '+p)
