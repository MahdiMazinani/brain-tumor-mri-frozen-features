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
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
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
 rs=rows(doc,role);md=digest_rows(rs);pre=hashlib.sha256(json.dumps(PROVENANCE,sort_keys=True).encode()).hexdigest();files=[]
 for bb,dim in BACKBONES.items():
  token=hashlib.sha256((md+bb+pre).encode()).hexdigest();xp=Path(run)/'native_cache'/(bb+'_'+role+'_'+token+'.npy');yp=xp.with_name(xp.stem+'_labels.npy');files.append((xp,yp,dim))
 return files
def load_features(doc,role,run):
 yy=labels(doc,rows(doc,role));parts=[]
 for xp,yp,dim in cache_files(doc,role,run):
  if not xp.is_file() or not yp.is_file():raise ValueError('Exact existing cache missing: '+str(xp)+'. No legacy file will be guessed or overwritten.')
  x=np.load(xp,allow_pickle=False);y=np.load(yp,allow_pickle=False)
  if x.shape!=(len(yy),dim) or not np.array_equal(y,yy) or not np.isfinite(x).all():raise ValueError('Invalid cache: '+str(xp))
  parts.append(np.asarray(x,dtype=np.float32))
 return np.concatenate(parts,axis=1),yy
def locked(out):
 class Lock:
  def __enter__(self):
   self.p=Path(out)/'ACTIVE.lock'
   try:
    with self.p.open('x') as f:f.write(str(os.getpid()))
   except FileExistsError:raise ValueError('ACTIVE.lock exists. Confirm no process is running before manually clearing only this lock.')
  def __exit__(self,*a):self.p.unlink(missing_ok=True)
 return Lock()
