"""Dynamic raster-image manifests. No resampling-to-balance and no deduplication."""
from pathlib import Path
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
ALIASES={'train':'train','training':'train','valid':'valid','val':'valid','validation':'valid','test':'test','testing':'test'}
class DataError(ValueError): pass

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def role(s):
    if not s.strip():return 'train'
    if s.strip().lower() not in ALIASES:raise DataError('Unknown split name: '+s+'; use train/valid/test in CSV.')
    return ALIASES[s.strip().lower()]

def folders(root,layout='auto'):
    children=sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'));splitdirs={};rows=[]
    if layout!='flat':
        for p in children:
            if p.name.lower() in ALIASES:
                r=ALIASES[p.name.lower()]
                if r in splitdirs:raise DataError('Multiple folders for split '+r+'; use a CSV manifest.')
                splitdirs[r]=p
    if layout=='split' and not splitdirs:raise DataError('No recognized split folders; use CSV for arbitrary layouts.')
    sources=splitdirs or {'train':root}
    if splitdirs and len(splitdirs)!=len(children):raise DataError('Mixed split folders and other directories; choose --layout flat or use --csv explicitly.')
    for r,base in sources.items():
        for cls in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith('.')):
            for p in sorted(cls.rglob('*')):
                if p.is_file() and p.suffix.lower() in EXT:rows.append({'path':str(p.resolve()),'label':cls.name,'split':r,'group':None})
                elif p.is_file() and (p.suffix.lower() in ('.dcm','.nii','.mat','.npy') or p.name.lower().endswith('.nii.gz')):raise DataError('Unsupported medical/array format: '+str(p)+'; convert explicitly before using this image classifier.')
    if not rows:raise DataError('No supported class-folder images found. Use CSV for another layout.')
    return rows

def csv_rows(root,path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'path','label'}.issubset(reader.fieldnames or []):raise DataError('CSV needs path,label; optional split,group.')
        rows=[]
        for n,r in enumerate(reader,2):
            p=Path(r['path']);p=p if p.is_absolute() else root/p
            if not r['label'].strip():raise DataError('Blank label on CSV line '+str(n))
            rows.append({'path':str(p.resolve()),'label':r['label'].strip(),'split':role(r.get('split','')),'group':r.get('group','').strip() or None})
    return rows

def validate(rows):
    if not rows:raise DataError('Empty dataset.')
    seen=set();groups={}
    for r in rows:
        p=Path(r['path'])
        if not p.is_file():raise DataError('Missing file: '+str(p))
        if p.suffix.lower() not in EXT:raise DataError('Unsupported format: '+str(p))
        if r['path'] in seen:raise DataError('Same path listed more than once: '+r['path'])
        seen.add(r['path']);key=r['group'];prior=groups.setdefault(key,(r['label'],r['split']))
        if prior!=(r['label'],r['split']):raise DataError('A group crosses splits or labels: '+str(key)+'. Original assignments were NOT changed.')

def carve(rows,target,fraction,seed):
    if not 0<fraction<1:raise DataError('Split fractions must be between 0 and 1.')
    pools=defaultdict(dict)
    for r in rows:
        if r['split']=='train':pools[r['label']][r['group']]=True
    rng=random.Random(seed);moved=set()
    for label,ids in sorted(pools.items()):
        ids=sorted(ids);rng.shuffle(ids)
        if len(ids)<2:raise DataError('Insufficient train groups for '+target+' in class '+label+'. Supply explicit splits or more groups.')
        n=max(1,min(len(ids)-1,round(len(ids)*fraction)));moved.update(ids[:n])
    for r in rows:
        if r['split']=='train' and r['group'] in moved:r['split']=target

def prepare(data,out,csv_path=None,layout='auto',val_fraction=.2,test_fraction=.2,seed=42,normal_class=None):
    root=Path(data).resolve();out=Path(out)
    if not root.is_dir():raise DataError('Dataset directory not found: '+str(root))
    if out.exists() and any(out.iterdir()):raise DataError('Prepared output must be a new empty directory.')
    rows=csv_rows(root,csv_path) if csv_path else folders(root,layout);with_group=[bool(r['group']) for r in rows]
    if any(with_group) and not all(with_group):raise DataError('Some CSV groups are missing. Supply all groups or none.')
    grouped=bool(rows and all(with_group))
    for r in rows:
        if not grouped:r['group']='image:'+r['path']
    validate(rows);original={r['path']:r['split'] for r in rows};present={r['split'] for r in rows};generated=[]
    if 'train' not in present:raise DataError('A training pool is required.')
    if 'test' not in present:carve(rows,'test',test_fraction,seed);generated.append('test')
    if 'valid' not in present:
        fraction=val_fraction/(1-test_fraction) if 'test' in generated else val_fraction;carve(rows,'valid',fraction,seed+1);generated.append('valid')
    validate(rows);classes=sorted({r['label'] for r in rows})
    if len(classes)<2:raise DataError('At least two classes required; one-class/multilabel tasks need a different learner.')
    if normal_class is not None and normal_class not in classes:raise DataError('Normal-class name not found in labels.')
    for split in ('train','valid','test'):
        labels={r['label'] for r in rows if r['split']==split}
        if labels!=set(classes):raise DataError('Every split must contain all classes for this comparison. Missing in '+split+': '+str(set(classes)-labels))
    out.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    modes=Counter()
    for r in rows:
        with Image.open(r['path']) as image:
            if image.mode in ('I','F') or image.mode.startswith('I;16'):raise DataError('High-bit-depth image needs an explicit intensity conversion: '+r['path'])
            if getattr(image,'n_frames',1)!=1:raise DataError('Multi-frame image must be converted to explicit slices: '+r['path'])
            modes[image.mode]+=1;image.verify()
        r['sha256']=sha(r['path'])
    rows.sort(key=lambda r:(r['split'],r['label'],r['path']))
    manifest={'version':1,'root':str(root),'classes':classes,'normal_class':normal_class,'group_unit':'supplied_group' if grouped else 'image','seed':seed,'generated_splits':generated,'rows':rows,'warning':'User-supplied groups must identify independent units. With image IDs, patient overlap is unassessed. No deduplication was performed.'}
    save(out/'dataset.json',manifest)
    summary={'images':len(rows),'classes':classes,'group_unit':manifest['group_unit'],'counts':{s:dict(Counter(r['label'] for r in rows if r['split']==s)) for s in ('train','valid','test')},'group_counts':{s:len({r['group'] for r in rows if r['split']==s}) for s in ('train','valid','test')},'generated_splits':generated,'seed':seed,'image_modes':dict(modes),'deduplication_performed':False,'class_balancing_resampling_performed':False,'warning':manifest['warning']}
    save(out/'dataset_summary.json',summary);print(json.dumps(summary,indent=2,ensure_ascii=False));return manifest

def load(folder):return json.loads((Path(folder)/'dataset.json').read_text(encoding='utf-8'))
def check_content(manifest,roles):
    for r in manifest['rows']:
        if r['split'] in roles and (not Path(r['path']).is_file() or sha(r['path'])!=r['sha256']):raise DataError('Image content changed after manifest preparation: '+r['path'])
class Split:
    def __init__(self,manifest,split):
        self.role=split;self.rows=[r for r in manifest['rows'] if r['split']==split];self.classes=manifest['classes'];self.files=[r['path'] for r in self.rows];self.groups=[r['group'] for r in self.rows];self.n=len(self.rows);self.recipe={'source':'dynamic manifest','role':split,'generated':split in manifest['generated_splits']}
    def y(self):
        import numpy as np
        return np.asarray([self.classes.index(r['label']) for r in self.rows],dtype=np.int64)
    def abs_paths(self):return self.files
    def n_groups(self):return len(set(self.groups))
    def class_counts(self):return dict(Counter(r['label'] for r in self.rows))
    def digest(self):return hashlib.sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()
