"""Reuse the supplied learners, replacing ALL legacy dataset/cache dispatch."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import ADAPTIVE_DATA as D
import ADAPTIVE_PREPROCESS as PRE

def install(V,run,final=False):
    import numpy as np
    V.PRE=PRE
    def load_bed(key,split_dir=None,normal_class='auto'):
        doc=D.load(split_dir)
        splits={r:D.Split(doc,r) for r in ('train','valid','test')}
        bm=SimpleNamespace(splits=splits,class_names=doc['classes'],group_unit=doc['group_unit'])
        normal=doc['classes'].index(doc['normal_class']) if doc['normal_class'] is not None else None
        return V.Bed5(key='dataset',manifest=bm,class_names=tuple(doc['classes']),normal_idx=normal,
                      group_unit=doc['group_unit'],has_pids=doc['group_unit']=='supplied_group')
    V.load_bed=load_bed
    V.S.cohort_overlap=lambda key:{'corpus_shared_with':'','cohort_test_in_other_dev':'','cohort_overlap_note':'not audited for this dataset'}
    class Store:
        def __init__(self,cfg,bed):
            self.cfg=cfg;self.bed=bed;self.binds=[];self.memory={};self.cache=Path(run)/'native_cache';self.cache.mkdir(parents=True,exist_ok=True)
        def backbone_features(self,role,backbone,tta_n):
            if role=='test' and not final:raise RuntimeError('Test features blocked during development.')
            if role not in ('train','valid','test') or tta_n!=1:raise ValueError('Unsupported role/TTA.')
            key=(role,backbone)
            if key in self.memory:return self.memory[key]
            m=self.bed.split(role);token=hashlib.sha256((m.digest()+backbone+PRE.preprocess_rule_id(self.cfg.img_size)).encode()).hexdigest();xp=self.cache/(backbone+'_'+role+'_'+token+'.npy');yp=self.cache/(backbone+'_'+role+'_'+token+'_labels.npy')
            if xp.exists() and yp.exists():X=np.load(xp,allow_pickle=False);y=np.load(yp,allow_pickle=False)
            else:
                import torch
                from PIL import Image
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net,dim=V.build_extractor_v5(backbone,device,self.cfg.img_size);net.eval();transform=PRE.build_transform(self.cfg.img_size,backbone);chunks=[]
                with torch.inference_mode():
                    for i in range(0,m.n,16):
                        images=[]
                        for path in m.files[i:i+16]:
                            with Image.open(path) as im:images.append(transform(im.convert('RGB')))
                        z=net(torch.stack(images).to(device));z=z.logits if hasattr(z,'logits') else z;z=z.flatten(1)
                        if z.shape[1]!=V.ALL_DIM[backbone]:raise RuntimeError('Unexpected embedding dimension for '+backbone)
                        chunks.append(z.float().cpu().numpy())
                X=np.concatenate(chunks).astype(np.float32);y=m.y();np.save(xp,X);np.save(yp,y);del net
                if device.type=='cuda':torch.cuda.empty_cache()
            if X.shape!=(m.n,V.ALL_DIM[backbone]) or not np.array_equal(y,m.y()):raise RuntimeError('Native cache does not match manifest.')
            if not np.isfinite(X).all():raise RuntimeError('Nonfinite embeddings.')
            self.binds.append({'role':role,'backbone':backbone,'tta_requested':1,'tta_effective':1,'file':str(xp)});self.memory[key]=(X,y);return X,y
        def spec_features(self,role,spec,tta_n):
            pairs=[self.backbone_features(role,b,tta_n) for b in spec.split('+')]
            if not all(np.array_equal(pairs[0][1],y) for x,y in pairs):raise RuntimeError('Fusion labels differ.')
            return np.concatenate([x for x,y in pairs],axis=1),pairs[0][1]
        def effective_tta(self,role,spec=None,requested=None):return {'requested':[1],'effective':[1],'substituted':[],'uniform':True,'notes':[]}
        def effective_tta_scalar(self,role,spec=None,requested=None):return 1
        def report(self):return {'binds':self.binds,'preprocessing':PRE.provenance(self.cfg.img_size)}
    V.FeatureStore=Store;return V
