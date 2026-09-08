"""Tests use temporary synthetic fixtures only, never article evidence."""
import unittest,tempfile,json,math,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from OOF_ANALYSIS import metrics,exact_mcnemar,validate,run
from EXPERIMENT_COMMON import *

class CoreTests(unittest.TestCase):
    def fixture(self):
        rr=[]
        for m in ['proposed','linear_probe']:
            for c in range(2):
                for i in range(3):
                    for s in range(2):rr.append(dict(fold=i+1,method=m,path=f'{c}/{i}/{s}',patient_id=f'{c}-{i}',y_true=c,y_pred=c))
        return pd.DataFrame(rr)
    def test_perfect_metrics(self):np.testing.assert_allclose(metrics(np.diag([4,7,3])),[1,1,1])
    def test_hand_metrics(self):np.testing.assert_allclose(metrics(np.array([[3,1],[2,4]])),[.7,(.75+4/6)/2,((6/9)+(8/11))/2])
    def test_exact_mcnemar(self):
        y=np.zeros(11,int);a=np.r_[np.zeros(4),np.ones(7)];b=1-a
        self.assertAlmostEqual(exact_mcnemar(y,a,b)['p_exact_two_sided'],2*sum(math.comb(11,i) for i in range(5))/2**11)
    def test_no_discordance(self):self.assertEqual(exact_mcnemar(np.array([0]),np.array([0]),np.array([0]))['p_exact_two_sided'],1)
    def test_alignment(self):df=self.fixture();df=df.drop(df.index[-1]);self.assertRaises(ValueError,validate,df)
    def test_duplicate(self):df=self.fixture();self.assertRaises(ValueError,validate,pd.concat([df,df.iloc[:1]],ignore_index=True))
    def test_patient_crossfold(self):df=self.fixture();df.loc[0,'fold']=99;self.assertRaises(ValueError,validate,df)
    def test_patient_conflicting_label(self):df=self.fixture();df.loc[0,'y_true']=1;self.assertRaises(ValueError,validate,df)
    def test_stats_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);self.fixture().to_csv(t/'oof.csv',index=False)
            a=SimpleNamespace(oof=str(t/'oof.csv'),out=str(t/'out'),a='proposed',b='linear_probe',resamples=100,seed=42);run(a)
            d=pd.read_csv(t/'out/paired_differences.csv');np.testing.assert_allclose(d[['difference','ci95_low','ci95_high']],0)
            self.assertEqual(read(t/'out/analysis.json')['identifiers'],6);self.assertRaises(ValueError,run,a)
    def test_probability_validation(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);df=self.fixture();df['p_0']=.8;df['p_1']=.8;df.to_csv(t/'oof.csv',index=False)
            self.assertRaises(ValueError,run,SimpleNamespace(oof=str(t/'oof.csv'),out=str(t/'out'),a='proposed',b='linear_probe',resamples=100,seed=42))
    def test_mean_pooling(self):
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);df=self.fixture();df['p_0']=1-df.y_pred;df['p_1']=df.y_pred;df.to_csv(t/'oof.csv',index=False)
            run(SimpleNamespace(oof=str(t/'oof.csv'),out=str(t/'out'),a='proposed',b='linear_probe',resamples=100,seed=42))
            self.assertTrue(read(t/'out/analysis.json')['mean_probability_available'])
    def test_manifest_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'dataset.json';write(p,{'classes':['a','b'],'group_unit':'supplied_group','rows':[{'path':'x','group':'same','label':'a','split':'train'},{'path':'y','group':'same','label':'b','split':'valid'}]})
            self.assertRaises(ValueError,validate_manifests,[{'fold':1,'manifest':str(p)}])
    def test_lock(self):
        with tempfile.TemporaryDirectory() as t:
            with locked(t):
                with self.assertRaises(ValueError):
                    with locked(t):pass
            self.assertFalse((Path(t)/'ACTIVE.lock').exists())
    def test_cache_exact_token(self):
        with tempfile.TemporaryDirectory() as t:
            d={'classes':['a'],'rows':[{'path':'x','label':'a','split':'train','group':'g'}]};d['generated_splits']=[]
            self.assertRaises(ValueError,load_features,d,'train',t)
            for xp,yp,dim in cache_files(d,'train',t):xp.parent.mkdir(exist_ok=True);np.save(xp,np.ones((1,dim),np.float32));np.save(yp,np.array([0]))
            x,y=load_features(d,'train',t);self.assertEqual(x.shape,(1,2304))
            d['rows'][0]['path']='changed';self.assertRaises(ValueError,load_features,d,'train',t)

@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn absent: run this integration test in venv311')
class LPIntegration(unittest.TestCase):
    def test_full_lifecycle(self):
        from NEW_CONTROLS import init,develop,evaluate
        with tempfile.TemporaryDirectory() as t:
            t=Path(t);mp=t/'manifest';rp=t/'old_run';mp.mkdir();rp.mkdir();rr=[]
            for role,n in [('train',16),('valid',6),('test',6)]:
                for i in range(n):
                    path=t/(role+str(i)+'.fixture');path.write_bytes(b'fixture, not MRI')
                    rr.append(dict(path=str(path),label=['a','b'][i%2],split=role,group=role+str(i),sha256=sha(path)))
            d=dict(classes=['a','b'],rows=rr,group_unit='image',generated_splits=[]);write(mp/'dataset.json',d)
            write(rp/'frozen_selection.json',{'manifest_digests':{r:digest_rows(rows(d,r)) for r in ['train','valid','test']},'baselines':{'linear_probe':{'kwargs':{'class_weight':'balanced'}}}})
            rng=np.random.default_rng(1)
            for role in ['train','valid','test']:
                y=labels(d,rows(d,role))
                for xp,yp,dim in cache_files(d,role,rp):
                    xp.parent.mkdir(exist_ok=True);np.save(xp,(rng.normal(0,.05,(len(y),dim))+y[:,None]).astype(np.float32));np.save(yp,y)
            out=t/'new';init(SimpleNamespace(out=str(out),plan=None,manifest=str(mp),run=str(rp),kind='lp',epochs=2,batch=2,patience=1,device='cpu'))
            before={str(p):sha(p) for p in rp.rglob('*') if p.is_file()};a=SimpleNamespace(out=str(out))
            with self.assertRaises(FileNotFoundError):evaluate(a)
            develop(a);develop(a);evaluate(a);evaluate(a)
            df=pd.read_csv(out/'new_controls_oof.csv');self.assertEqual(len(df),12);self.assertEqual(set(df.method),{'lp_tuned_raw','lp_tuned_pca95'})
            self.assertEqual(before,{str(p):sha(p) for p in rp.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):develop(a)

if __name__=='__main__':unittest.main(verbosity=2)
