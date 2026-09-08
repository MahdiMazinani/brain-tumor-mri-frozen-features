import contextlib
import io
import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import ADAPTIVE_DATA as D
import ADAPTIVE_RUN as R

class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.data=self.base/'images';self.data.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def img(self,path):
        p=self.data/path;p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('L',(8,8),100).save(p);return p
    def build(self,**kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return D.prepare(self.data,self.base/'prepared',**kw)
    def add(self,split,counts=(7,11)):
        for cls,n in zip(('a','b'),counts):
            for i in range(n):self.img(Path(split)/cls/(str(i)+'.png'))
    def test_preserve_all_splits(self):
        for s in ('Training','Validation','Testing'):self.add(s)
        m=self.build();self.assertEqual(len(m['rows']),54);self.assertEqual(m['generated_splits'],[])
        self.assertEqual(len([r for r in m['rows'] if r['split']=='test']),18)
        # No downsampling or equalization of unequal classes.
        self.assertEqual(D.Split(m,'train').class_counts(),{'a':7,'b':11})
    def test_train_test_carve_valid_only(self):
        self.add('Train',(15,20));self.add('Test',(4,6))
        m=self.build();self.assertEqual(m['generated_splits'],['valid'])
        self.assertEqual(D.Split(m,'test').n,10);self.assertEqual(len(m['rows']),45)
    def test_flat(self):
        self.add('',(15,20));m=self.build()
        self.assertEqual(m['generated_splits'],['test','valid'])
        self.assertEqual(sum(D.Split(m,s).n for s in ('train','valid','test')),35)
    def test_keep_valid_when_test_missing(self):
        self.add('train',(12,12));self.add('val',(3,4));m=self.build()
        self.assertEqual(m['generated_splits'],['test']);self.assertEqual(D.Split(m,'valid').n,7)
    def test_identical_content_not_removed(self):
        for s in ('train','valid','test'):self.add(s,(3,3))
        m=self.build();self.assertEqual(len(m['rows']),18)
        self.assertEqual(len({r['sha256'] for r in m['rows']}),1)
    def csv(self,rows):
        p=self.base/'map.csv'
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','label','split','group']);w.writeheader();w.writerows(rows)
        return p
    def test_csv_groups(self):
        rows=[]
        for cls in ('a','b'):
            for group in range(10):
                for image in range(2):
                    p=self.img(Path('arbitrary')/(f'{cls}_{group}_{image}.png'))
                    rows.append(dict(path=str(p),label=cls,split='train',group=f'{cls}_{group}'))
        m=self.build(csv_path=self.csv(rows));self.assertEqual(m['group_unit'],'supplied_group')
        groups={s:set(D.Split(m,s).groups) for s in ('train','valid','test')}
        self.assertFalse(groups['train']&groups['test']);self.assertFalse(groups['train']&groups['valid'])
    def test_cross_split_group_rejected(self):
        rows=[]
        for split in ('train','test'):
            p=self.img(Path(split)/'a.png');rows.append(dict(path=str(p),label='a',split=split,group='same'))
        with self.assertRaises(D.DataError):self.build(csv_path=self.csv(rows))
    def test_missing_class_rejected(self):
        self.add('train');self.add('valid');self.img(Path('test')/'a'/'x.png')
        with self.assertRaises(D.DataError):self.build()
    def test_content_mutation_detected(self):
        for s in ('train','valid','test'):self.add(s,(3,3))
        m=self.build();D.check_content(m,('train','valid'))
        r=next(r for r in m['rows'] if r['split']=='train');Path(r['path']).write_bytes(b'changed')
        with self.assertRaises(D.DataError):D.check_content(m,('train',))
    def test_seed_reproducible(self):
        self.add('',(20,20));m=self.build()
        with contextlib.redirect_stdout(io.StringIO()):n=D.prepare(self.data,self.base/'second')
        self.assertEqual(m['rows'],n['rows'])
    def test_high_bit_depth_rejected(self):
        for s in ('train','valid','test'):self.add(s,(3,3))
        Image.new('I;16',(8,8)).save(self.data/'train'/'a'/'high.tif')
        with self.assertRaises(D.DataError):self.build()
    def test_development_guard(self):
        with self.assertRaises(R.ProtocolError):R.guard('test')
        R.guard('valid')
    def test_holm(self):self.assertEqual(R.holm([.001,.04]),[.002,.04])
    def test_paired(self):
        r=R.paired([0,1],[0,1],[0,1],['p1','p2'],100)
        self.assertEqual(r['delta_accuracy'],0);self.assertEqual(r['p_raw'],1)

if __name__=='__main__':unittest.main(verbosity=2)
