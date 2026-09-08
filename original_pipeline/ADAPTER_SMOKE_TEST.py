"""Optional smoke test against the original v5 types; no torch/model execution."""
from pathlib import Path
import tempfile
import contextlib
import io
import numpy as np
from PIL import Image
import ADAPTIVE_DATA as D
from ADAPTIVE_ADAPTER import install
import code_final_fixed_v5 as V

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);data=root/'data';data.mkdir()
    for split in ('train','valid','test'):
        for cls in ('alpha','beta'):
            for i in range(3):
                p=data/split/cls/(str(i)+'.png');p.parent.mkdir(parents=True,exist_ok=True)
                Image.new('L',(8,8),100).save(p)
    with contextlib.redirect_stdout(io.StringIO()):D.prepare(data,root/'manifest')
    install(V,root/'run',False)
    cfg=V.make_cfg('dataset',run_dir=str(root/'run'),split_dir=str(root/'manifest'),tta=1)
    bed=V.load_bed('dataset',cfg.split_dir)
    assert bed.n_classes==2 and bed.split('test').n==6
    assert bed.group_unit=='image'
    store=V.FeatureStore(cfg,bed)
    try:store.spec_features('test','resnet18',1)
    except RuntimeError as e:assert 'blocked' in str(e)
    else:raise AssertionError('Test guard missing')
    assert V.S.cohort_overlap('dataset')['corpus_shared_with']==''
    assert V.PRE.provenance()['channel_audit'].startswith('not performed')
    print('PASS: original v5 Bed5/config compatibility, dynamic counts/classes, native test guard, no old corpus measurements.')
