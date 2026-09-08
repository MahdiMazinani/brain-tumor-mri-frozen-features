"""Align old/new OOF outputs for majority-vote and clustered analyses."""
import argparse
from pathlib import Path
import pandas as pd
from OOF_ANALYSIS import validate
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)
if out.exists():raise ValueError('Output already exists')
dfs=[pd.read_csv(f,dtype={'patient_id':str,'path':str}) for f in a.inputs]
required=['fold','method','path','patient_id','y_true','y_pred']
# Old OOF files have no probabilities. Do not invent probabilities or zero-fill them.
probsets=[set(c for c in d if c.startswith('p_')) for d in dfs]
keep=required+(sorted(probsets[0]) if probsets[0] and all(s==probsets[0] for s in probsets) else [])
combined=pd.concat([d[keep] for d in dfs],ignore_index=True);validate(combined)
out.parent.mkdir(parents=True,exist_ok=True);combined.to_csv(out,index=False)
print('MERGED:',out,'| probability columns retained:',len(keep)>len(required), '| originals unchanged')
