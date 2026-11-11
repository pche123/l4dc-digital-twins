"""Median unseen-half (negative-velocity) rollout errors for the EMPS study.   usage: python code/emps_summary.py data"""
import sys, pandas as pd
from pathlib import Path
r = pd.read_csv(Path(sys.argv[1]) / 'emps_results.csv'); g = pd.read_csv(Path(sys.argv[1]) / 'emps_gate.csv')
print(g.groupby(['verifier', 'candidate'])[['defect', 'accepted', 'e_X', 'e_SX']].median().round(3).to_string(), '\n')
for reg in ['negative (unseen)', 'positive (seen)']:
    x = r[r.region == reg].groupby(['learner', 'arm'])[['nrmse_h1', 'nrmse_h10', 'nrmse_h50']].median()
    print('==', reg); print(x.round(3).to_string(), '\n')
