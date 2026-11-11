"""Regenerate available studies without ever overwriting data/*.csv.
Not all archived results can be regenerated: read docs/REPRODUCIBILITY.md.
"""
from __future__ import annotations
import argparse,sys,subprocess,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def workflow(name, dest):
    py=sys.executable; P=str(ROOT/'harness');D=str(dest)
    cmds={
      'lorenz':[[py,'code/model_class_exp.py',P,D]],
      'duffing':[[py,'code/duffing_exp.py',P,D]],
      'neural':[[py,'code/mlp_verified.py',P,str(dest/'mlp_lorenz_fresh.csv'),'lorenz','1000','1020','fast']],
      'verifier':[[py,'code/verify_exp.py',P,D,mode] for mode in ('main','lownoise','tilt','control')],
      'theory':[[py,'code/sandwich_check.py',P,D],[py,'code/selective_safety_check.py',D],[py,'code/finite_bound_check.py',D],[py,'code/eiv_check.py',D],[py,'code/verify_multidirection.py']],
      'twin':[[py,'code/twin_screen.py',D],[py,'code/twin_screen.py',D,'exposure']],
      'coverage':[[py,'code/coverage_exp.py','18000','18040',str(dest/'coverage_twin.csv')]],
      'closedloop':[[py,'code/closed_loop_exp.py','17000','17040',str(dest/'closed_loop_twin.csv')]],
      'earlier':[[py,'code/extended/coverage_value_sweep.py','--out',str(dest/'coverage_value_sweep.csv')],[py,'code/extended/coverage_proxy_audit.py','--out',str(dest/'coverage_proxy_audit.csv')],[py,'code/extended/noise_value_sweep.py','--out',str(dest/'noise_value_sweep.csv')]],
      'derived':[[py,'code/gate_sweep.py',D],[py,'code/gate_riskcov.py',D]],
    }
    return cmds[name]

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--group',choices=['lorenz','duffing','neural','verifier','theory','twin','coverage','closedloop','earlier','derived','available']);p.add_argument('--out',default='.runs/reproduction');p.add_argument('--no-seed-archived',action='store_true',help='For derived postprocessing, do not copy existing archived prerequisites')
    a=p.parse_args(); choices=['lorenz','duffing','neural','verifier','theory','twin','coverage','closedloop','earlier','derived']
    if a.list or not a.group:
      for k in choices: print(f'{k:12s}', ' | '.join(' '.join(x[1:]) for x in workflow(k,Path('<out>'))))
      print('Some studies require additional raw data or an unavailable original generator. See docs/REPRODUCIBILITY.md.')
      return 0
    dest=(ROOT/a.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    group_list=choices if a.group=='available' else [a.group]
    if 'derived' in group_list and not a.no_seed_archived:
      print('NOTICE: derived calculations will use pre-existing archived prerequisites if unavailable from fresh runs; that is postprocessing, not full raw-data reproduction.')
      for fn in ('verify_main_defects.csv','verify_main_arms.csv','duf_lownoise_defects.csv','verify_tilt_defects.csv','control_lv_defects.csv','twin_screen_laws.csv','twin_screen_arms.csv'):
        if not (dest/fn).exists():shutil.copy2(ROOT/'data'/fn,dest/fn)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    for grp in group_list:
      for cmd in workflow(grp,dest):
        print('RUN',grp, ' '.join(cmd),flush=True)
        proc=subprocess.run(cmd,cwd=ROOT,env=env)
        if proc.returncode:print('FAIL',grp,proc.returncode);return proc.returncode
    print('Completed requested groups. Compare each output with archived data/*.csv; not all published outputs can currently be regenerated.')
    return 0
if __name__=='__main__':sys.exit(main())
