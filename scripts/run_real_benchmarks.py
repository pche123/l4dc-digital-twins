"""Execute four real-data benchmark pipelines if the expected original raw files exist.
No raw files are distributed here; results do not become verified until compared with data/.
"""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'external_data'
CASES={
 'silverbox': (['SNLS80mV.csv'],[['code/silverbox_exp.py','SNLS80mV.csv'],['code/silverbox_imposed.py','SNLS80mV.csv']]),
 'emps': (['DATA_EMPS.mat','DATA_EMPS_PULSES.mat'],[['code/emps_exp.py','<directory>']]),
 'wiener-hammerstein': (['WienerHammerBenchmark.csv'],[['code/wh_exp.py','WienerHammerBenchmark.csv']]),
 'tanks': (['dataBenchmark.csv'],[['code/tanks_exp.py','dataBenchmark.csv']]),
}
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(CASES)+['all'],default='all');p.add_argument('--out',default='.runs/real');p.add_argument('--execute',action='store_true',help='Actually run scripts; default only describes available inputs and commands')
 a=p.parse_args();out=(ROOT/a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 names=list(CASES) if a.dataset=='all' else [a.dataset];bad=0
 for name in names:
  needed,cases=CASES[name];missing=[x for x in needed if not (RAW/x).is_file()]
  print('\n'+name+' needed:',', '.join(needed),'missing:',', '.join(missing) if missing else 'none',flush=True)
  if missing:
   if a.execute:bad+=1
   continue
  for prog,stem in cases:
   arg=RAW if stem=='<directory>' else RAW/stem
   cmd=[sys.executable,prog,str(arg),str(out)]
   print(' '.join(cmd),flush=True)
   if a.execute:
    proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    if proc.returncode:bad+=1
 print('\nAll outputs, if any, went to',out,'and did NOT overwrite archival data/*.csv. Compare with scripts/compare_csv.py and audit preprocessing.')
 return 1 if bad else 0
if __name__=='__main__':sys.exit(main())
