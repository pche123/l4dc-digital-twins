#!/usr/bin/env python3
"""Check the released paper dependencies and run the mathematical identity tests."""
from pathlib import Path
import argparse, subprocess, sys, re
ROOT=Path(__file__).resolve().parents[1]

def files():
    basics=['main.tex','supplement.tex','refs.bib','l4dc2027.cls','jmlr.cls','jmlrutils.sty','main.bbl','paper.pdf','harness/code/phase3_common.py']
    raw=(ROOT/'main.tex').read_text()+(ROOT/'supplement.tex').read_text()
    figs=['figures/'+x for x in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}',raw)]
    absent=[f for f in basics+figs if not (ROOT/f).is_file()]
    csvs=sorted((ROOT/'data').glob('*.csv'))
    print(f'Paper dependencies: {len(basics)+len(figs)}; absent: {len(absent)}')
    print(f'Active archived result CSVs: {len(csvs)}')
    if absent: print('Missing:',*absent,sep='\n  ')
    return len(absent)==0 and len(csvs)==40

def math():
    cmd=[sys.executable,'code/math_audit.py']
    p=subprocess.run(cmd,cwd=ROOT,timeout=240)
    return p.returncode==0

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--files',action='store_true')
    a.add_argument('--math',action='store_true')
    opt=a.parse_args()
    commands=[files,math] if not (opt.files or opt.math) else ([files] if opt.files else [])+([math] if opt.math else [])
    sys.exit(0 if all(f() for f in commands) else 1)
