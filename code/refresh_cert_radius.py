"""Recompute the finite-sample certificate radius with declared coefficient bounds.

The archived 6000 trial rows contain r_exact computed with oracle coefficient
norms. Reproduce each seeded design and rescale that radius by B_new / B_old;
all other statistics (d_exact, practical gate decisions) remain unchanged.
This shortcut is exact because r=B*(LX+LSX)/norm(fhat), and only B changed.
One can also rerun selective_safety_check.py in full to independently verify.
"""
import sys, itertools, numpy as np, pandas as pd
from pathlib import Path

POW=[c for d in (1,2,3) for c in itertools.combinations_with_replacement(range(2),d)]
def F(X): return np.hstack([np.ones((len(X),1))]+[np.prod(X[:, list(c)],1,keepdims=True) for c in POW])
M=np.array([2.,2.]); ORACLE=np.array([1.,np.sqrt(2.)]); LAM=1e-3; DELTA=.05

def main(inp,out):
    D=pd.read_csv(inp)
    assert len(D)==6000, 'The archived table must have 500 draws per setting.'
    rng=np.random.default_rng(2027)
    ix=0; rn=[]
    for n,sigma,wells in itertools.product([200,2000],[.01,.05,.2],[1,2]):
        for _ in range(500):
            row=D.iloc[ix]
            assert (int(row.n),float(row.sigma),int(row.wells)) == (n,sigma,wells)
            xs=rng.uniform(.35,1.4,n)*(rng.choice([-1,1],n) if wells==2 else 1)
            vs=rng.uniform(-.5,.5,n)
            X=np.stack([xs,vs],1)
            rng.normal(size=(n,2))  # consume the same target-noise draws as the original script
            P=F(X); p=P.shape[1]; H=P.T@P+LAM*np.eye(p)
            sign,ld=np.linalg.slogdet(H)
            assert sign>0
            rad=sigma*np.sqrt(ld-p*np.log(LAM)+2*np.log(2/DELTA))
            bold=np.linalg.norm(rad+np.sqrt(LAM)*ORACLE)
            bnew=np.linalg.norm(rad+np.sqrt(LAM)*M)
            rn.append(float(row.r_exact)*bnew/bold)
            ix+=1
    D['r_exact']=rn
    # Only the exact candidate had any certified acceptances in the archived tests.
    D['cert_accept_true']=(D['d_exact']+D['r_exact']<=0.5)
    assert D['cert_wrong'].sum()==0
    assert D['holds_all'].all()
    D.to_csv(out,index=False)
    G=D.groupby(['wells','n','sigma'],sort=True).agg(bound_holds=('holds_all','mean'),r_exact=('r_exact','median'),d_exact=('d_exact','median'),std_accept=('std_accept','mean'),std_wrong=('std_wrong','mean'),cert_accept=('cert_accept_true','mean'),cert_wrong=('cert_wrong','mean'))
    print(G.round(4).to_string())

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
