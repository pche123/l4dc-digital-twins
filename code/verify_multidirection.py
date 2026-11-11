"""Numerical check of the multi-direction exposure corollary (exact Woodbury form)."""
import numpy as np
rng=np.random.default_rng(0); worst=0
for trial in range(2000):
    p=int(rng.integers(3,12)); r=int(rng.integers(1,min(p,5)+1))
    X=rng.normal(size=(30,p)); H=X.T@X+rng.uniform(.01,2)*np.eye(p)
    Q=(lambda A:A@A.T)(rng.normal(size=(p,p))); e=rng.normal(size=p); g=rng.normal(size=p)
    V=rng.normal(size=(p,r)); R=(lambda A:A@A.T+.1*np.eye(r))(rng.normal(size=(r,r)))
    G=V@R@V.T; eG=np.linalg.solve(H+G,H@e+G@g)
    Hi=np.linalg.inv(H); h=V.T@Hi@V; s=V.T@(e-g); M=V.T@Hi@Q@e; C=V.T@Hi@Q@Hi@V
    K=np.linalg.inv(np.linalg.inv(R)+h)
    pred=-2*s@K@M+s@K@C@K@s; act=eG@Q@eG-e@Q@e
    worst=max(worst,abs(pred-act)/(abs(act)+1e-9))
print('max relative error over 2000 random instances:',worst)
