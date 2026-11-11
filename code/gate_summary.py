import pandas as pd, numpy as np, sys
def gate(df, tau=0.5, r=0.5):
    out=[]; keys=[c for c in ['system','noise','eps','seed','verifier'] if c in df]
    for k,g in df.groupby(keys):
        g=g.sort_values('defect'); d1,d2=g.defect.iloc[0],g.defect.iloc[1]; best=g['transform'].iloc[0]; bt=int(g.is_true.iloc[0])
        out.append({**dict(zip(keys,k)), 'best':best,'d1':d1,'d2':d2,'true_d':g.loc[g.is_true==1,'defect'].min(),
                    'acc':int(d1<=tau and d1<=r*d2),'racc':int(d1<=r*d2),'best_true':bt})
    return pd.DataFrame(out)
if __name__=='__main__':
    for f in sys.argv[1:]:
        G=gate(pd.read_csv(f)); keys=[c for c in ['system','noise','eps','verifier'] if c in G]
        s=G.groupby(keys).apply(lambda g: pd.Series(dict(n=len(g), accept=g.acc.sum(), wrong=(g.acc*(1-g.best_true)).sum(),
            ratio_only_accept=g.racc.sum(), ratio_only_wrong=(g.racc*(1-g.best_true)).sum(), med_true_defect=round(g.true_d.median(),3), med_d1=round(g.d1.median(),3))),include_groups=False)
        print(f); print(s.to_string()); print()
