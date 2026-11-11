"""Neural (MLP) learner with the same arms: Lorenz (original and fresh seeds) and Duffing (fresh seeds).

usage: python mlp_verified.py <phase3> <out_csv> <system> <seed_start> <seed_stop>
Arms: base, duplicate, true symmetry, false reflection, opposite-side real trajectory, same-side real trajectory,
and the MLP's own rollout (the neural analogue of the 'own-model' arm).
"""
import sys, json, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
PH3, OUT, SYSTEM, A, B = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
FAST = len(sys.argv) > 6 and sys.argv[6] == 'fast'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(PH3 / 'code'))
from phase3_common import build_lorenz_split, fit_standardizer, lorenz_reservoir, extract_wing_windows
from phase3_learners import fit_mlp_var, forecast_mlp, recursive_nrmse_mlp_batched
M = json.loads((PH3 / 'protocol/phase3_protocol.json').read_text())['lorenz']
LAG, H, GRID = 4, (1, 4, 8), M['mlp_alpha_grid']
sys.argv = ['x', str(PH3), str(OUT.parent / '_t3')]
sp = importlib.util.spec_from_file_location('verify_exp', HERE / 'verify_exp.py')
sys.argv = ['x', str(PH3), str(OUT.parent / '_t3'), 'none']
ve = importlib.util.module_from_spec(sp); sp.loader.exec_module(ve)

def fit_grid(train, tune, seed):
    rec = []
    for a in GRID:
        mdl = fit_mlp_var(train, LAG, float(a), seed=seed); rec.append((float(recursive_nrmse_mlp_batched(mdl, tune, LAG, H)), float(a), mdl))
    rec.sort(key=lambda r: (r[0], r[1])); return rec

def one_seed(seed):
    if SYSTEM == 'lorenz':
        raw = build_lorenz_split(seed, 36, 5, 5, 10, 0.015, 0.95)
        same = extract_wing_windows(lorenz_reservoir(700000 + seed * 31, obs_noise=0.015), 36, +1, 1, min_fraction=.95, min_separation=36)[0]
        true, false, dim = ve.LOR_TRUE, ve.LOR_FALSE, 3
    else:
        raw = ve.duf_split(seed, 40, 0.01, 0.0)
        same = ve.duf_split(seed + 500, 40, 0.01, 0.0)['train'][0]
        true, false, dim = ve.DUF_TRUE, ve.DUF_FALSE, 2
    cands = ve.flips(dim)
    mu, sd = fit_standardizer(raw['train']); st = lambda w: (w - mu) / sd
    z = {k: ([st(w) for w in v] if isinstance(v, list) else st(v)) for k, v in raw.items()}
    src = z['train'][0]
    base_rec = fit_grid(z['train'], z['tune'], seed * 1000 + 1)
    own = np.concatenate([src[:, :LAG], forecast_mlp(base_rec[0][2], src[:, :LAG], src.shape[1] - LAG, LAG)], axis=1)
    arms = {'base': None, 'duplicate': src.copy(), 'true': st(cands[true] @ raw['train'][0]), 'false': st(cands[false] @ raw['train'][0]),
            'opposite': z['extra_real'], 'same_side': st(same), 'own_mlp': own}
    if FAST: arms = {k: arms[k] for k in ('base', 'duplicate', 'true', 'false', 'opposite')}
    rows = []
    for j, (arm, cand) in enumerate(arms.items()):
        tr = z['train'] if cand is None else z['train'] + [cand]
        rec = base_rec if arm == 'base' else fit_grid(tr, z['tune'], seed * 1000 + 101 + j)
        _, alpha, mdl = rec[0]
        row = dict(system=SYSTEM, seed=seed, learner='mlp', arm=arm, alpha=alpha,
                   headroom_loss=recursive_nrmse_mlp_batched(mdl, z['headroom'], LAG, H),
                   balanced_loss=recursive_nrmse_mlp_batched(mdl, z['test'], LAG, H))
        for h in H:
            row[f'balanced_h{h}'] = recursive_nrmse_mlp_batched(mdl, z['test'], LAG, (h,))
            row[f'negative_h{h}'] = recursive_nrmse_mlp_batched(mdl, z['test'][10:], LAG, (h,))
            row[f'positive_h{h}'] = recursive_nrmse_mlp_batched(mdl, z['test'][:10], LAG, (h,))
        rows.append(row)
    return rows

if __name__ == '__main__':
    from joblib import Parallel, delayed
    outs = Parallel(n_jobs=2, verbose=3)(delayed(one_seed)(s) for s in range(A, B))
    pd.DataFrame([r for o in outs for r in o]).to_csv(OUT, index=False); print('done', SYSTEM, A, B)
