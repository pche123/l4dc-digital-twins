"""Frozen 40-seed predictive-maintenance digital-twin study (Section 5, Figure 4a-b, Supplement N).

Follows the frozen protocol: seeds 12000-12039; TRAIN at wear {0, .05, .10, .15}; TUNE at {.025, .125};
healthy TEST at {.05, .15}; degraded TEST at {.55, .70, .85} (three trajectories each). Ridge alpha is chosen
once per seed on TUNE and frozen. Every arm adds two trajectories at high wear {.65, .85} with matched inputs:
duplicate (two TRAIN trajectories repeated), own_twin (the fitted learner's recursive rollouts),
wrong_physics (damping-only twin), correct_physics (correct twin), same_regime_real (fresh plant data at light
wear) and degraded_real (fresh plant data at high wear, a positive control).
Outputs: data/digital_twin_results.csv, digital_twin_gains.csv, digital_twin_self_label.csv, digital_twin_summary.csv.

usage: python code/twin_frozen_study.py data
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from twin_screen import ALPHAS, DT, T, Learner, excitation, feats, params, simulate, step, traj

SEEDS = range(12000, 12040)
H = (1, 5, 20)
ARMS = ('duplicate', 'own_twin', 'wrong_physics', 'correct_physics', 'same_regime_real', 'degraded_real')
WEAR_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def fit(L, data, alpha):
    """Ridge fit; a block may carry its own targets 'Y' (used for self-labels)."""
    F = L.z(np.vstack([feats(d['X'][:-1], d['U'][:-1], d['w']) for d in data]))
    Y = np.vstack([d['Y'] if 'Y' in d else d['X'][1:] for d in data])
    return np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)


def diagnostic_rms(next_state, w):
    """RMS of v2 over the last 5 s of a 10 s diagnostic run from rest."""
    t = np.arange(int(10 / DT)) * DT
    u = np.sin(2 * np.pi * 2.40 * t) + 0.20 * np.sin(2 * np.pi * 1.10 * t)
    x = np.zeros(4); v2 = []
    for k in range(len(t)):
        x = np.clip(next_state(x, u[k], w), -1e3, 1e3); v2.append(x[3])
    v2 = np.array(v2[len(t) // 2:])
    return float(np.sqrt(np.mean(v2 ** 2)))


def crossing(next_state, threshold):
    for w in WEAR_GRID:
        if diagnostic_rms(next_state, w) > threshold: return float(w)
    return 1.05


def one_seed(seed):
    rng = np.random.default_rng(seed)
    train = [traj('plant', w, rng) for w in (0.0, 0.05, 0.10, 0.15)]
    tune = [traj('plant', w, rng) for w in (0.025, 0.125)]
    healthy = [traj('plant', w, rng) for w in (0.05, 0.15)]
    degraded = [traj('plant', w, rng) for w in (0.55, 0.70, 0.85) for _ in range(3)]
    blocks = [dict(U=excitation(rng), x0=rng.normal(0, .01, 4), w=w) for w in (0.65, 0.85)]
    L = Learner(train)
    alpha = min(ALPHAS, key=lambda a: L.onestep(fit(L, train, a), tune))
    B0 = fit(L, train, alpha)
    law = lambda kind: [dict(X=simulate(kind, b['w'], b['U'], b['x0']), U=b['U'], w=b['w']) for b in blocks]
    arms = {'base': [],
            'duplicate': train[-2:],
            'own_twin': [dict(X=L.rollout(B0, b['x0'], b['U'], b['w']), U=b['U'], w=b['w']) for b in blocks],
            'wrong_physics': law('damping-only'),
            'correct_physics': law('correct'),
            'same_regime_real': [dict(X=simulate('plant', w, b['U'], b['x0'], rng), U=b['U'], w=w) for b, w in zip(blocks, (0.05, 0.15))],
            'degraded_real': [dict(X=simulate('plant', b['w'], b['U'], b['x0'], rng), U=b['U'], w=b['w']) for b in blocks]}
    plant_next = lambda x, u, w: step(x, u, params('plant', w))
    threshold = 1.5 * diagnostic_rms(plant_next, 0.0)
    true_cross = crossing(plant_next, threshold)
    rows = []
    for arm, extra in arms.items():
        B = fit(L, train + extra, alpha)
        model_next = lambda x, u, w, B=B: L.predict(B, x[None], np.array([u]), w)[0]
        pred_cross = crossing(model_next, threshold)
        r = dict(seed=seed, arm=arm, alpha=alpha, param_update_rel=float(np.linalg.norm(B - B0) / max(np.linalg.norm(B0), 1e-12)),
                 healthy_h20_nrmse=L.nrmse(B, healthy, 20), threshold=threshold, true_cross_wear=true_cross)
        for h in H: r[f'degraded_h{h}_nrmse'] = L.nrmse(B, degraded, h)
        r.update(pred_cross_wear=pred_cross, cross_abs_error=abs(pred_cross - true_cross))
        rows.append(r)
    # same states, two labellings: the learner's own one-step predictions versus the correct degradation law
    phys = arms['correct_physics']
    selfl = [dict(X=d['X'], U=d['U'], w=d['w'], Y=L.predict(B0, d['X'][:-1], d['U'][:-1], d['w'])) for d in phys]
    Bs, Bp = fit(L, train + selfl, alpha), fit(L, train + phys, alpha)
    sl = dict(seed=seed, alpha=alpha, self_update_rel=float(np.linalg.norm(Bs - B0) / np.linalg.norm(B0)),
              phys_update_rel=float(np.linalg.norm(Bp - B0) / np.linalg.norm(B0)))
    for h in H:
        sl[f'base_h{h}'], sl[f'self_h{h}'], sl[f'phys_h{h}'] = (L.nrmse(B, degraded, h) for B in (B0, Bs, Bp))
    return rows, sl


def bootstrap(x, rng, n=10000):
    x = np.asarray(x, float); means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.quantile(means, [0.025, 0.975])


def main(out):
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=2)(delayed(one_seed)(s) for s in SEEDS)
    R = pd.DataFrame([r for rr, _ in res for r in rr]); SL = pd.DataFrame([s for _, s in res])
    d = R.set_index(['seed', 'arm'])
    G = []
    for arm in ARMS:
        if arm == 'duplicate': continue
        for seed in SEEDS:
            a, b = d.loc[(seed, arm)], d.loc[(seed, 'duplicate')]
            G.append(dict(seed=seed, arm=arm, **{f'gain_h{h}': b[f'degraded_h{h}_nrmse'] - a[f'degraded_h{h}_nrmse'] for h in H},
                          healthy_delta_h20=b.healthy_h20_nrmse - a.healthy_h20_nrmse,
                          cross_error_improvement=b.cross_abs_error - a.cross_abs_error, param_update_rel=a.param_update_rel))
    G = pd.DataFrame(G); rng = np.random.default_rng(20261109); S = []
    for arm, g in G.groupby('arm', sort=False):
        for m in ['gain_h1', 'gain_h5', 'gain_h20', 'healthy_delta_h20', 'cross_error_improvement', 'param_update_rel']:
            lo, hi = bootstrap(g[m], rng)
            S.append(dict(arm=arm, metric=m, mean=g[m].mean(), ci_lo=lo, ci_hi=hi, median=g[m].median(), n=len(g)))
    out = Path(out)
    R.to_csv(out / 'digital_twin_results.csv', index=False); G.to_csv(out / 'digital_twin_gains.csv', index=False)
    SL.to_csv(out / 'digital_twin_self_label.csv', index=False); pd.DataFrame(S).to_csv(out / 'digital_twin_summary.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1])
