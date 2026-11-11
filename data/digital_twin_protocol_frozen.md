# Frozen protocol — predictive-maintenance digital-twin case study

Frozen before outcome-producing execution.

## Scientific question
Does a learned digital twin trained only on healthy/light-wear operation add useful information when its own high-wear rollouts are fed back to the same learner, or is deployment value obtained only when synthetic trajectories contain externally specified degradation dynamics absent from the fitted learner?

## Physical asset
Two-degree-of-freedom forced mass–spring–damper system with state x=(q1,q2,v1,v2), force input u on mass 1, and scalar wear index w in [0,1].

True plant parameters:
- m1=1.02, m2=0.68
- k1=82, kc=58, k2(w)=103*(1-0.45*w)
- c1=0.75, cc=0.40, c2(w)=0.55*(1+1.20*w)

Nominal physics twin parameters:
- m1=1.00, m2=0.70
- k1=80, kc=60, k2=100, c1=0.80, cc=0.35, c2=0.60
- correct degradation hypothesis: k2(w)=100*(1-0.45*w), c2(w)=0.60*(1+1.20*w)
- wrong degradation hypothesis: damping-only wear, k2(w)=100, c2(w)=0.60*(1+1.20*w)

Continuous dynamics are integrated with fixed-step RK4, dt=0.02 s. Input is a seed-specific multisine plus a low-amplitude PRBS term. Sensor noise is added only to real-plant observations: position sd=0.0025 and velocity sd=0.010.

## Data firewall
40 independent seeds, 12000–12039. For each seed:
- TRAIN: four true-plant trajectories at w={0,0.05,0.10,0.15}; length 300 states each.
- TUNE: two independent true-plant trajectories at w={0.025,0.125}.
- healthy TEST: independent true-plant trajectories at w={0.05,0.15}.
- degraded TEST: three wear levels w={0.55,0.70,0.85}, three independent trajectories per level.
- augmentation block size: two trajectories at high wear w={0.65,0.85}, matched across synthetic arms.
No TEST trajectories are used for tuning or arm construction.

## Learner
State-action ridge predictor x_{t+1}=Theta^T phi(x_t,u_t,w), with
phi=[x, u, w*x, w*u, w, 1].
Feature means/scales are estimated from TRAIN only (intercept unscaled). Ridge alpha is selected once per seed on TUNE from {1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1}, then frozen for every augmentation arm.

## Arms
All refits use the same frozen alpha and equal added row count.
1. duplicate: matched TRAIN rows duplicated.
2. own_twin: recursive high-wear rollouts from the fitted base learner itself. Targets are exactly its own one-step predictions.
3. wrong_physics: nominal physical twin with damping-only wear.
4. correct_physics: nominal physical twin with the correct stiffness-loss + damping-growth wear law.
5. same_regime_real: fresh true-plant lightly worn trajectories at w={0.05,0.15}.
6. degraded_real: fresh true-plant degraded trajectories at w={0.65,0.85} (positive-control oracle for information availability).

## Primary metrics
- Recursive degraded-TEST NRMSE at horizons h={1,5,20}; state errors normalized by TRAIN state standard deviations and averaged over states/origins.
- Duplicate-adjusted gain = NRMSE_duplicate - NRMSE_arm (positive is better).
- Relative parameter update ||Theta_arm-Theta_base||_F / max(||Theta_base||_F,1e-12).

## Secondary maintenance metric
A standardized diagnostic excitation u(t)=sin(2*pi*2.40*t)+0.20*sin(2*pi*1.10*t) is applied for 10 s from zero state over wear grid {0,0.05,...,1}. Condition indicator is RMS v2 over the final 5 s. The maintenance threshold is fixed at 1.5 times the TRUE plant's w=0 RMS. The predicted crossing wear is the first grid point exceeding this fixed threshold; if none, record 1.05. Report absolute crossing-wear error.

## Inference
Report all 40 seeds. Paired nonparametric bootstrap, 10,000 resamples, seed as the resampling unit, fixed RNG seed 20261109. No seed filtering and no post-outcome hyperparameter changes.

## Interpretation guardrails
- own_twin is a direct self-consistency test, not a claim that all digital twins are redundant.
- correct_physics and wrong_physics differ in the degradation mechanism, not merely sample count.
- degraded_real is a positive-control upper benchmark, not a deployable synthetic method.
- If wrong_physics helps, report it; support expansion and physical correctness are separate mechanisms.
