# Degradation-law screening on the digital twin — protocol (written before any screening run)

Question: before a digital twin uses a candidate degradation law to generate high-wear training data, can the
light-wear operating data certify or reject that law, using the same symbolic gate as the symmetry study?

Twin: re-implementation of the frozen two-DOF maintenance protocol (`digital_twin_protocol_frozen.md`):
same plant, nominal twin, noise, learner, alpha grid, train/tune/test wear levels and augmentation block.
Unspecified details fixed here: m1 is tied to ground by (k1, c1), m2 by (k2(w), c2(w)), coupled by (kc, cc);
force on m1; input = 4-tone multisine (0.4-3.5 Hz, random phases, total RMS 1) + PRBS of amplitude 0.2
switching every 0.2 s; initial state N(0, 0.01^2); alpha chosen by one-step TUNE MSE; recursive NRMSE over the
h predicted steps, states normalised by TRAIN sd, origins every 10 steps.

Candidate laws (nominal twin parameters in all): none; damping-only (the paper's wrong twin);
stiffness-only; correct (stiffness loss + damping growth); overstated (both rates doubled).

Consistency defect of law A (no test data): with Delta_hat(x,u) = f_hat(x,u,w_hi) - f_hat(x,u,w_lo) from the base
learner and Delta_A the same difference of the nominal twin's one-step map under law A,
d(A) = RMS(Delta_hat - Delta_A) / RMS(Delta_hat) over TRAIN states and inputs, in TRAIN-standardised units,
with w_lo = 0 and w_hi = the largest observed wear.
Gate: identical to the symmetry gate, accept argmin if d1 <= 0.5 and d1/d2 <= 0.5, otherwise abstain
(gated arm = duplicate).

Seeds 13000-13039 (fresh). Main condition: observed wear up to 0.15, nominal noise.
Sweep: observed wear max in {0.05, 0.10, 0.15, 0.30} x noise multiplier in {1, 4}; 40 seeds each.
Arms per seed: duplicate, own twin, each candidate law, gated, degraded real. Primary metric: duplicate-adjusted
recursive degraded-test NRMSE gain at h = 1, 5, 20; report medians and paired differences.

## History and final design (written before the final run on seeds 15000-15039)
1. The learner-based screen above was run first (seeds 13000-13039, all cells). The ridge learner's degradation
   signature is dominated by estimation error (fit error ~1 in every cell): 0 accepts, 0 wrong, in all 320 seeds.
2. Development on seeds 14000-14009 only: fit the twin's symbolic structure instead of the black-box learner.
   Zero-phase 4th-order Butterworth low-pass (5 Hz) on observed states; accelerations of mass 2 by central
   differences; drop 20 samples at each end; OLS of a2 on per-level [q2, v2] and shared [q2-q1, v2-v1] gives
   kappa_j = k2/m2, gamma_j = c2/m2 per wear level. Without filtering, errors-in-variables bias swamps the trend.
   At nominal noise the stiffness trend is recovered (bias < 0.01) but damping is not (sd ~ 1.2 in relative units).
3. Final structured defect: rho_hat = per-level relative change vs level 0 (stiffness and damping), law prediction
   rho_A likewise; weights 1/se^2 from OLS (se divided by |level-0 estimate|);
   d(A) = sqrt(sum w (rho_hat - rho_A)^2) / sqrt(sum w rho_hat^2); 'none' has d = 1 by construction.
4. Two decisions with the frozen thresholds (d1 <= 0.5, d1/d2 <= 0.5):
   full gate over the five laws; binary screen between the paper's two twins (correct vs damping-only).
   Gated arms fall back to duplication on abstention.
5. Final run: seeds 15000-15039; cells observed-wear max {0.05, 0.15, 0.30} x noise multiplier {0, 0.25, 1, 4}.
   Both the learner-based and the structured defects are recorded on the same seeds.
