# Across-XR alignment — results certificate (zero-shot 3 seeds; C1, C1-full, Z-676, C2-hi seed 1; C2-lo 2 seeds; P3 and the half arm in progress)

## C2-lo, seed 2 — the headline replicates; the rotation does not (2026-09-11 03:30)

Row `f55fd57db721` at `517cdaa57b`, epoch 116/120, verification AUC 0.701 (seed 1: 0.699).
Gate PASS at 1.6e-4.

| C2-lo | A1 | 10-min | A2′ − A1 | A2 − A1 | A2-null − A1 |
| --- | --- | --- | --- | --- | --- |
| seed 1 | 0.368 [0.318, 0.424] | 0.693 | **+0.148** [+0.124, +0.168] | −0.008 | −0.286 |
| seed 2 | 0.378 [0.314, 0.450] | 0.709 | **−0.004** [−0.016, +0.008] | −0.002 | −0.210 |
| two seeds, pooled over users | **0.373 [0.319, 0.437]** | 0.701 | per seed only — the pooled +0.072 averages two different solutions | −0.005 [−0.016, +0.007] | −0.248 |

**C2-lo − zero-shot(4096) = +0.143 [+0.099, +0.187] over two paired seeds**; C2-hi − C2-lo
stands at −0.061 (one paired seed). The 0.37 headline and the +0.14 exposure gain are
replicated.

**The orthogonal structure is run-dependent.** At the same configuration, seed 1 carries
+0.148 of recoverable orthogonal structure and seed 2 none; the permuted null costs 0.21-0.29
on both, so the person-specific structure is equally strong — what differs is whether any of
it is an orthogonal difference between applications. Across the exposed, trained-out arms
the ceiling reads C1-full +0.089, C2-hi +0.002, C2-lo +0.148 / −0.004: **present in some
runs and absent in others, not a property of scale, exposure or budget but of the solution a
run converges to.** The "aligned ceiling rises monotonically with pretraining scale" sentence
below rested on one seed per point and is withdrawn with it; "C2-hi already at its ceiling"
was the common case, not an anomaly. What holds on every instrument and every seed (nine
checkpoints): **the honest train-user fit never carries** (A2 − A1 ≤ 0 everywhere), and the
corpus-limitation sentence stands as the whole alignment result — with the addition that
even the test-fitted ceiling Schach reported is run-dependent here, so it was never a target.
Seed 3 (queued) decides how often the structure appears; the half arm's A2′ reading is
dropped as uninformative.

---

## C2-lo, seed 1 — the dose prediction is falsified, and the orthogonal structure appears (2026-09-11 00:40)

Row `a34056530b9d` at `517cdaa57b`: BOXRR (all 4,020) + alyx + Across-XR 0-22 (3,095
training identities, 540,107 windows of which 20,896 Across-XR — **dose 3.0%**), validation
1,033 (the 25% draw plus 23-31 explicit), evaluation 32-48; epoch 119 of 120. Cross-application
verification AUC on the 17 **0.699** (C2-hi 0.683, Z-676 0.556, zero-shot 0.569-0.588). Gate
PASS at 1.1e-4.

| C2-lo, seed 1 | A0 within | **A1 cross** | CI95 | 10-min cross | A2′ − A1 | A2 − A1 | A2-null − A1 | A2-full − A2 | m-curve |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BOXRR 4020 + alyx + Across-XR 0-22 | 0.624 | **0.368** | [0.318, 0.424] | **0.693** | **+0.148 [+0.124, +0.168]** | −0.008 [−0.029, +0.015] | −0.286 | −0.042 | flat (range 0.023), m*=16 |

| registered contrast (paired on the same 17 users) | measured | registered | verdict |
| --- | --- | --- | --- |
| **C2-lo − zero-shot(4096)** | **+0.137 [+0.084, +0.189]** | "above +0.05 is informative: a 3% dose already carries" | **a 3% dose carries, decisively** — the largest cross-application gain in the programme, on Schach's own protocol plus 4,096 pretraining identities |
| **C2-hi − C2-lo** | **−0.061 [−0.099, −0.026]** | predicted positive, "the dose effect itself" | **FALSIFIED, whole interval on the wrong side.** The 14% arm on 495 identities loses to the 3% arm on 3,095. Dose was not what limited C2; the two arms differ in dose *and* identity count and the pair cannot separate them, but both registered directions assumed dose was binding and both were wrong. A null at 3% would have been misread as "the objective barely saw Across-XR" — the registered reading — and the data say the opposite |
| A2′ − A1 on C2-lo | **+0.148 [+0.124, +0.168]** | mechanism hypothesis | **the orthogonal structure APPEARS.** The test-fitted ceiling lands at almost exactly the +0.15 the zero-shot registration expected and never saw. It is absent on zero-shot (+0.026), on 23 identities with exposure (+0.011), on 495 identities with exposure (+0.002), and present on 3,095 identities with exposure — so it is a property of a large-identity model that has seen the applications, not of exposure alone and not of the task |
| A2 − A1 on C2-lo | −0.008 [−0.029, +0.015] | the result arm | **the honest train-user fit still does not carry** — Schach's situation reproduced on a head-only model: the rotation exists, and 32 correspondences in 128-d cannot learn it for unseen people. Their section 8's future work is answered on this instrument: *not with this many training users* |
| A2-null − A1 | −0.286 | ≤ +0.03 | the person-specific structure is the strongest in the programme |

## C1-full, seed 1 — the budget-matched C1 (2026-09-11 01:15)

Row `9bc8a8908f69` at `517cdaa57b`: C1 at the 120-epoch cap with patience 0; selected
epoch 102; cross-application verification AUC 0.572 (C1 at epoch 15: 0.528). Gate PASS
at 2.0e-6. Selection diagnostic on the verification columns, as the row records it:
`selected_test_acc` 0.545 against `final_test_acc` 0.540 — nine-user selection over 120
evaluations bought +0.005 there.

| C1-full | A0 | **A1** | CI95 | 10-min | A2′ − A1 | A2 − A1 | A2-null − A1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 23 identities, 120 epochs | 0.291 | **0.164** | [0.128, 0.205] | 0.324 | **+0.089 [+0.077, +0.102]** | −0.004 | −0.106 |

| registered (Amendment 3) | measured | verdict |
| --- | --- | --- |
| C1-full − C1 in [0, +0.08] | **+0.033** | inside the band: C1 was under-trained by about 0.03, and nine-user selection did no measurable harm beyond that |
| C1-full A1 below zero-shot 0.234 | 0.164 [0.128, 0.205] | holds; falsifier not fired. Their protocol on our model, trained out, is still 0.07 below zero-shot from 4,096 identities of other activities |
| alignment on C1 unchanged (A2′ − A1 < +0.05) | **+0.089** | **fails — and it is the informative failure.** Trained out, the 23-identity exposed model carries the orthogonal structure C1 at epoch 15 did not (+0.011) |

**The alignment structure across every instrument, with the budget each arm actually ran
(read from `best_epoch` / `epochs_run`) and the headroom-normalised form
`(A2′ − A1) / (1 − A1)` beside the raw one:**

| arm | identities | exposure | selected / run epochs | A1 | A2′ − A1 raw | normalised |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 23 | yes | 15 / 30 (patience fired; under-trained) | 0.131 | +0.011 | 0.013 |
| C1-full | 23 | yes | 102 / 120 | 0.164 | +0.089 | 0.106 |
| Z-676 | 676 | no | 119 / 120 | 0.218 | +0.011 | 0.014 |
| C2-hi | 676 | yes | **120 / 120** (full budget, censored) | 0.307 | **+0.002** | 0.003 |
| zero-shot (3 seeds) | 4,096 | no | 120 / 116 / 118 of 120 | 0.234 | +0.026 | 0.034 |
| C2-lo | 4,096 | yes | 119 / 120 | 0.368 | +0.148 | 0.234 |

**C2-hi ran the full budget and sits at +0.002 between C1-full's +0.089 and C2-lo's
+0.148**, so "scale amplifies it" is withdrawn — and the gain column is the wrong thing to
read. Put the aligned ceiling itself in the table (Coordinator):

| exposed, trained-out arm | identities | A1 (unaligned) | A2′ (aligned ceiling) | gain |
| --- | --- | --- | --- | --- |
| C1-full | 23 | 0.164 | 0.253 | +0.089 |
| C2-hi | 676 | 0.307 | 0.309 | +0.002 |
| C2-lo | 4,096 | 0.368 | 0.516 | +0.148 |

**A1 is monotone in identity count and A2′ is monotone in identity count; only their
difference is not** — the non-monotone gain is two monotone series with different slopes
subtracted, and the fact underneath it is that **C2-hi's unaligned embedding already sits at
its own aligned ceiling** (0.307 against 0.309), so there was nothing for a rotation to
recover. The sentence that all three points support: **the aligned ceiling rises
monotonically with pretraining scale; what varies is how much of that ceiling the
unaligned embedding has already reached** — and we have no account of why C2-hi reached
all of it. The dip is not seed noise: the zero-shot arm measures the seed spread of A2′ − A1
at about 0.003 (+0.024 / +0.029 / +0.025), so +0.002 against +0.089 is far outside it. It
is not chased with card time; no claim depends on it. For the paper, the sharper statement
is about what the representation can support: **at 4,096 identities there is 0.148 of
recoverable structure on the embedding (ceiling 0.516) and 32 correspondences cannot reach
it** — the cleanest form of the corpus limitation. The rotation needs *exposure and enough
training* (C1 → C1-full moved it eight-fold at fixed identities and exposure) and is absent
without exposure at any scale. The headroom-normalised column keeps the same ordering.
Selection inflation on the verification
columns, measured on C1-full: selected 0.545 against final 0.540, **+0.005** — nine-user
selection over 120 epochs bought almost nothing, which retires the +0.02 caveat for figures
whose metric did not choose the epoch, as the file's own rule said it would.

The honest train-user fit never carries (A2 − A1 ≤ 0 everywhere), and the reason is **not
the rank argument** — that one is A2-full's, and holds in 6/6 checkpoints — but the
test-fitted advantage: A2′ fits on the 17 people it is scored on and reaches +0.148; A2 fits
on 32 *other* people and reaches −0.008, so more correspondences did not help. The recommendation
to the field that follows: **the correspondences available for fitting an alignment are
bounded by the number of people recorded in two or more applications — 32 here, and Schach
had the same 32 — and no amount of pretraining raises it.** The honest answer to their
section 8 is not "we failed to make it work" but "this corpus cannot support it; a corpus
that could would need far more multi-application participants."

**What C2-lo changes.** (1) The best cross-application number on their split is **0.368 single
window / 0.693 at ten minutes, head only**, against their 0.180 / 0.308 — on their own
protocol (train on 0-22, validate 23-31, test 32-48) with 4,096 pretraining identities of Beat
Saber and Alyx added. It is a placement against their published mean, as before, and it is
double. (2) Identity count is flat across a domain boundary *without* exposure (Z-676 ≈
zero-shot, three measurements in this file agree) and **is not flat with it**: the same
23 exposed people are worth +0.089 on a 676-identity base and +0.137 on a 4,096-identity
base, with the whole C2-hi − C2-lo interval below zero. Pretraining scale and exposure
interact. (3) The alignment closure is narrowed, not reversed: the rotation Schach measured
exists on our embedding at their scale of exposure plus ours of identities, and the honest
route still fails for the rank reason the registration named. One seed; seeds 2-3 of C2-lo
are queued.

---

## The matched pair, seed 1 — the decisive run (2026-09-10 23:30)

## The matched pair, seed 1 — the decisive run (2026-09-10 23:30)

Rows `1faeea6e5e70` (Z-676) and `4a1ba1eb4442` (C2-hi), both at `517cdaa57b`, both loaded
exactly as the pair lists specify (495 training identities; the identical 181 validation
people; 17 evaluation users; C2-hi additionally 147,921 training windows of which 20,896 are
Across-XR 0-22, dose 14.1%, and Across-XR 23-31 dropped). Gates PASS at 2.2e-5 and 4.5e-5.
Cross-application verification AUC on the 17: Z-676 0.556, **C2-hi 0.683**.

| arm, seed 1 | A0 within | **A1 cross** | CI95 | 10-min cross | A2′ − A1 | A2 − A1 | A2-null − A1 | A2-full − A2 | m-curve |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z-676 (600 BOXRR + alyx, no Across-XR) | 0.477 | **0.218** | [0.168, 0.271] | 0.356 | +0.011 [−0.021, +0.042] | −0.005 | −0.080 | −0.051 | peaked, unaligned is best (range 0.054) |
| C2-hi (the same minus 23 BOXRR train users, plus Across-XR 0-22) | 0.581 | **0.307** | [0.263, 0.354] | **0.604** | **+0.002 [−0.012, +0.016]** | −0.003 | **−0.184** | −0.042 | flat (range 0.042) |

| registered contrast (paired on the same 17 users) | measured | registered | verdict |
| --- | --- | --- | --- |
| **C2-hi − Z-676** | **+0.089 [+0.048, +0.131]** | +0.05 to +0.20; falsifier < +0.03 | **inside the band at the mean; the interval's lower edge sits on the band's edge; one seed.** Seeing the five applications on 23 people, at a 14% window dose and exactly fixed identity count, is worth +0.09 cross-application on unseen people. Further seeds registered as the priority pair |
| Z-676 − A1 (zero-shot 4096) | −0.013 [−0.039, +0.013] | within ±0.03 | **holds** — identity-count flatness across a domain boundary re-measured on this corpus at 676 against 4,096 |
| C1 − A1 (zero-shot 4096) | −0.100 [−0.147, −0.055] | not registered | carries C1's budget term (Amendment 3); C1-full pending |
| A2′ − A1 on C2-hi | **+0.002 [−0.012, +0.016]** | the mechanism hypothesis: does exposure at scale create the rotation? | **No.** With the applications seen at scale there is nothing orthogonal to fit — while the permuted null costs 0.184, so the person-specific structure is strong and already aligned across applications |
| A2-full − A2 on both | −0.051 / −0.042 | ≤ 0 | holds again — five checkpoints out of five |

**What the pair establishes.** (1) The alignment route is closed for this head-only model
**with a mechanism, on every instrument**: zero-shot (+0.026, 3 seeds), exposure on 23
identities alone (+0.011), exposure at scale on 495 identities (+0.002). Schach's +0.34
test-fitted gain is a property of their model — architecture, sensor set, or both — not of
the task or of training exposure; the orthogonal component is real and person-specific in
every arm (the null hurts everywhere) and never exceeds +0.03. Their section 8's future work
does not transfer to a head-only `dyn` embedding, and that is reported as the finding it is.
(2) **Domain exposure at fixed identity count is worth +0.09 cross-application and +0.25 on the
10-minute sequence (0.604 against Z-676's 0.356, and against Schach's 0.308)** — for
identifying people never seen, across applications *within a set the model was trained on*.
That generalises across **people**, this project's central question, and it is the first
composition lever measured to do so on this instrument. **It is not a lever crossing an
activity boundary**: C2-hi's applications are seen (its people are not), where the Nymeria
activity-diversity arm tested transfer to corpora the treatment never touched. The cell that
would earn the bigger claim is P3 — leave-one-application-out on unseen users — registered
in Amendment 4 and run next. (3) Identity count is flat again (Z-676 ≈ zero-shot 4096).


## Zero-shot arm, three seeds — the registered verdicts (2026-09-10 21:30)

Rows `655dd23af5ed` / `f2f947553746` / `981aa86f4bd4` (seeds 1-3), all at `517cdaa57b`, 77
min each on the 4060 Ti; cross-application verification AUC on the 17: 0.5688 / 0.5807 /
0.5880 (position lookup 0.585 / 0.592 / 0.598, amplitude 0.50); selected epochs 120 / 116 /
118 of 120. Gates PASS at 2.0e-5 / 2.9e-4 / 7.0e-5. Aggregation:
`across_xr_alignment_aggregate.py` → `across_xr_alignment_aggregate.json` (per-user
accuracies averaged over seeds inside each user, then bootstrapped over the 17 users, so a
seed never counts as an extra person).

| arm | rank-1 @17, 3 seeds | CI95 (users) | per seed | 10-min majority |
| --- | --- | --- | --- | --- |
| A0 within-application | 0.500 | [0.462, 0.538] | 0.499 / 0.505 / 0.497 | 0.947 |
| **A1 cross-application, no alignment** | **0.234** | **[0.181, 0.292]** | 0.231 / 0.230 / 0.240 | 0.357 |
| A2 alignment fitted on users 0-31 (m=4 in every seed) | 0.245 | [0.216, 0.276] | 0.247 / 0.241 / 0.246 | 0.482 |
| A2′ alignment fitted on the 17 test users (ceiling) | 0.260 | [0.222, 0.304] | 0.255 / 0.259 / 0.265 | 0.517 |
| A2-null permuted correspondence | 0.160 | [0.143, 0.179] | 0.151 / 0.161 / 0.167 | 0.271 |
| A2-full unrestricted 128-d fit | 0.189 | [0.165, 0.215] | 0.186 / 0.194 / 0.188 | 0.319 |

Schach et al., same 17 users, head + both controllers, 15 s: within 0.831, cross **0.180**,
10-min cross 0.308, test-fitted alignment 0.523.

| registered contrast | 3 seeds, paired, user bootstrap | per seed (range) | registered | verdict |
| --- | --- | --- | --- | --- |
| A1 | 0.234 [0.181, 0.292] | 0.231 / 0.230 / 0.240 (0.010) | P1: 0.18-0.35, falsifier < 0.12 | **inside the band.** Every seed above the published 0.180 and the interval's lower edge sits on it; by the registered power rule (≈0.10 absolute, or paired across seeds *and* an interval clear of the target) this is **"at or above the published controller-based figure, head-only, zero-shot" — not a resolved beat** |
| A2′ − A1 | **+0.026 [+0.000, +0.051]** | +0.024 / +0.029 / +0.025 (0.005) | ≥ +0.15; kill condition: upper bound < +0.05 | **band excluded decisively** — the whole interval is below +0.15 by 3× at its upper end and 13× below Schach's +0.34. The orthogonal component exists (below) and is an order of magnitude smaller than theirs. The registration's unnamed region [+0.05, +0.15) is recorded in Amendment 2 and does not bear on this |
| A2 − A1 | **+0.011 [−0.020, +0.041]** | +0.016 / +0.011 / +0.006 (0.010) | +0.05 to +0.20; falsifier < +0.05 | **band excluded** — train-only fitting does not carry, and there was almost nothing to carry |
| A2 − A2′ | −0.015 [−0.034, +0.000] | −0.008 / −0.018 / −0.019 | leakage if above A2′ beyond the CI | no leakage; the honest fit sits just under the ceiling in every seed |
| A2-null − A1 | −0.074 [−0.126, −0.029] | −0.080 / −0.069 / −0.073 | ≤ +0.03 | holds; the permuted fit costs 0.07 in every seed — the fit is person-specific, it is simply small |
| A2-full − A2 | −0.055 [−0.086, −0.029] | −0.061 / −0.047 / −0.057 | ≤ 0 in expectation | holds in every seed — the rank argument (32 correspondences, 128-d, arbitrary 96-d complement) appears in the data |
| seed spread of A2 − A1 | range 0.010 | | means within 0.05 | holds; the headline is the mean, not the spread |
| m-curve (validation, N=9) | flat in 3/3 seeds (ranges 0.045 / 0.050 / 0.048), m* = 4 in all three | | report flat or peaked | **FLAT**; the choice of m is immaterial |
| P3 direction (A1) | unseen-activity cells below seen-activity cells in every seed (seed 1: 0.196 vs 0.246) | | unseen < seen | holds |

**What the zero-shot arm establishes.** (1) Head-only `dyn`, trained on 4,096 identities of
Beat Saber and Alyx and never on this corpus, identifies Schach's 17 test users across
applications at 0.234 on a single 10 s window; every seed sits above their reported mean of
0.180 (controllers, 15 s, trained on those people's other applications), with three of four
confounds running against us (the fourth, identity count, this project has measured flat
across a domain boundary). **This is a placement against a published figure, not a test:**
their 0.180 is a mean with an across-cell sd of 15.1 and an unpublished distribution, our
interval is over users, and a significance claim against a published mean whose
distribution was not published is not available at any margin — the interval's excluding
0.180 by 0.001 is treated exactly as a 0.003 near-miss the other way was treated. (2) On
this embedding the cross-application gap is barely an orthogonal difference: the
test-fitted ceiling that gave them +0.34 gives +0.026 here, 13× smaller, and the honest
train-user fit is inside it; the component is real (the permuted null costs 0.074 in every
seed), person-specific and small. Alignment is not the paper's contribution on the
zero-shot instrument, and "there is no rotation" would be wrong. (3) A0 is confounded
(sensor set *and* domain exposure) and is not a result on its own; see C1 below.

## Matched arm C1 — Schach's protocol on our model (seed 1, 2026-09-10 22:00)

Row `984f4a622b4f`: Across-XR users 0-22 alone in training (23 identities, all five
applications, cross-application positives), validation 23-31, evaluation 32-48; patience
fired at epoch 15 of 30; cross-application verification AUC 0.528. Gate PASS at 1.0e-6.

| C1, seed 1 | rank-1 @17 | CI95 | 10-min | registered | verdict |
| --- | --- | --- | --- | --- | --- |
| A0 within-application | 0.252 | [0.200, 0.307] | 0.453 | — | against their 0.831 under the same protocol and exposure |
| A1 cross-application | **0.131** | [0.088, 0.177] | 0.194 | 0.10-0.25, falsifier < 0.089 | inside the band; the lower edge sits on the falsifier |
| A2′ − A1 | +0.011 | [−0.005, +0.027] | | | no orthogonal structure from exposure alone at 23 identities |
| A2 − A1 | −0.002 | [−0.017, +0.013] | | | nothing to carry |
| A2-null − A1 | −0.051 | [−0.083, −0.020] | | | the small component is still person-specific |
| m-curve | flat (range 0.021) | | | | |

**C1 is under-trained, read from its own checkpoint history (Coordinator's ask).** At the
selected epoch 15 the nine-user validation accuracy was still rising monotonically (0.514 →
0.523 over epochs 10-15) and training loss was still falling steeply (14.57 → 12.49);
patience then fired because the nine-user signal never exceeded
0.523 in epochs 16-30. Nine users is the smallest selection set in the programme, and
"patience fired" on nine people reads as noise, not convergence. **So 0.131 is depressed by
an unknown amount**, and every comparison involving C1 below carries a budget term until
C1-full (Amendment 3: the same arm at the zero-shot arm's 120-epoch cap, patience 0) lands.

What C1 separates, and what it does not. It matches Schach on protocol, split and exposure
and differs in **sensor set and model family together**, and it trains on **23 identities —
below the 48 at which this project measured the behavioural arm at chance**. So 0.252
within-application is roughly what our own prior says 23 identities buy, and **the
surprising number is their 0.831 at the same 23 identities, 3.3× ours**: the candidates are
the sensor set, their architecture's sample efficiency, or both, and isolating the sensor
set needs their architecture head-only, which the public code does not contain. "Head-only
costs 0.58 within-application" is not a claim this experiment can support and is not made.

On the same cells the zero-shot model reads 0.500 within-application and **0.234 against
0.131 cross-application** — identity count from other corpora over exposure on 23 people,
in the direction everything this project has measured about identity count points.
**Stated with its budget term:** the two arms ran different budgets (zero-shot censored at
120/120 and still improving; C1 stopped at 15 on nine users, under-trained), so the +0.10 is
not registered and is a lower bound only in the sense that the censoring runs against it —
the C1 side is the weak half. C1-full is the number to quote once it exists. Exposure alone
does not create the orthogonal structure Schach measured (+0.011); C2-hi against Z-676 tests
exposure on top of 653 pretraining identities and is the decisive run of the programme.

---

## Seed 1 (superseded by the table above; kept as the first record)

Registration: `across_xr_alignment_REGISTERED.md` (with Amendment 1). Harness:
`across_xr_alignment.py`. Per-seed artefacts: `across_xr_alignment_seed<N>.json` and
`across_xr_alignment_seed<N>_gate.json`. Every figure is rank-1 at **N = 17** on Schach's
test users 32-48 (chance 0.0588), single 10 s probe window, gallery = all windows of the
enrolment application, `dyn` encoding, head only, cluster-bootstrap CI over the 17 users.

**Instrument, seed 1.** Row `655dd23af5ed`, code identity `517cdaa57b`, trained on this
node in 77 minutes (RTX 4060 Ti, numpy 2.5.3, torch 2.14.0+cu130, CUDA 13.0, capability
8.9): BOXRR-23 (all) + who_is_alyx, `dyn` 10 s / stride 5, 3072 training identities, 1024
validation, evaluation = exactly the 17 (15,592 windows). Cross-application verification
AUC on those 17 **0.5688** (`position_lookup_auc` 0.585 — the height cue; amplitude 0.496);
selected epoch 120 of 120, right-censored like every 10 s run in 9.14. `git_sha` on the row
reads `2096df9-dirty` because the corpus symlink was untracked at write time; the code was
committed (`2096df9`) and the identity is the clean tree's.

**Gate: PASS.** Rescored on CPU through the pipeline's own loader: 0.568812 against the
recorded 0.568793, gap 2.0e-5 (tolerance 2e-3 across devices), 17 users.

## Seed 1

| arm | rank-1 @17 | CI95 | 10-min majority | unseen-activity cells | seen-activity cells |
| --- | --- | --- | --- | --- | --- |
| A0 within-application | 0.499 | [0.462, 0.537] | 0.941 | — | — |
| **A1 cross-application, no alignment** | **0.231** | **[0.175, 0.293]** | 0.339 | 0.196 | 0.246 |
| A2 alignment fitted on users 0-31 (m=4) | 0.247 | [0.207, 0.288] | 0.493 | 0.202 | 0.266 |
| A2′ alignment fitted on the 17 test users (ceiling) | 0.255 | [0.212, 0.303] | 0.496 | 0.213 | 0.273 |
| A2-null permuted correspondence | 0.151 | [0.129, 0.176] | 0.277 | | |
| A2-full unrestricted 128-d fit | 0.186 | [0.157, 0.218] | 0.319 | | |

Schach et al., same users, same N, head + both controllers, 15 s window: within 0.831,
cross **0.180**, 10-min cross 0.308, test-fitted alignment 0.523.

| registered contrast | measured (paired, user bootstrap) | registered | verdict |
| --- | --- | --- | --- |
| A1 | 0.231 [0.175, 0.293] | P1: 0.18-0.35, falsifier < 0.12 | **inside the band**; at or above the published 0.180, not a resolved beat (interval includes 0.18; one seed; power rule needs ~0.10 or several seeds) |
| A2′ − A1 | **+0.024 [−0.006, +0.053]** | ≥ +0.15; programme falsifier: upper bound < +0.05 | **the band is excluded decisively**: the entire interval sits below +0.15 (2.8× at its upper end) and 14× below Schach's test-fitted +0.34. Whether the upper bound is 0.047 or 0.053 changes nothing — a registered band is settled by where the interval falls. The registration left [+0.05, +0.15) unnamed and the interval's upper end landed in it (Amendment 2 of the registration records that defect). The orthogonal component *exists* — the permuted null hurts — and is an order of magnitude smaller than theirs |
| A2 − A1 | **+0.016 [−0.011, +0.041]** | +0.05 to +0.20; falsifier < +0.05 | **band entirely excluded** — train-only fitting does not carry, and there was almost nothing to carry |
| A2 − A2′ | −0.008 [−0.035, +0.013] | leakage if A2 above A2′ beyond the CI | no leakage |
| A2-null − A1 | −0.080 [−0.135, −0.030] | ≤ +0.03 | holds; the permuted fit *hurts*, so the fit is person-specific |
| A2-full − A2 | −0.061 [−0.102, −0.025] | ≤ 0 in expectation | holds; the rank argument shows in the data |
| m-curve (validation, N=9) | unaligned 0.359; m=4 0.381, 8 0.377, 16 0.356, 24 0.342, 32 0.335 | report flat or peaked | **FLAT** (range 0.045 < 0.05); m*=4 is immaterial |
| P3 direction | unseen-activity cells 0.196 < seen-activity 0.246 (A1) | unseen below seen | holds on this instrument |

**Per ordered cell, A1 (A2):** synth_riders→beat_saber 0.459 (0.439), beat_saber→synth_riders
0.406 (0.423), half_life_alyx→superhot_vr 0.272 (0.325), beat_saber→superhot_vr 0.261
(0.241), superhot_vr→beat_saber 0.253 (0.269), synth_riders→superhot_vr 0.238 (0.207),
superhot_vr→half_life_alyx 0.235 (0.279), half_life_alyx→beat_saber 0.229 (0.281),
social_vr→beat_saber 0.224 (0.215), superhot_vr→synth_riders 0.216 (0.240),
synth_riders→social_vr 0.207 (0.203), half_life_alyx→social_vr 0.206 (0.241),
social_vr→synth_riders 0.196 (0.188), synth_riders→half_life_alyx 0.189 (0.172),
half_life_alyx→synth_riders 0.189 (0.215), beat_saber→half_life_alyx 0.182 (0.208),
social_vr→half_life_alyx 0.171 (0.220), beat_saber→social_vr 0.169 (0.194),
social_vr→superhot_vr 0.167 (0.194), superhot_vr→social_vr 0.151 (0.178). The two rhythm
games transfer to each other at twice the mean in both directions.

**Per user, A1 (the distribution PAPER_PLAN asks for beside the mean):** 0.40 0.12 0.33 0.24
0.15 0.14 0.41 0.16 0.23 0.41 0.48 0.17 0.14 0.08 0.18 0.17 0.12 — five of seventeen above
0.33 and one below 0.10; "the model identifies at 0.23" and "a user has a 0.23 chance of
being identified" are different claims and only the first is supported.

## Reading, provisional on seeds 2-3

Alignment is not the paper's contribution on the zero-shot instrument: the test-fitted
ceiling that gave Schach +0.34 gives +0.024 here, and the honest fit is inside that. The
orthogonal component exists (the permuted null costs 0.08) and is an order of magnitude
smaller than theirs. **Hypothesis, testable on the matched arms:** their structure was
measured on a model trained on all five applications for the same 23 people, and this
model never saw the applications — if the rotation appears on C1 / C2-hi it is a property
of training exposure rather than of the task, which is a finding either way. A2 is re-run
on those embeddings as registered.

What survives regardless is A1: head only, never trained on the corpus, 10 s against their
15 s — three disadvantages against one (4096 training identities against 23, which this
project has measured flat across a domain boundary) — at or above the controller-based
published cross-application figure on their own split. **A0 is confounded and is not yet
a result:** 0.499 against 0.831 differs in sensor set *and* in domain exposure (zero-shot
against a model trained on those people's other applications); C1 and C2-hi separate the
two. "The scope cost sits within-app" is a hypothesis until they land.

## The alignment section, as it stands after nine checkpoints (2026-09-11 04:00)

Three sentences that survive every seed, in the order the paper should give them:

1. **The honest train-user fit never carries.** A2 − A1 ≤ 0 on all nine checkpoints
   (zero-shot ×3, C1, C1-full, Z-676, C2-hi, C2-lo ×2), whatever the arm, the scale, the
   exposure or the budget.
2. **The correspondences available for fitting are capped by the corpus at 32
   multi-application participants**, and no amount of pretraining raises that; Schach had
   the same 32. The recommendation to the field is a corpus specification, not a method.
3. **The test-fitted ceiling that motivates the idea is itself run-dependent at identical
   configuration** — +0.148 and −0.004 on two seeds of C2-lo — so **a single-run test-fitted
   diagnostic bound is not evidence that application embeddings differ by a rotation.** This
   is the form of evidence their +0.34 is; we do not claim their number is wrong, we show the
   quantity has run-to-run variance a single measurement cannot see, which raises the bar for
   every claim of this shape, including ours.

The permuted null costing 0.21-0.29 on every exposed arm stays as the evidence that the
structure is person-specific and strong; what varies between runs is only whether any of it
is *orthogonal between applications*. Two sentences withdrawn on this evidence, both the
Coordinator's and recorded as such: "the C2-hi dip is thirty sigma" (seed variance was
imported from a different arm — 0.003 on zero-shot against 0.152 on C2-lo, thirty times
larger; the dip is 0.8 σ at this arm's spread) and "the aligned ceiling rises monotonically
with scale" (one seed per point). Seed variance is a property of an arm, not of a pipeline;
when the choice is between an elegant account of n = 1 and a second run, take the run.
Seed 3 of C2-lo, when it lands, is reported as "present in k of three runs", never as a
rate; three seeds cannot estimate one. The half arm's A2′ column is dropped as uninformative.

## P3 — leave-one-application-out on unseen users, five runs (2026-09-11 06:30)

Five checkpoints, one per held-out application, on C2-hi's exact lists (495 trained /
181 validation / 17 evaluated; 23-31 dropped) with the held-out application's sessions absent
from every Across-XR user (`CrossApplicationXR_LOAO_<X>`, four sessions per user; dose
≈11.6%); all at `517cdaa57b`; gates PASS at 2.8e-5 / 1.9e-5 / 2.5e-5 / 1.5e-5 / 5.2e-6; scored
on the full corpus with each checkpoint's own statistics under its copy's name. Unit: the
eight ordered cells involving X, paired on the 17 users against Z-676 (no exposure) and C2-hi
(full exposure), pooled over X (`across_xr_alignment_p3.py`, `_p3.json`, `_p3_split.py`).

| held out X | rows | P3 on X-cells | Z-676 | C2-hi | **P3 − Z-676** | P3 − C2-hi | non-X control | A2′ − A1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Superhot VR | `c2e7eadf2f9f` | 0.223 | 0.189 | 0.288 | +0.034 [−0.005, +0.070] | −0.065 | +0.003 | +0.003 |
| Half-Life: Alyx | `7b957695ac6e` | 0.231 | 0.205 | 0.292 | +0.026 [−0.004, +0.058] | −0.061 | −0.029 | +0.039 |
| Beat Saber | `ab1affebece8` | 0.351 | 0.267 | 0.352 | **+0.084 [+0.043, +0.122]** | −0.001 | −0.013 | +0.135 |
| Synth Riders | `9292a5747e7d` | 0.324 | 0.247 | 0.337 | **+0.077 [+0.038, +0.115]** | −0.013 | −0.001 | +0.136 |
| Social VR | `6a0675b0b8f9` | 0.224 | 0.180 | 0.265 | +0.044 [−0.000, +0.084] | −0.041 | −0.004 | +0.131 |
| **pooled, five** | | 0.271 | | | **+0.053 [+0.022, +0.083]** | −0.036 [−0.054, −0.018] | −0.009 [−0.025, +0.008] | |
| uncovered triple (Superhot, Synth, Social) | | | | | **+0.052 [+0.019, +0.083]** | −0.040 [−0.059, −0.020] | | |
| covered pair (Beat Saber, Alyx) | | | | | +0.055 [+0.024, +0.089] | −0.031 [−0.055, −0.007] | | |

| registered (Amendment 4) | measured | verdict |
| --- | --- | --- |
| P3 − Z-676 pooled in +0.02..+0.07; headline needs lower bound > +0.03; falsifier ≤ 0; (0, +0.02] unresolved | +0.053 [+0.022, +0.083]; uncovered triple +0.052 [+0.019, +0.083] | **inside the band at the mean; the falsifier is excluded; the headline is NOT made** (lower bound 0.019-0.022 against 0.030). Exposure to four applications carries about +0.05 of the +0.09 in-set gain to an unseen fifth on unseen people; whether the boundary claim holds sits on the interval's lower edge, and the registration's own remedy applies: a second seed on one application |
| P3 − C2-hi < 0 (direction, dose-confounded) | −0.036 [−0.054, −0.018] | holds; two sufficient explanations (the held-out application, and 20% less Across-XR data) |
| control P3(non-X) − C2-hi(non-X) within ±0.03 | −0.009 [−0.025, +0.008] | holds: 20% less in-domain data at fixed exposure cost nothing measurable on the seen cells, so the runs are comparable to C2-hi |
| rhythm games carry best | Beat Saber +0.084, Synth Riders +0.077 — full carry (P3 ≈ C2-hi on their cells) | **holds**: Synth Riders is an activity no pretraining corpus covers, and with Beat Saber in training it is reached as if seen |
| Social VR carries least, below +0.03 | +0.044 [−0.000, +0.084]; Alyx is least at +0.026 | **fails** — Social VR carries about as much as the mean; the task-structure ordering holds at the top and not at the bottom |
| covered vs uncovered | +0.055 vs +0.052 | the carry is the same whether or not pretraining covered the activity — it is not pretraining leaking through the hold-out |
| A2′ − A1 below +0.03 | +0.003 / +0.039 / +0.135 / +0.136 / +0.131 | run-dependent, as established on C2-lo: present on three of five, absent on two, at identical configuration; A2 − A1 ≤ 0 on all five (14 checkpoints now) |

**What P3 establishes.** Exposure to a set of applications on 23 people carries to an
application none of them played *in the training set*, for people never seen: +0.05
[+0.02, +0.08] against no exposure, against +0.09 for full exposure, with the two rhythm
games reached as if seen. It is registered as not the headline, because the interval's lower
edge sits 0.01 under the line on one seed per application; it is also the first measurement
in this project of a data-side lever moving anything across an activity boundary, and the
honest sentence is "partial carry, unresolved at the registered threshold, a second seed
away from either verdict".

**Stated at the right strength (Coordinator, 2026-09-11 07:00).** Supported: **exposure to
four applications carries to an unseen fifth** — the whole interval above zero, the falsifier
excluded, and the first data-side lever in this project measured to cross an activity
boundary (identity count was flat across one; activity diversity of other people fired its
falsifier). Not yet supported: "by at least 0.03", the stricter phrase the registration set
for the headline, which read +0.022 at the lower bound. The lead instance is **Synth Riders**:
a rhythm game in no pretraining corpus, reached essentially fully (+0.077, P3 ≈ C2-hi) from
exposure to the other four. The coverage control (uncovered +0.052 against covered +0.055) is
what answers the first reviewer objection. The prediction is scored in both halves: "rhythm
games carry best" held (Beat Saber and Synth Riders are the top two); "Social VR least"
failed (Alyx is least). Two re-seeds (Synth Riders, Social VR) run under the third addendum's
purpose and rules.

## C2-lo, three seeds — the headline is firm; the rotation is present in one of three (2026-09-11 08:00)

Seed 3: row `26fbf01d1ad9`, epoch 116/120, verification AUC 0.719, gate PASS at 1.9e-4.

| C2-lo | seed 1 | seed 2 | seed 3 | three seeds, pooled over users |
| --- | --- | --- | --- | --- |
| A1 cross-application @17 | 0.368 | 0.378 | 0.377 | **0.375 [0.321, 0.435]** (range 0.010) |
| 10-min majority vote | 0.693 | 0.709 | 0.731 | 0.711 |
| A2′ − A1 | +0.148 | −0.004 | +0.001 | **present in one of three runs** — reported as that, never as a rate |
| A2 − A1 | −0.008 | −0.002 | −0.000 | −0.003 [−0.011, +0.005]: never carries |
| A2-null − A1 | −0.286 | −0.210 | −0.209 | −0.235: person-specific structure on every seed |

**C2-lo − zero-shot(4096) = +0.141 [+0.100, +0.183] over three paired seeds.** The best
cross-application figure on Schach's own split — head only, their protocol plus 4,096
pretraining identities — is 0.375 on a single 10 s window and 0.711 at ten minutes,
against their 0.180 / 0.308 with controllers, as a placement against a published mean.
The alignment section's three sentences stand on twelve checkpoints of this programme
plus the five P3 runs.

## Amendment 5 — dose separated from scale by window count (2026-09-11 09:30)

Row `67e26f8e9022` at `517cdaa57b`: C2-lo's composition with every Across-XR session
truncated to its first half (`CrossApplicationXR_HALF`; Across-XR ≈10,400 of 529,594
training windows, dose ≈2.0% against C2-lo's 3.0%; the same 3,095 identities and the same
23 people); epoch 109/120; gate PASS at 2.2e-4 on its own half-session evaluation; scored
on the full corpus under the copy's name.

| | A0 | **A1** | CI95 | 10-min | A2′ − A1 | A2-null − A1 |
| --- | --- | --- | --- | --- | --- | --- |
| C2-lo-half, seed 1 | 0.639 | **0.347** | [0.290, 0.410] | 0.587 | −0.004 (uninformative by rule) | −0.145 |
| C2-lo, three seeds | 0.616 | 0.375 | [0.321, 0.435] | 0.711 | | −0.235 |

| registered | measured (paired on 17 users, half vs the mean of C2-lo's three seeds) | verdict |
| --- | --- | --- |
| within ±0.03 → dose not binding; below −0.03 → dose binds | **−0.028 [−0.062, +0.009]**; 10-min −0.124 | **unresolved between the two named outcomes**: the mean sits on the −0.03 edge and the interval spans it, on one seed. Halving the in-domain windows at fixed identities and people costs about 0.03 single-window at the mean and 0.12 at ten minutes. Read with the non-X dose control of P3 (−0.009 [−0.025, +0.008] for a 20% cut): dose has a modest, probably real cost in this range that is far smaller than the exposure effect (+0.14) and does not overturn the C2-hi / C2-lo reading — the 14% arm on 495 identities lost to the 3% arm on 3,095 by 0.061, which a dose cost of this size cannot account for |

The half arm's A2′ (−0.004) is not read, per the rule that a single-seed value of a
run-dependent quantity is uninformative.

**Dose figures reconciled from the loaders' own window counts (Coordinator, 2026-09-11
10:00).** The C2-lo dose was stated as 3.0% from a pre-run estimate against the whole 707k
corpus *including validation windows*. The loaders say: zero-shot training set 519,211
windows (BOXRR + alyx after the 25% draw); C2-lo 540,107 with the same draw plus Across-XR
0-22, so Across-XR contributes exactly **20,896 → 3.87%**; C2-lo-half 529,594, so **10,383 →
1.96%**, i.e. halved, consistent. C2-hi's 14.1% (20,896 of 147,921) was already the loader's
figure. **Every "3.0%" above reads 3.9%**; the C2-hi / C2-lo contrast is 14.1% against 3.9%.

**Why one seed is defensible for the half arm, and only there.** The arm reports an A1
difference; A1's seed range on C2-lo is 0.010 (0.368 / 0.378 / 0.377) against a user-bootstrap
interval width of 0.071, so seeds are a seventh of the uncertainty and one is enough. A2′'s
seed range on the same three runs is 0.152, which is why its one-seed values are not read.
Declining to seed is defensible only when the measurement that makes it so can be pointed at;
this is that measurement.

**The dose reading, stated at full strength.** Halving costs −0.028, a 20% cut −0.009 —
roughly linear and modest — and **C2-hi has the higher dose and still loses to C2-lo by
0.061**, so dose was working in C2-hi's favour: correcting for it *widens* the scale effect
rather than narrowing it.

**The pair's composition is verified by the numbers, not by the procedure that produced
them:** C2-lo 540,107 − Across-XR 20,896 = 519,211 = the zero-shot arm's training set,
exactly. The treatment arm is provably the control's training windows plus Across-XR 0-22
and nothing else. The 3.0% was a pre-run estimate against a 707k denominator that included
validation windows — a wrong denominator, the commonest way a dose figure goes wrong — and
the count was made even though the cause had been guessed correctly, because a correct
guess is exactly the moment checking stops. P2 of PAPER_PLAN (`raw` minus `dyn`) was not
run by this programme and is recorded as not tested, not dropped; it does not open here.
