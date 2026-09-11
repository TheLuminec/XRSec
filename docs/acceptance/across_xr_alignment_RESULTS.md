# Across-XR alignment — results certificate (zero-shot arm complete, 3 seeds; matched arms pending)

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
0.523 over epochs 10-15), training loss was still falling steeply (14.57 → 12.49) and
training accuracy was 1.3%; patience then fired because the nine-user signal never exceeded
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
