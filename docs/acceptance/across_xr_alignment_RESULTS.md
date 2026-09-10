# Across-XR alignment — results certificate (living; seed 1 of 3 as of 2026-09-10 19:00)

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
