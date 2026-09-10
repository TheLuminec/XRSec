# Across-XR: train-user-only orthogonal embedding alignment — REGISTERED BEFORE RUNNING

**Written 2026-09-10 on the Miami server (host `feng-MS-7B51`, session b47df677), before any
checkpoint had been scored on Across-XR anywhere in this project, and before the converted
corpus had finished arriving on this machine.** Registered per the project rule that a
prediction written after the number is an excuse, and per `docs/PAPER_PLAN.md`, whose
predictions P1-P3 this file inherits without change and extends.

## What is being measured, and against what

Schach, Rack, McMahan, Latoschik 2026 (arXiv:2509.08539), their test users **32-48** (the
`split` column of our converted corpus reproduces their 23/9/17 partition digit-exactly),
**N = 17, chance 1/17 = 0.0588**:

| their figure | value |
| --- | --- |
| within-application rank-1, single 15 s window | **0.831** (0.723-0.880) |
| cross-application rank-1, single 15 s window, 20 ordered off-diagonal pairs | **0.180** (0.105-0.226), sd 0.151 |
| cross-application, 10-minute sequence, majority vote | 0.308 (0.090-0.577) |
| cross-application, single window, **after orthogonal alignment fitted on the TEST users** | **0.523** — their section 6.2.5, disqualified by their section 9 as "a diagnostic upper bound, not a deployable, generalizing solution" |
| the same at 10 minutes | 0.943 |

Their model is head **plus both controllers**, BRV encoding, Transformer-into-GRU, 480-d,
trained on users 0-22 across all five applications. Ours is **head only**, `dyn` encoding,
`bilstm`, 128-d, `identity_softmax`, trained on **BOXRR-23 + who_is_alyx and never on
Across-XR**. Two of the five applications (Beat Saber, Half-Life: Alyx) are our training
*activities* on different people and different rigs; Superhot VR, Synth Riders and Social VR
are activities the model has never seen. So every cross-application cell here is
*unseen users*, and the cells not involving Beat Saber or Alyx are *unseen users of an unseen
activity* — the cell their design never tests (PAPER_PLAN). The two claims are kept
separate throughout.

## The instrument

A `dyn` 10 s / 20 Hz / stride 5 `bilstm` `identity_softmax` checkpoint trained on
BOXRR-23 (all users) + who_is_alyx (76), `epochs=120`, `early_stopping_patience=15`,
`val_user_fraction=0.25`, `normalize=per_dataset`, `eval_normalize=target_fit`,
`within_dataset_negatives=true`, `cross_session_positives=true`, margin 0.35 / scale 30,
batch 1024, lr 0.001, `samples_per_user=512`, embedding 128 — the 9.14 configuration, read
field by field off the recorded row `661054c98a12`. **Three seeds** (1, 2, 3): the
bootstrap below carries the user-sampling uncertainty and not the seed variance, and the
project's own arithmetic puts the n=2→3 step at 3.6x on the paired multiplier.

**Trained on this node** (the DESKTOP-C 9.14 checkpoints are on DESKTOP-C only and that
session is offline; the Coordinator ruled to train here), with
`test_dirs = CrossApplicationXR_Dataset`, `exclude_users = users 32-48`,
`test_on_excluded=true` — so the training row's own `selected_test_auc` is cross-application
verification AUC on exactly the 17 test users, through the pipeline's loader, and is the
gate referent for everything below. The seven seated corpora are pulled from AVALON so the
same checkpoints can be rescored on them (`mode=rescore`) and set beside 9.14's per-corpus
transfer figures; that comparison is by value, not digit-exact, because the evaluation
population there differs from the recorded rows. Every training row quotes
`position_lookup_auc` (height survives across applications at P=0.754; lateral placement
does not, 0.527) and `amplitude_auc` beside the model.

## Protocol, fixed before the data is looked at

- **Windows**: 10 s at 20 Hz, stride 5 s, `encoding=dyn`, the checkpoint's own normaliser
  (target-fit on Across-XR as an unseen corpus, exactly as `evaluate()` does it). Each
  (user, application) cell is one continuous recording of ~14-17 minutes, so ~170-200
  windows per cell.
- **Embedding**: the extractor's output, L2-normalised; similarity is cosine, which is
  the checkpoint's own head.
- **Gallery template** for user *u* in application *A*: the mean of the normalised
  embeddings of all of *u*'s enrolment windows in *A*, renormalised. **Probe**: a single
  window. rank-1 = the fraction of probes whose own user's template is the nearest of the
  17. Ties are rank-averaged, as everywhere in this repo.
- **Cross-application cell (A→B)**: enrolment = all windows of *A*; probes = all windows
  of *B*. 20 ordered off-diagonal cells; the headline is their unweighted mean, quoted
  beside the per-cell table.
- **Within-application cell (A→A)**: enrolment = the first half of the recording by start
  time, probes = the second half. This is the same-session regime; it is also theirs (one
  recording per cell), so the comparison is like for like, and it is why `dyn` is mandatory
  — within-application lateral placement reads P=0.7525 on this corpus
  (`across_xr_within_application.json`) and a `raw` within-vs-across contrast would credit
  the within side with a rig cue.
- **Evidence mismatch, stated rather than hidden**: their probe is one 15 s window; ours is
  one 10 s window. Their enrolment is "reference embeddings" from the application; ours is
  every window of it. Neither is adjusted to match the other.
- **Per-user accuracy** is computed for every cell (each user's own probes), and every
  headline carries a **cluster bootstrap over the 17 users** (10,000 resamples of the
  per-user accuracy vector, gallery held fixed) for its CI and for every paired difference.
  Bands below are read against the interval, not against p < 0.05.

## The alignment

For an ordered pair (A→B): let C_A and C_B be the n_fit × d matrices of per-user
centroids (mean of normalised embeddings per user, per application) over the **fitting
users**. Orthogonal Procrustes: R = U Vᵀ from the SVD of C_Bᵀ C_A, the orthogonal matrix
(rotation or reflection — Schach found both) minimising ‖C_B R − C_A‖_F. It is applied
**unchanged** to the test users' probe embeddings from B before scoring them against the
test users' gallery templates from A. Nothing about the test users enters the fit in the
result arm; the fit set is the design variable that distinguishes the arms.

**The unrestricted fit is ill-posed here, and the restriction is part of A2's definition,
not a follow-up (Coordinator, 2026-09-10).** With n_fit = 32 correspondences in d = 128 the
cross-covariance C_Bᵀ C_A has rank at most 32, so 96 of its singular values are zero and the
corresponding columns of U and V are an arbitrary orthonormal completion: R is determined
by the data on the span of the fitting centroids and is an **arbitrary isometry on the
96-dimensional complement**, which it then applies to the test users' embeddings. Two runs
could differ for reasons unconnected to the data. So the alignment is fitted on a
**subspace**: the top-m principal components of the fitting users' window embeddings
(both applications pooled), m ≤ 32; Procrustes is solved on the projected centroids
(m × m), and R is the identity on the complement. The unrestricted fit is reported as a
*variant* beside it, never as the headline. The fixture test pins the rank argument: two
different orthonormal completions of the unrestricted fit must give different scores for
points off the fitting span, and identical scores on it.

**m is chosen on the validation split, never on the test users.** For each m ∈ {4, 8, 16,
24, 32}: fit on train users 0-22, score cross-application rank-1 on validation users
23-31 (N = 9, quoted with that N), take the m with the highest 20-cell mean; then refit
with that m on all of 0-31 and apply once to 32-48. The full m-curve on the validation
split is recorded on the certificate so the choice is auditable. The same m serves the
test-fitted ceiling A2′, so the two differ only in whose centroids the fit saw.

## Arms — all on the same embeddings, therefore paired by construction

| arm | fit set | what it is |
| --- | --- | --- |
| **A0** | — | within-application, no alignment |
| **A1** | — | cross-application, no alignment — PAPER_PLAN's P1 |
| **A2′** | test users 32-48, subspace m as selected above | Schach's own illegitimate route, on our embedding: the **diagnostic ceiling** |
| **A2** | users 0-31 (32 people, all five applications), subspace m selected on 23-31 | **the result** |
| **A2-null** | users 0-31 with the user correspondence **permuted** (B-centroid rows shuffled), same m | the guard, in the direction that matters |
| **A2-full** | users 0-31, unrestricted 128-d Procrustes | the ill-posed variant, reported beside A2 and never as the headline |
| raw counterpart of A0/A1 | — | P2: `raw` minus `dyn` on the same cells, the anthropometric share |

## Predictions, registered

- **A0**: Beat Saber and Alyx (our training activities) above Superhot, Synth Riders and
  Social VR. No band on the level — head-only against 0.831 with controllers is a scope
  cost we report, not predict.
- **A1**: inherits P1 — **0.18 to 0.35, falsifier below 0.12**. Below 0.12 says head-only
  `dyn` cannot approach controller-based cross-application identification and that is
  reported as the cost of our scope. Unseen-activity cells (pairs among Superhot / Synth /
  Social) below seen-activity cells — P3's direction, measured on our instrument.
- **A2′ − A1 ≥ +0.15** on the 20-cell mean. **FALSIFIER FOR THE WHOLE PROGRAMME: A2′ − A1
  < +0.05** (bootstrap CI upper bound below +0.05): our embedding's cross-application gap is
  not an orthogonal difference, alignment cannot be the paper's contribution on this
  instrument, and that is reported as the finding. Schach measured +0.34 on theirs.
- **A2 − A1 in +0.05 to +0.20**, with **A1 < A2 < A2′**. **FALSIFIERS**: A2 − A1 < +0.05
  (train-only fitting does not carry — an honest negative, reported as prominently as a
  positive); **A2 exceeding A2′ beyond the bootstrap CI on the paired per-user difference
  is evidence of LEAKAGE, not success** — A2′ dominates in expectation, not on every cell
  and not within noise, so a small excess is a draw and only an excess the interval
  excludes raises the alarm (the certificate says so and no number from that run is quoted
  until the leak is found); **A2 > 0.523** is flagged under PAPER_PLAN's rule even though
  the ceiling that binds *our* embedding is A2′.
- **A2-null ≤ A1 + 0.03.** A permuted-correspondence fit that helps would mean the
  "alignment" is doing something other than aligning people — a guard verified in the
  direction whose failure mode is to pass.
- **A2-full ≤ A2** in expectation (the arbitrary complement can only add noise); if
  A2-full reads above A2 beyond the interval, the subspace choice is costing signal and
  the m-curve on the validation split is re-read before anything else is concluded.
- **Seed variance**: the three seeds' 20-cell means for A2 − A1 within 0.05 of each other;
  wider than that and the headline is the seed spread, not the mean.
- **Secondary, 10-minute sequences**: majority vote over the probe windows of a 600 s span
  of B. At ~15 minutes per cell that is one sequence per (user, cell): 17 decisions per cell,
  coarse by construction, reported with that caveat beside 0.308 / 0.943.

## Power, computed before anything runs

At N=17 over 17 test users the effective sample is users, not windows: binomial sd on a
rank-1 near 0.18 is sqrt(0.18·0.82/17) = **0.093** per cell. The 20 off-diagonal cells share
the same 17 users and are not 20 independent samples. Every difference above is therefore a
*paired* difference on the same embeddings, bootstrapped over users, over two seeds. The
+0.05 falsifiers sit at about half a single-cell sd; they are resolvable only because they
are paired, and the certificate reports the achieved interval rather than asserting they
were.

## Gates, both required before any figure is quoted

1. **Checkpoint gate.** The harness re-scores each checkpoint on its own recorded
   evaluation users through the pipeline's own loader and `evaluate()`, and must reproduce
   the recorded `selected_test_auc` within 1e-4 on the device that wrote the row (1e-3
   across devices, per CLAUDE.md's measured cuDNN band). Certificate:
   `docs/acceptance/across_xr_alignment_gate.json`.
2. **Fixture gate.** The alignment and scoring code is run first on synthetic embeddings
   with a known answer: 49 users in 128-d, application B = application A rotated by a random
   orthogonal Q plus noise. On that fixture A1 must sit near chance, A2′ and A2 near 1.0,
   A2-null near chance; and on a fixture where B = A + noise (no rotation) A2 must not fall
   below A1 by more than the noise. The tests assert on the fixture's content, not only on
   the result (`assert` that Q is orthogonal and that the fixture's unaligned rank-1 really
   is near chance) — a test whose subject failed to load reports the subject's success.

## Deviations to record on the certificate rather than here

Which checkpoint (relayed or trained here), device, seeds actually run, and the exact
window count per cell. If the instrument changes from what "The instrument" specifies, that
is an amendment appended below this line with a date, and the original text stays.

## The matched arm — approved by the Coordinator 2026-09-10 as a SEPARATELY LABELLED arm

The converter's docstring and PROVENANCE say never to pool Across-XR into training; that
rule protects the zero-shot claim from the corpus being *silently* absorbed, and the
Coordinator is amending both to record the rule and this exception. The ruling:

| | |
| --- | --- |
| **zero-shot arm** (everything above) | Across-XR never in training. The strong claim. Unchanged. |
| **matched arm** | training may use Across-XR users **0-22 only**, epoch selection on **23-31 only**. The like-for-like comparison to Schach, who trained on exactly those 23 users. |
| **absolute** | users **32-48** are never trained on, never validated on, never used to fit an alignment, never used to choose an epoch or an m. |
| **never pooled** | the two arms are never averaged, and no figure is quoted without naming its arm. |

Two matched configurations, registered now so their predictions precede their rows:

- **C1 — their protocol, our model**: Across-XR 0-22 alone in training (23 identities, all
  five applications, `cross_session_positives=true` so positives are cross-application by
  construction), validation on 23-31, 10 s `dyn`. Prediction: **cross-application rank-1
  at N=17 in 0.10 to 0.25** — 23 identities is where this project measured the behavioural
  signal at chance (48 identities, pooled corpus), so a low figure is expected and is the
  identity-count story, not a failure of the encoding. Falsifier: below chance + 0.03.
- **C2 — pretrained plus matched**: BOXRR + alyx + Across-XR 0-22 in training, validation
  on 23-31 (explicit) for the Across-XR share and the usual 25% draw for the rest.
  Prediction: **C2 − A1 (zero-shot) in +0.05 to +0.20** on the same 20 cells; 23 people
  seen in all five applications is the only training signal in this project that spans an
  activity boundary within one person. Falsifier: C2 − A1 < +0.03, which would say that
  seeing the applications does not help even with the users held out — the same shape as
  the Nymeria activity-diversity null, and worth knowing.
- **Alignment on the matched arms**: A2 is re-run on C2's embeddings (fit on 0-31, apply
  to 32-48). Prediction: the alignment gain shrinks as the model has already seen the
  applications, A2(C2) − C2 < A2 − A1.

Explicit validation users require a small change to the training path (a
`validation_users` list beside `val_user_fraction`); it is a `model/*.py` edit on this
node's own checkout, committed before any matched row is written, and its identity is
recorded on those rows.

---

# AMENDMENT 1 — 2026-09-10, before any matched row: C2's dose is 3.0%, so C2 becomes a pair

**Amended for a fact about the instrument, known before any number exists.** The original
text above is left intact. Measured by the Coordinator on the real files: Across-XR users
0-22 hold 9,963,704 rows = 30.4 h = **~21,900 windows** at 10 s / stride 5, against 707,017
BOXRR+alyx windows at 4096 identities - **3.0% of C2's training windows**, the same dose at
which the Nymeria activity-diversity null was uninterpretable (2.9%). `identity_softmax`
samples windows uniformly, so 23 identities of 3,095 is not the quantity that matters. A
null on C2 as registered cannot distinguish "seeing the applications does not carry" from
"the objective barely saw them", and only the first licenses the conclusion. **A dose is
part of a treatment's definition; a null without one is a result about the dose.**

The fix is composition, not sampling: `balance_identities=cap` would trim Across-XR (≈950
windows per identity against BOXRR's ≈150) and lower the dose further - the identical
wrong-direction fix the Nymeria arm found. And cutting BOXRR is measured to cost nothing on
the axis C2 reports: transfer is flat in identity count across a domain boundary (419 →
2096 moved pooled transfer by 0.001, 2096 → 4096 by 0.000).

**C2 is therefore two arms, and each has its own zero-shot control at the same identity
count, so the only variable inside a pair is whether Across-XR 0-22 was trained on:**

| arm | training | Across-XR dose | control |
| --- | --- | --- | --- |
| **C2-lo** | BOXRR (all 4020) + alyx + Across-XR 0-22 | **3.0%** | the zero-shot arm above (4096 ids) |
| **C2-hi** | Z-676's users and Z-676's validation list, minus the **last 23 BOXRR training users** of the seeded permutation, plus Across-XR 0-22 (train); Across-XR 23-31 **dropped** (`drop_users`: neither trained on, validated on, nor evaluated) | **14.1%** measured (20,896 of 147,921 training windows, seed 1) | **Z-676**: BOXRR seeded subsample of 600 + alyx 76, no Across-XR, validation = the pipeline's own 25% draw made explicit and shared with C2-hi |

**Exact by construction, not by argument (Coordinator's swap, made exact 2026-09-10).** A
cap of 577 against 600 equalises identities only before the validation draw: the 25% draw
runs over each arm's own pool, so the arms would validate on different people and train on
513 against 507 identities, with BOXRR training sets not nested. So both arms take the same
explicit validation list (`val_user_fraction=0`) and C2-hi drops 23 BOXRR *training* users.
Verified on the lists the loaders hold (`docs/acceptance/c2_pair_lists.py`, seed 1,
`c2_pair_users_seed1.json`): Z-676 train **495** / val **181** / test 17; C2-hi train **495** /
val **181** (the identical people) / test 17; BOXRR training users 435 against 412, a strict
subset; alyx training users identical; C2-hi's Across-XR training ids exactly 0-22, and
23-31 nowhere in it. The only difference inside the pair is which 23 identities did which
activity - the Nymeria arm-B design. The lists are regenerated per seed and committed.

**Why 23-31 are dropped rather than validated on (Coordinator, 2026-09-10).** With 23-31 in
C2-hi's validation the arms would validate on 181 against 190, and C2-hi would choose its
epoch with nine target-corpus users in the signal - under 5%, but pointing at the arm we
hope wins, the shape this project has already paid for once. Under `test_on_excluded=true`
the exclude list *is* the evaluation set, so removing them needed a third list:
`drop_users`, added as a second numerics-free identity step (`73ecbf9232 → 517cdaa57b`; see the
certificate) before the first row, so every row still carries one identity. C1 keeps
23-31 as validation: it is Schach's protocol and is not improved into something else.

Registered:

- **C2-hi − Z-676 in +0.05 to +0.20** on the 20 cross-application cells at N=17 (the band
  the original C2 carried, now attached to the arm that can test it). Falsifier: **< +0.03**,
  which at a 14% dose does say that seeing the five applications on 23 people does not
  carry to unseen people.
- **C2-lo − A1**: registered as a dose statement, not a treatment test. Above +0.05 is
  informative (a 3% dose already carries); a null is a result about 3% and is reported as
  exactly that.
- **Z-676 − A1 within ±0.03**: the identity-count flatness measured elsewhere in this file,
  re-measured here at 676 against 4096 on Across-XR. If it fails, the C2-hi comparison is
  read against Z-676 only and the pooled-vs-capped difference is reported separately.
- **C2-hi − C2-lo**: the dose effect itself; predicted positive. If C2-hi ≤ C2-lo the dose
  argument was wrong and both arms are read against their own controls without it.
- The alignment (A2) is re-run on C2-hi's embeddings; prediction unchanged from the
  original text (gain shrinks once the applications were seen).

Order on the card, after the three zero-shot seeds: C1 (seed 1), Z-676 (seed 1), C2-hi
(seed 1), C2-lo (seed 1); further seeds as the card allows, C2-hi and Z-676 first because
that is the pair that resolves the question. C1 is untouched by this amendment - 23 users
alone is a 100% dose and is Schach's protocol exactly.

The counts in `unseen_users_guard_both_directions.md` (3095 / 1033) are consistent with
`validation_users` being honoured but one user away from it being ignored; the membership
lists (0-22 / 23-31 / 32-48) are what that certificate rests on.
