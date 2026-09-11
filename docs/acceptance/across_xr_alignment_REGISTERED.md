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

---

# AMENDMENT 2 — 2026-09-10, after seed 1: a registration defect, recorded, not re-registered

The A2′ − A1 registration named two outcomes — a band at ≥ +0.15 and a programme falsifier
at "CI upper bound < +0.05" — and left **[+0.05, +0.15) unnamed**. Seed 1's interval,
[−0.006, +0.053], put its upper end in the unnamed region. That is the same defect the
within-application placement registration had twelve hours earlier (two outcomes named, the
data choosing a third), reviewed by the Coordinator on the same day without the subtraction
that would have caught it: a rule can be in CLAUDE.md, read by both parties, and still not
fire at review.

**The verdict does not depend on the gap.** A registered band is settled by where the
interval falls: the whole interval sits below +0.15 by 2.8× at its upper end and 14× below
Schach's test-fitted +0.34, so the band is excluded decisively, and whether the upper bound
is 0.047 or 0.053 changes nothing. Nothing in the bands above is changed by this amendment;
it exists so the next reader sees the defect beside the number rather than a paragraph
arguing a near-miss.

---

# AMENDMENT 3 — 2026-09-10 22:30, after C1 seed 1: C1-full, a budget-matched C1

C1 seed 1 (row `984f4a622b4f`) selected epoch 15 on nine validation users with the
validation curve still rising, training loss still falling steeply and training accuracy at
1.3%, then stopped on patience at epoch 30. The zero-shot arm it is compared with ran the
full 120-epoch cap and was still improving there. The two arms therefore differ in budget as
well as in identity count and exposure, and this file's own rule says patience on an
uncharacterised axis cuts short whichever arm converges slower.

**C1-full**: identical to C1 (Across-XR 0-22 alone in training, validation 23-31, evaluation
32-48, 10 s `dyn`) with `early_stopping_patience=0` and the same 120-epoch cap as the
zero-shot arm. Registered before it runs: **C1-full − C1 in [0, +0.08]** on A1 (a
budget-only difference; if it exceeds +0.08 the nine-user selection was doing real harm and
every nine-user-selected figure in the programme carries that note); A1 of C1-full still
below the zero-shot 0.234 (falsifier: at or above it, which would mean exposure on 23 people
matches 4,096 identities of other activities once trained out); the alignment verdict on C1
unchanged (A2′ − A1 < +0.05). C1's own figures stay on the certificate as the
patience-selected run they are.

**Addendum to Amendment 3, before C1-full runs (Coordinator, 2026-09-10 22:45).** Three
corrections. (i) The band `[0, +0.08]` left **C1-full − C1 < 0 unnamed** — the third unnamed
region today; it is a live outcome (23 identities for 120 epochs with no patience can
overfit) and is registered as *"the 23-identity arm is capacity-limited rather than
budget-limited, and the zero-shot comparison stands as measured"*. (ii) C1-full does not
remove what broke C1 — the nine-user selection signal — it keeps it over a four-fold longer
budget, so part of any gain is selection inflation (this file prices a max over ~20
evaluations at about +0.02). The pipeline writes only the validation-selected checkpoint, so
the final-epoch weights will not exist and rank-1 at the final epoch cannot be scored
without a code change that would move the identity mid-programme; the diagnostic is taken
from the row as it is already recorded — `selected_test_acc` against `final_test_acc` on
the verification metric — and stated as a verification-metric figure, not a rank-1 one.
(iii) The "training accuracy 1.3%" line is struck as evidence: AM-softmax subtracts the
margin from the true class before the argmax, so training accuracy sits below chance early in
every `identity_softmax` run; the under-training diagnosis rests on the validation curve
still rising and the training loss still falling steeply, which are sufficient.

**Narrowing of (ii), same evening (Coordinator).** The epoch is chosen on validation
*verification* accuracy and A1 is rank-1 identification; CLAUDE.md's 2026-09-06 measurement
puts selection inflation on a metric that did not choose the epoch at +0.004 (the wrong sign
for optimism) and says not to carry the +0.02 onto such a figure. So the
`selected_test_acc` / `final_test_acc` diagnostic measures the inflation where it exists -
the verification columns - and is not read as contaminating A1; the residual on rank-1 is
bounded by how tightly the two metrics track, measured once at 4096 identities on 94 users,
which is a different regime from nine validation users at 23 and is why the free diagnostic
is still recorded.

---

# AMENDMENT 4 — 2026-09-10 23:50, before any run: P3, leave-one-application-out on unseen users

C2-hi measured exposure to the *target* application set: its people are unseen, its
applications are seen. PAPER_PLAN's P3 is the cell that crosses an activity boundary — train
on four applications, test on the fifth, unseen users — and Schach never ran it. It goes
ahead of the pair's seeds 2-3 (Coordinator's priority: seeds firm up a result already held,
P3 decides whether there is one more).

**Design.** For each held-out application X: C2-hi's exact composition and lists (seed 1:
BOXRR 600 minus the same 23 training users, alyx, the identical 181 validation people,
Across-XR 0-22 in training, 23-31 dropped, 32-48 evaluated) with **X's sessions absent from
every Across-XR user**, via a symlinked copy `CrossApplicationXR_LOAO_<X>`
(`build_loao_corpus.py`; 49 users × 4 sessions, every link resolving into the verified
corpus). Five runs, one per X. The training row's own evaluation (the gate referent) is
users 32-48 of the copy. Scoring is on the full corpus with the checkpoint's own statistics
applied under the copy's name (`--normalizer-dataset`).

**The unit is the cells involving X** — the four ordered cells with X as gallery and the four
with X as probe, on the 17 test users — paired against the same cells of Z-676 (no exposure)
and C2-hi (full exposure) on the same users, then pooled over the five X. The other twelve
cells of each P3 run are a within-run control: the applications the model *did* see.

**Registered.**
- **P3(X-cells) − Z-676(X-cells), pooled over X: +0.02 to +0.07** — exposure to four
  applications carries *part* of the +0.089 to a fifth. **The headline "exposure crosses an
  activity boundary" requires the interval's lower bound above +0.03**; below that the claim
  is not made whatever the mean.
- **Falsifier: P3(X-cells) − Z-676(X-cells) ≤ 0** — exposure to other applications does not
  carry to a new one, and the +0.089 is strictly in-set. Named outcome *below* the band:
  (0, +0.02] — a carry too small to distinguish from the tail of the in-set gain; reported as
  unresolved rather than as either.
- **P3(X-cells) − C2-hi(X-cells) < 0**: the held-out application costs against full
  exposure; if it does not (interval includes 0 or above), exposure to four is as good as
  five and the "seen application" distinction was not doing the work.
- **Control: P3(non-X cells) − C2-hi(non-X cells) within ±0.03** — removing one application
  from training leaves the seen-application cells where they were. If it fails, the five
  runs are not comparable to C2-hi and are read against Z-676 only.
- Per held-out application the same contrasts are reported individually; the two rhythm
  games (Beat Saber ↔ Synth Riders transfer at twice the mean) are the pair most likely to
  carry, and the Social VR scenario the least; that ordering is a prediction, not a band.
- Alignment on P3 embeddings: A2′ − A1 below +0.03, as on every instrument so far.

Power: 8 cells × 17 users per run, five runs pooled, user bootstrap. The bands are
narrower than the single-arm ones because the comparison is paired on cells *and* users.

**Addendum to Amendment 4, before any P3 run (Coordinator, 2026-09-11 00:05).**
(i) **Removing an application changes the dose**: C2-hi 20,896 of 147,921 windows (14.1%,
five applications); P3 ≈ 16,717 of 143,742 (≈11.6%, four) — 20% less Across-XR data. So
**P3 − Z-676 is clean** (Z-676 has 0%; the dose *is* the treatment and 11.6% is what the
band is about), **P3 − C2-hi is dose-confounded** (exposure to X *and* 20% less in-domain
data; a negative there has two sufficient explanations and it stays a direction, not a
measurement), and the non-X control **P3(non-X) − C2-hi(non-X)** is a free measurement of
what 20% less in-domain data is worth at fixed exposure — expected to lean slightly negative
for that reason; a fired control is read as a dose reading, and only a large one degrades the
runs to Z-676-only comparison. The actual dose per run is recorded from the loader.
(ii) **No seed replication**: five runs, one per held-out application, pooled over
applications — the unit is the application, and seed variance is *imported* from the
zero-shot arm's observed seed agreement (~0.01), which is an assumption written down as one.
If the pooled interval lands near the +0.03 headline threshold, a second seed on one X is
the cheapest way to stop it resting on an import.
(iii) **The per-application ordering is the mechanism claim** and is protected: rhythm games
(Beat Saber, Synth Riders — stationary, task-structured) carrying best and Social VR (no
predefined task) least would say exposure transfers along task structure rather than along
corpus identity. The per-X breakdown is reported whatever the pooled interval does.

---

# AMENDMENT 5 — 2026-09-11 01:30, before any run: dose separated from scale by window count

C2-hi − C2-lo (−0.061 [−0.099, −0.026]) falsified the dose direction, but with the treatment
corpus fixed at 23 identities `dose = axr_windows / (axr_windows + base_windows)`, so raising
the base mechanically lowers the dose: the two arms differ on one variable read two ways, and
a 1,500-identity point would land on the same confounded line (withdrawn). What separates
them is to **vary the Across-XR window count at fixed identity count**: at C2-lo's 3,095
identities, the same 23 people and the same lists, with **the first half of every Across-XR
session only** (a real-file copy `CrossApplicationXR_HALF`, each session CSV truncated to its
first half by row; users 32-48's sessions truncated too, so the training row's own
evaluation is on half-sessions and is the gate referent; scoring is on the full corpus with
the checkpoint's statistics under the copy's name). Dose ≈ 1.5% against 3.0%; scale and
people fixed.

Registered: **C2-lo-half − C2-lo (A1, same users) within ±0.03** — dose is not the binding
variable in this range, and the falsification of the dose direction is explained rather than
merely observed. Named outcomes: below −0.03, dose does bind (and the C2-hi/C2-lo pair is
read as scale *minus* a dose cost); above +0.03, less in-domain data helps, which would be
read as a regularisation effect and flagged for a seed before anything is concluded.
A2′ − A1 expected to stay near C2-lo's +0.148 (scale and exposure unchanged).
