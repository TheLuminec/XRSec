# Paper outline: generalisation of head-motion identification across XR applications

Drafted 2026-09-15. **Outline only, not prose.** Every number carries its source tag; the tags are
defined once below. Where two sources disagree the discrepancy is flagged inline with **[DISCREPANCY]**
and collected in "Open questions and gaps" at the end. Nothing marked **PENDING** has a number, and
none should be invented for it.

## Source tags

| tag | file |
| --- | --- |
| [RES] | `git show origin/worktree-across-xr-alignment:docs/acceptance/across_xr_alignment_RESULTS.md` (**unmerged branch**) |
| [REG] | same branch, `docs/acceptance/across_xr_alignment_REGISTERED.md` (registration + Amendments 1-7) |
| [AGG] | same branch, `docs/acceptance/across_xr_alignment_aggregate.json` (zero-shot, C1, Z676, C2hi, C2lo; per-user and per-cell fields) |
| [P3J] / [P3S] / [P3SPLIT] | same branch, `across_xr_alignment_p3.json` / `_p3_stability.json` / `_p3_split.json` |
| [P2J] / [DOSE] / [MARGIN] | same branch, `across_xr_alignment_p2.json` / `_dose.json` / `_margin.json` |
| [PLAN] | `docs/PAPER_PLAN.md` (OUTCOMES section; **partly stale**, see gaps) |
| [CL] | `CLAUDE.md` (Across-XR section through P2; "Head-only is the scope"; margin/scale "TESTED AT SCALE") |
| [LIT] | `docs/LITERATURE_BRIEFING.md` |
| [SCH] | `external_sota/schach2026.pdf` = **arXiv v1** (10 Sep 2025, 10 pages) |
| [SCH-FV] | Schach et al., Frontiers in VR 2026, doi:10.3389/frvir.2026.1743491 - **not in the repo**; known only via [LIT]/[PLAN] |
| [SCH-REL] | `external_sota/schach2026/training-and-evaluation/evaluation/files/slm_model_data/accuracy_values.json` (their released per-user values) |

---

## 1. Title options

1. *Behaviour Travels With Exposure: Head-Only Motion Identification Across XR Applications*
2. *Who Are You in the Next Application? Head-Motion Identification Across XR Applications, With a Static-Cue Audit*
3. *Head Motion Alone Identifies XR Users Across Applications - and Anthropometry Makes It Easier Than a Behaviour-Only Assessment Admits*

(Option 1 foregrounds the exposure finding; option 3 foregrounds the privacy reading. Choose after
the PENDING formal comparison lands, since it may change how strongly the headline can be worded.)

## 2. Abstract draft (one paragraph, numbers to be re-checked against the final comparison)

Motion-based identification in XR is near-solved within an application and collapses across
applications: the reference study reports 18.0% rank-1 cross-application among 17 unseen users
using head and both controllers [SCH]. We study the same instrument - the Across-XR corpus, 49 users
in five applications, the authors' own 23/9/17 user split - with **head motion only**, a scope chosen
so the method applies to AR glasses that have no hand tracking. A head-only model trained on
Beat Saber and Half-Life: Alyx recordings from other people (BOXRR-23, who-is-alyx) and never on
Across-XR reaches 0.234 cross-application rank-1 at N=17 [RES]; adding the reference study's own
23 training users raises it to 0.375 on a single 10 s window and 0.711 over ten minutes [RES].
Leave-one-application-out training shows exposure to four applications carries to an unseen fifth
(+0.049, CI [+0.021, +0.078]) [P3S], including an application absent from all pretraining data.
Train-user-only orthogonal embedding alignment, proposed by the reference study as future work,
does not carry, and we identify why the corpus cannot support it [RES]. Finally, auditing our own
result, a model that keeps absolute head pose reaches 0.351 after a single training epoch [P2J]:
static anthropometry and posture are available on the surface of the input, so behaviour-only risk
assessments understate cross-application identification risk. [PENDING: sentence on the formal
paired per-user comparison against the released per-user values.]

## 3. Contributions (each tied to its evidence)

1. **A head-only, zero-shot placement against the controller-based cross-application figure.**
   0.234 [0.181, 0.292], 3 seeds, N=17, never trained on Across-XR, vs published 0.180 [RES, AGG, SCH].
   Framed as a placement pending the formal comparison (section 8, P-1).
2. **Recommended headline: pretraining plus the reference study's training users.** C2-lo 0.375
   [0.321, 0.435], 3 seeds, N=17; ten-minute 0.711 vs 0.308 [RES, AGG, SCH]. Head-only vs head + both
   controllers. Must be reported beside C1-full (our model on their 23 training users alone: 0.164)
   so readers see the gain needs the pretraining data [RES].
3. **The unseen-application cell the reference study never tested**, and a carry across it: +0.049
   [+0.021, +0.078] pooled over five held-out applications; coverage control uncovered +0.046
   [+0.017, +0.074] vs covered +0.055; Synth Riders, in no pretraining corpus, carries (+0.077 /
   +0.053 on two seeds). The stricter registered threshold (lower bound > +0.030) is **not met**
   [RES, P3S, P3SPLIT, REG Amendment 4].
4. **Identity count interacts with exposure; dose does not explain it.** Z-676 - zero-shot -0.013
   [-0.039, +0.013]; C2-hi - C2-lo -0.061 [-0.099, -0.026]; halving in-domain windows -0.028
   [-0.062, +0.009]; 20% cut -0.009 [-0.025, +0.008] [RES, DOSE].
5. **A negative answer, with mechanism, to the reference study's train-user-only alignment
   proposal**: the honest fit never resolvably carries; fitting correspondences are capped at 32
   multi-application participants by the corpus; the test-fitted ceiling itself is run-dependent at
   identical configuration (C2-lo: +0.148 / -0.004 / +0.001) [RES, AGG].
6. **A static-cue audit of our own headline, in the main results.** raw - dyn +0.117 [+0.042, +0.192]
   zero-shot, every raw seed selected at epoch 1 of 16; with exposure the static advantage is
   unresolved (+0.029 [-0.068, +0.134], 1 seed) and over ten minutes the exposed behavioural model
   leads (0.711 vs 0.497) [RES, P2J].
7. **A hyperparameter reversal with a registered mechanism**: margin/scale 0.1/15 gave +0.016 at 419
   identities and costs -0.028 [-0.045, -0.012] at the 4,096-identity scale [CL, RES, MARGIN].
8. **Methodological**: preregistered bands with amendments kept in place; every checkpoint gated
   against its own recorded row (23 gates PASS, gaps 5.3e-8 to 2.9e-4) [RES]; per-user distributions
   reported beside means [PLAN metric section, AGG per_user fields].

---

## 4. Section-by-section outline

### 4.1 Introduction
- Motion data is shared by necessity in XR; identification is both an authentication tool and a
  privacy risk [SCH section 1].
- Within-application identification is strong; cross-application collapses: 83.1% within vs 18.0%
  cross, single window, N=17 [SCH 6.2.2]. State "chance 5.88% at N=17" beside it.
- Two gaps in the reference design [PLAN]: (i) its "cross-application" cell is unseen users of
  applications seen in training, never an unseen application; (ii) its BRV encoding removes head
  position by construction, so it assesses behavioural risk only.
- Scope statement (not a limitation): head-only so the method runs on AR glasses without hand
  tracking [CL "Head-only is the scope"]. Consequence stated up front: every comparison to the
  reference is head-only vs head + both controllers.
- Contributions list (section 3). Preregistration statement.
- **Figure 1** (teaser): cross-application rank-1 at N=17 for zero-shot, C2-lo, Schach 0.180, chance
  0.0588, single window and ten minutes. Data: [AGG], [SCH].

### 4.2 Related work
- Motion identification in XR: Rogers et al. 2015; Miller et al. 2020; Rack et al. 2023 "Who is Alyx"
  (71-user dataset; 20 s windows at 15 Hz; seen-user cross-session 76.6-78.3%) [LIT section 3, 7];
  Nair et al. 2023 "Unique Identification of 50,000+ VR Users" [SCH ref 17] - **DUA-mandated
  citation** (see 4.3).
- Pretrainable similarity learning: Rack et al. 2024 "Versatile User Identification" (arXiv:2302.07517)
  [SCH ref 23]; Miller et al. 2021 cross-system Siamese [SCH ref 13].
- Encodings: Rack et al. 2022 SR/BR/BRV/BRA, BRA best in their setting [LIT section 2]; note our
  measured inversion (raw >> bra > brv > br) on head-only pooled corpora [CL "Input encodings"] -
  one sentence, to motivate why `dyn` rather than BRV.
- Cross-application: Baldoni et al. 2025 (~30% across two applications, classification) [SCH 2.1];
  Schach et al. 2026 - the reference [SCH].
- Embedding alignment: Schach et al. section 6.2.5 test-fitted orthogonal alignment 18.0% -> 52.3%,
  30.8% -> 94.3%, disqualified as a diagnostic upper bound, train-user-only fitting named as future
  work [LIT section 5, PLAN] - **cite the Frontiers version [SCH-FV]; not present in arXiv v1
  [DISCREPANCY, gap G1]**. Their released code contains `orthogonal_procrustes`-based multi-domain
  alignment (`evaluation/helpers/compute_transformation_matrix.py`), corroborating the method.
- Datasets table: cite BOXRR-23 dataset paper (Nair et al., doi:10.1109/TVCG.2024.3372087) and
  who-is-alyx (Zenodo 10.5281/zenodo.8379914) [SCH refs, docs/DATASET_CATALOGUE.md C7].

### 4.3 Datasets
- **Across-XR** (evaluation instrument and, in matched arms, training users 0-22):
  - 49 users x 5 applications (Synth Riders, Superhot VR, Beat Saber, Half-Life: Alyx, Social VR),
    10-15 min each, standing, HP Reverb G2, one sitting in fixed order [SCH 4.1-4.2]; CC BY-NC-SA 4.0
    [PLAN, DATASET_CATALOGUE X1]; 90.9 Hz native, positions in cm, y-up [CL].
  - Split 0-22 / 23-31 / 32-48 read from the corpus's `split` column, digit-exact [PLAN].
  - Structure facts to report: all 245 (participant, application) cells present; each cell is one
    unbroken recording (`take_id` redundant with `game_id`) [CL]; our `takeN`/`game_id` order is not
    play order [PLAN "Corpus fact"]; cross-application pairs are separated by up to ~1 h in one sitting,
    never by days [PLAN].
  - Static-cue geometry of the corpus (motivates `dyn` and the audit): across applications lateral
    placement P(within<between) 0.527 (chance-level), height 0.754; within an application lateral
    0.7525 [CL].
  - **Table 1** (dataset summary). Data: [SCH], [CL], [PLAN].
- **BOXRR-23** (pretraining): 4,020 users converted, head (HMD) track only, Beat Saber in our copy
  and in the release's labelled index [CL]. **Citation obligation (DUA clause 5)**: any public
  disclosure must cite Nair et al. 2023 - per our own `CITATION.txt` text in `prepare_boxrr.py`
  this is *"Unique Identification of 50,000+ Virtual Reality Users from Head & Hand Motion Data"*,
  arXiv:2302.08927 [gap G9: confirm against the DUA whether the dataset paper must also be cited;
  cite both]. State the DUA and ethics approval in the data statement [CL].
- **who-is-alyx** (pretraining): 76 players, 146 sessions, mostly two sessions on different days [CL];
  cite Rack et al. 2023.
- Pretraining coverage: Beat Saber and Half-Life: Alyx are pretraining *activities* (different
  people, rigs); Superhot VR, Synth Riders, Social VR are not [REG "What is being measured"].

### 4.4 Method
- Input: head pose only (quaternion + HMD position), 10 s windows at 20 Hz, stride 5 s [REG instrument].
- **`dyn` encoding**: pose relative to the window's mean pose, gravity kept; removes height, seat,
  placement; invariant to rigid transforms of the capture frame [CL "Cross-corpus evaluation"].
  Contrast with BRV (HMD rotation + controller pose, differentiated; head position discarded) [SCH 5.2].
- Backbone `bilstm`, 128-d embedding; objective `identity_softmax` (AM-Softmax, margin 0.35, scale 30),
  cosine scoring [REG, CL "Training objectives"].
- Training: per-dataset normalisation, `target_fit` statistics for an unseen corpus, within-dataset
  negatives, cross-session positives, 120 epochs / patience 15, 25% validation-user draw,
  validation-selected epoch [REG instrument]. C1-full: 120 epochs, patience 0 [RES].
- **Arms** (Table 2 of the paper; data [RES], [REG Amendments 1, 3-7]):

  | arm | training | Across-XR in training | trained identities | dose |
  | --- | --- | --- | --- | --- |
  | zero-shot (3 seeds) | BOXRR 4,020 + alyx 76 | none | 3,072 (1,024 val) | 0 |
  | C1 / C1-full (1 seed each) | Across-XR 0-22 only | yes | 23 | 100% |
  | Z-676 (1 seed) | BOXRR 600 + alyx | none | 495 | 0 |
  | C2-hi (1 seed) | Z-676 minus 23 BOXRR train users, plus Across-XR 0-22 | yes | 495 | 14.1% |
  | **C2-lo (3 seeds)** | BOXRR 4,020 + alyx + Across-XR 0-22 | yes | 3,095 | 3.87% |
  | C2-lo-half (1 seed) | as C2-lo, first half of each Across-XR session | yes | 3,095 | 1.96% |
  | P3 x5 (1 seed; Synth Riders and Social VR 2 seeds) | C2-hi lists, one application removed | 4 of 5 apps | 495 | ~11.6% |
  | raw zero-shot (3) / raw C2-lo (1) | as zero-shot / C2-lo, `encoding=raw` | as base | as base | as base |
  | M-zero (1 seed) | zero-shot at margin 0.1 / scale 15 | none | 3,072 | 0 |

  Composition closure to state: 540,107 - 20,896 = 519,211 windows [RES, PLAN].
  **[DISCREPANCY, gap G5]** identity naming: [RES]/[PLAN] call zero-shot and C2-lo "4,096 identities"
  and Z-676/C2-hi "676"; [CL] calls them 3,072/3,095 and 495. The larger figures include validation
  users. The paper must use trained identities and say so.
- **Alignment variants** (on fixed embeddings): A2' (Procrustes fitted on test users 32-48, a
  diagnostic ceiling), A2 (fitted on users 0-31, the honest arm), A2-null (permuted correspondence),
  A2-full (unrestricted 128-d fit, ill-posed: rank <= 32 in 128-d); subspace dimension m chosen on
  validation users 23-31 (N=9) [REG "The alignment"].

### 4.5 Evaluation protocol
- **Primary metric**: rank-1 at N=17 on test users 32-48, chance 0.0588; single 10 s probe vs a
  per-user gallery template (renormalised mean of all enrolment windows of application A); 20 ordered
  off-diagonal cells, unweighted mean [REG protocol].
- **Within-application**: first half of the recording as gallery, second half as probe [REG].
- **Ten-minute sequence**: majority vote over the first 600 s of probe windows; one decision per user
  per cell (17 per cell), coarse by construction [REG, across_xr_alignment.py `score_cell`].
- **Uncertainty**: cluster bootstrap over the 17 users (10,000 resamples), seeds averaged inside each
  user before resampling, all contrasts paired on the same users [REG, RES]. Bands read against
  intervals, not p-values.
- **Power statement** to include: single-cell binomial sd ~0.093 at rank-1 0.18, N=17 [PLAN]; the 20
  cells share users and are not independent [PLAN, REG]. Same-composition contrasts are about twice
  as well powered as different-composition ones (half-width 0.017 vs ~0.037) [RES close-out].
- **Metric mismatch vs the reference, stated explicitly**: matched in N and in users; *not* matched
  in decision rule or window. Theirs: nearest reference *window* embedding (kNN with majority vote),
  15 s windows at 30 fps, BRV; ours: mean-embedding template, 10 s at 20 Hz, `dyn` [SCH 5.4; REG
  "Evidence mismatch"; coordinator note]. The PENDING single-harness comparison (P-1, P-2) resolves this.
- **Gates** [REG "Gates"]: (1) checkpoint gate - rescored through the pipeline loader must reproduce
  the recorded verification AUC (23/23 PASS, gaps 5.3e-8 to 2.9e-4) [RES]; (2) fixture gate for
  alignment code on synthetic rotated embeddings. Verification AUC appears only as a gate referent,
  never beside a rank-1.
- **Preregistration**: P1-P3 in [PLAN] (dated 2026-09-10), arm-level bands and Amendments 1-7 in
  [REG]; amendments kept in place, not edited [REG].
- **Table 3** (protocol comparison, theirs vs ours: sensors, encoding, window, rate, architecture,
  embedding size, decision rule, training data). Data: [SCH 5.2-5.4, Table 2], [REG].

### 4.6 Results

**R1. Zero-shot, head-only (P1).**
- A1 cross 0.234 [0.181, 0.292], seeds 0.231 / 0.230 / 0.240; A0 within 0.500 [0.462, 0.538];
  10-min cross 0.357, within 0.947 [RES, AGG].
- P1 band 0.18-0.35, falsifier < 0.12: **held** [PLAN].
- Wording rule: "at or above the published controller-based mean; every seed above 0.180; not a
  beat" - lower CI edge 0.181 vs 0.180 [RES]. Do not argue the 0.001 [CL].
- Unseen-activity cells below seen-activity cells on every seed (seed 1: 0.196 vs 0.246) [RES].
- A0 is confounded (sensor set and exposure) and is not a result on its own [RES].
- **Table 4** (zero-shot A0/A1/10-min, per seed, with Schach 0.831/0.180/0.308). Data: [AGG], [SCH].
- **Figure 2** (20-cell heatmap of A1, ours next to theirs). Data: [AGG] `per_cell`, [SCH-REL] or [SCH Fig. 4].

**R2. Exposure on the reference study's training users (recommended headline).**
- C2-lo A1 0.375 [0.321, 0.435], seeds 0.368 / 0.378 / 0.377; 10-min 0.711 (0.693 / 0.709 / 0.731);
  A0 0.616 [RES, AGG, P2J].
- C2-lo - zero-shot +0.141 [+0.100, +0.183], 3 paired seeds [RES].
- Registered: C2 - A1 in +0.05..+0.20 [REG matched arm]; measured inside.
- **Recommendation, stated as one**: lead with C2-lo; zero-shot second. Justification: same Across-XR
  split and training users as the reference, and the largest margin (0.375 vs 0.180; 0.711 vs 0.308).
  Obligatory qualifiers in the same paragraph: (a) it adds 4,096-user-pool pretraining the reference
  did not use; (b) head-only vs head + controllers; (c) formal comparison PENDING (P-1).
- The control that keeps the headline honest: **C1 (their data only, our model) 0.131 [0.088, 0.177]
  at epoch 15, under-trained; C1-full 0.164 [0.128, 0.205]** - below 0.180 [RES]. The gain is
  pretraining x exposure, not our model on their data.
- C2-hi - Z-676 +0.089 [+0.048, +0.131], 1 seed; 10-min 0.604 vs 0.356 [RES].
- **Table 5** (all arms: A0, A1 with CI, 10-min, seeds). Data: [AGG], [RES].
- **Figure 3** (A1 by arm, with CIs, Schach 0.180 line and chance line). Data: [AGG], [RES].

**R3. Unseen application (P3).**
- Pooled over five held-out applications, P3 - Z-676 on the held-out application's 8 cells: +0.049
  [+0.021, +0.078] seed-averaged [P3S, RES claim 3]. Falsifier (<= 0) excluded; registered band
  +0.02..+0.07 held at the mean; **headline threshold (lower bound > +0.030) not met** [REG Amendment 4].
  **[DISCREPANCY, gap G2]** [PLAN] OUTCOMES quotes the seed-1-only values +0.053 [+0.022, +0.083],
  uncovered +0.052, which [RES] later superseded.
- Coverage control: uncovered triple +0.046 [+0.017, +0.074] vs covered pair +0.055 [+0.024, +0.089]
  [P3S, P3SPLIT].
- Per application (seed 1): Superhot +0.034 [-0.005, +0.070]; Alyx +0.026 [-0.004, +0.058]; Beat
  Saber +0.084 [+0.043, +0.122]; Synth Riders +0.077 [+0.038, +0.115] (seed 2 +0.053); Social VR
  +0.044 [-0.000, +0.084] (seed 2 +0.032) [RES, P3J, P3S].
- P3 - C2-hi -0.036 [-0.054, -0.018] (dose-confounded, direction only); non-X control -0.009
  [-0.025, +0.008] [RES, REG addendum]. PLAN's P3 prediction (unseen below seen) **held** [PLAN].
- Mechanism predictions scored both ways: "rhythm games carry best" held; "Social VR least" failed
  (Alyx least) [RES].
- **Table 6** (per held-out application, P3 / Z-676 / C2-hi, contrasts, seeds). Data: [P3J], [P3S].
- **Figure 4** (forest plot of P3 - Z-676 per application + pooled + covered/uncovered, with the +0.030
  threshold drawn). Data: [P3S], [P3SPLIT].

**R4. Identity count x exposure, and dose.**
- Without exposure identity count is flat: Z-676 - zero-shot -0.013 [-0.039, +0.013] [RES]; consistent
  with earlier transfer measurements on other corpora [CL "Cross-corpus transfer", "`dyn`"].
- With exposure it is not: C2-hi (495 ids, 14.1%) - C2-lo (3,095 ids, 3.87%) -0.061 [-0.099, -0.026]
  [RES].
- Dose separated at fixed identities: C2-lo-half - C2-lo -0.028 [-0.062, +0.009], unresolved against
  the -0.03 edge, 10-min -0.124 [DOSE, RES]; 20% cut -0.009 [-0.025, +0.008] [RES]. Dose favours C2-hi,
  so correcting for it widens the scale effect [RES].
- **Figure 5** (2x2 exposure x identity count, A1 with CIs, dose annotated). Data: [AGG], [DOSE].

**R5. Static-cue audit of our own headline (P2) - main results, not appendix.**
- Zero-shot raw A1 0.351 (0.364 / 0.353 / 0.335); raw - dyn +0.117 [+0.042, +0.192]; A0 0.723,
  +0.223 [+0.184, +0.263]; 10-min 0.434 (+0.077) [P2J, RES]. **Every raw seed selected epoch 1 of 16**
  [RES]. Registered band +0.00..+0.06, falsifier < -0.03: falsifier excluded, size unresolved against
  the band edge [RES].
- Raw verification AUC vs the recorded-position lookup (0.704-0.733 vs 0.585-0.598) shows raw reads
  more than mean position - posture as well as height. Report as a within-verification comparison
  only, never against rank-1 [RES, CL].
- With exposure: raw C2-lo A1 0.404, raw - dyn +0.029 [-0.068, +0.134], 1 seed, unresolved; 10-min raw
  0.497 vs dyn 0.711 (-0.214) [P2J, RES].
- Four-arm averaging table: ten-minute gain dyn zero-shot +0.123, raw zero-shot +0.103, dyn C2-lo
  +0.336, raw C2-lo +0.093 [RES] - the time advantage belongs to exposure, not encoding.
- Epoch-1 selection is ~3x less seed-stable (A1 range 0.029 vs 0.010) [RES].
- Why the headline stays on `dyn`: decided before any raw number; BRV discards head position by
  construction, so a raw comparison would win partly on a cue the reference excludes [CL, RES].
  Their numbers are **not** placement-inflated and the paper must not imply so [PLAN].
- Cross-application raw carries height (P=0.754), not placement (0.527); within-application raw
  carries placement (0.7525) and is never quoted as biometric [CL, RES].
- **Table 7** (four arms: 1 window, 10 min, averaging gain, seeds). Data: [RES] four-arm table, [P2J].
- **Figure 6** (paired per-user raw vs dyn, zero-shot, 17 users). Data: [AGG] per_user +
  raw seed JSONs on the branch.

**R6. Train-user-only alignment (the reference's future-work proposal) - negative.**
- Zero-shot: A2' - A1 +0.026 [+0.000, +0.051] (registered >= +0.15; 13x below their +0.34); A2 - A1
  +0.011 [-0.020, +0.041] (band +0.05..+0.20 excluded); A2-null - A1 -0.074 [-0.126, -0.029]
  (person-specific but small); A2-full - A2 -0.055 [-0.086, -0.029] (rank argument) [RES, AGG].
- Exposed arms: C2-lo A2 - A1 -0.003 [-0.011, +0.005] over 3 seeds; A2' - A1 +0.148 / -0.004 / +0.001
  ("present in one of three runs", never a rate); A2-null -0.235 [RES].
- Across instruments: C1 +0.011, C1-full +0.089, Z-676 +0.011, C2-hi +0.002 (0.8 sigma at that arm's
  spread) [RES].
- Mechanism: 32 people appear in >= 2 applications among users 0-31, the cap on correspondences; no
  pretraining raises it; a test-fitted single-run bound is not evidence of an orthogonal relationship
  [RES]. We do not claim their +0.34 is wrong [RES].
- **[DISCREPANCY, gap G3]** "A2 - A1 <= 0 on 14 / 18 / all checkpoints" ([PLAN] 14, [RES] claim 4 "all
  18", [RES] alignment section "all nine ... zero-shot x3") contradicts the zero-shot per-seed values
  +0.016 / +0.011 / +0.006 [RES, AGG] and P3 Superhot +0.005 [P3J]. Safe wording: "never resolvably
  above zero; largest +0.011 [-0.020, +0.041]".
- **[DISCREPANCY, gap G4]** "run-dependent at identical configuration ... present in three of five P3
  runs" - the P3 runs are five *different* configurations; only the C2-lo seeds are identical. Keep
  the two statements separate.
- **Table 8** (alignment arms x instruments). Data: [AGG], [P3J], [RES].
- **Figure 7** (A2' - A1 and A2 - A1 per checkpoint, dot plot). Data: [AGG], [P3J].

**R7. Margin/scale reversal.**
- +0.016 (5 folds, paired, t=4.31) at 419 identities on verification AUC [CL]; zero-shot A1 at 0.1/15:
  0.206 [0.158, 0.262], difference -0.028 [-0.045, -0.012], 1 seed; A0 -0.039, 10-min -0.045
  [MARGIN, RES]. **Note the metrics differ** (AUC at 419 vs rank-1 at N=17); the sign reversal is the
  claim, not a magnitude ratio. Registered as a screen with the sign flip named in advance [RES
  Amendment 7].
- Short subsection or discussion paragraph; no table needed beyond one row in Table 5.

**R8. Per-user distributions (metric contribution).**
- Zero-shot seed-1 per-user A1 ranges 0.08-0.48; five of 17 above 0.33 [RES]. Produce for all arms
  from [AGG] `per_user`, and for the reference from [SCH-REL] (their per-user off-diagonal means
  range ~0.07-0.37 - verified from the file, not yet in any project document).
- **Figure 8** (per-user cross-application rank-1, ours vs theirs, same 17 users once user order is
  verified - gap G7). Data: [AGG], [SCH-REL].

### 4.7 Discussion
- **What transfers**: identity count alone does not cross a domain boundary; exposure does, and
  carries partially to an unseen application; identity count pays only with exposure (R2-R4).
  Contrast with earlier null levers (identity count flat across corpora; activity diversity null)
  [CL]. Present as "first data-side lever *we have measured*", not a field-wide claim.
- **Task structure**: rhythm games transfer to each other at twice the mean (seed-1 cells 0.459 /
  0.406) [RES]; rhythm-game hold-outs carry fully; Alyx least (R3).
- **Privacy implications**:
  - Behaviour-only assessments (BRV-style) understate single-window risk: epoch-1 static cues add
    +0.117 head-only [P2J].
  - Static-only reasoning understates what a trained behavioural model does with time: 0.711 at ten
    minutes when the model has seen the application class [RES].
  - Head-only suffices for non-trivial cross-application identification, so glasses without hand
    tracking are in scope for the risk, not outside it [CL scope].
  - Aggregate rank-1 hides exposed individuals: report per-user distributions (R8) [PLAN metric].
  - The reference paper judged the threat "moderate at best" [SCH section 1]; our numbers bear on that
    reading - phrase carefully and only after P-1.
- **Alignment as a corpus specification**: an honest alignment route needs many more
  multi-application participants than 32 [RES].
- **Within-application gap** (0.616 C2-lo / 0.291 C1-full vs 0.831) is unresolved and confounded
  between sensor set and model family; not attributable to head-only [RES, PLAN]. If P-2 lands, it
  scores their model on their inputs but still does not isolate the sensor set.

### 4.8 Limitations (not including scope)
- Across-XR is one sitting per participant: no cross-day evidence; cross-application separation is
  up to ~1 h [PLAN, CL]. No within-application temporal separation [CL].
- 17 test users: user-level uncertainty dominates; intervals are wide [PLAN power].
- Many arms are single-seed (C1, C1-full, Z-676, C2-hi, C2-lo-half, three P3 applications, raw C2-lo,
  M-zero) [RES].
- Pretraining corpora are two activities (Beat Saber, Half-Life: Alyx), both also Across-XR
  applications [REG].
- Decision rule, window length and rate differ from the reference until P-1 [REG, coordinator note].
- Single node, single code identity (`517cdaa57b`), single stack [RES].

### 4.9 Conclusion
- Three sentences: head-only placement zero-shot; exposure + pretraining gives the best
  cross-application figure on the reference split and carries partially to unseen applications;
  static-cue audit and the negative alignment result change how cross-application risk should be
  assessed.

---

## 5. Claims and their evidence

All rank-1 at **N=17**, test users 32-48, chance 0.0588. CIs: user bootstrap over the 17.

| claim | number | CI95 | seeds | source |
| --- | --- | --- | --- | --- |
| Reference within-app rank-1 (15 s, head + controllers) | 0.831 | range 0.723-0.880 over cells | - | [SCH 6.2.2], reproduced from [SCH-REL] as 0.8314 |
| Reference cross-app rank-1 | 0.180 | range 0.105-0.226; mean per-cell user sd 0.151 | - | [SCH 6.2.2], [SCH-REL] 0.1804 |
| Reference cross-app 10-min | 0.308 | range 0.090-0.577 | - | [SCH 6.2.3], [SCH-REL] 0.3082 |
| Reference test-fitted alignment | 0.523 single / 0.943 10-min | - | - | [SCH-FV] via [LIT], [PLAN] (not in [SCH] v1) |
| Zero-shot cross-app | 0.234 | [0.181, 0.292] | 3 | [RES], [AGG] |
| Zero-shot within-app | 0.500 | [0.462, 0.538] | 3 | [AGG] |
| Zero-shot cross-app 10-min | 0.357 | not reported | 3 | [AGG] |
| C1 (their 23 users only) | 0.131 | [0.088, 0.177] | 1 | [RES] |
| C1-full | 0.164 | [0.128, 0.205] | 1 | [RES] |
| Z-676 | 0.218 | [0.168, 0.271] | 1 | [RES] |
| C2-hi | 0.307 | [0.263, 0.354] | 1 | [RES] |
| **C2-lo cross-app (headline)** | **0.375** | [0.321, 0.435] | 3 | [RES], [AGG] |
| C2-lo 10-min | 0.711 | not reported | 3 | [RES] |
| C2-lo - zero-shot | +0.141 | [+0.100, +0.183] | 3 paired | [RES] |
| C2-hi - Z-676 | +0.089 | [+0.048, +0.131] | 1 | [RES] |
| Exposure carries to unseen app (P3 - Z-676, pooled) | +0.049 | [+0.021, +0.078] | 1-2 per app | [P3S], [RES] |
| P3 threshold (lower bound > +0.030) | not met (0.017-0.022) | - | - | [RES], [REG Amd 4] |
| Uncovered vs covered carry | +0.046 vs +0.055 | [+0.017, +0.074] / [+0.024, +0.089] | as above | [P3S], [P3SPLIT] |
| Synth Riders carry | +0.077 / +0.053 | seed-avg [+0.034, +0.095] | 2 | [P3S] |
| P3 - C2-hi (unseen below seen) | -0.036 | [-0.054, -0.018] | 1 | [RES], [PLAN] |
| Identity count flat without exposure | -0.013 | [-0.039, +0.013] | 1 | [RES] |
| Identity count not flat with exposure (C2-hi - C2-lo) | -0.061 | [-0.099, -0.026] | 1 vs 1 | [RES] |
| Dose, halving | -0.028 | [-0.062, +0.009] | 1 vs 3 | [DOSE] |
| Dose, 20% cut (P3 non-X control) | -0.009 | [-0.025, +0.008] | 1 | [RES] |
| Raw zero-shot cross-app | 0.351 | per seed 0.364/0.353/0.335 | 3 | [P2J] |
| raw - dyn, zero-shot | +0.117 | [+0.042, +0.192] | 3 vs 3 | [P2J] |
| raw epoch selected | epoch 1 of 16, all seeds | - | 3 (+1 C2-lo) | [RES] |
| raw - dyn, with exposure | +0.029 | [-0.068, +0.134] | 1 vs 3 | [P2J] |
| 10-min, exposed: dyn vs raw | 0.711 vs 0.497 | not reported | 3 vs 1 | [RES], [P2J] |
| Honest alignment A2 - A1, zero-shot | +0.011 | [-0.020, +0.041] | 3 | [RES] |
| Honest alignment A2 - A1, C2-lo | -0.003 | [-0.011, +0.005] | 3 | [RES] |
| Test-fitted ceiling A2' - A1, zero-shot | +0.026 | [+0.000, +0.051] | 3 | [RES] |
| Test-fitted ceiling run-dependent, C2-lo | +0.148 / -0.004 / +0.001 | per seed | 3 | [RES] |
| Correspondence cap | 32 people in >= 2 apps | - | - | [RES] |
| Margin/scale at 419 ids (verification AUC) | +0.016 | t(4)=4.31, 5/5 folds | 5 folds | [CL] |
| Margin/scale at scale (rank-1) | -0.028 | [-0.045, -0.012] | 1 vs 3 | [MARGIN] |
| Gates | 23/23 PASS | gaps 5.3e-8 to 2.9e-4 | - | [RES] |

## 6. Threats to validity

- **Internal**
  - *Headline chosen after P1 was registered.* P1 targeted zero-shot; C2-lo was registered later as a
    matched arm (C2 - A1 band in [REG]). Say explicitly that the headline recommendation is post hoc
    relative to P1, and that C2-lo's contrast was registered before it ran.
  - *Many arms and seven amendments* (forking paths). Mitigation: amendments are dated, precede their
    runs, and are kept in place [REG]; bands read by interval; negatives reported.
  - *Single-seed arms* (section 4.8). Seed variance is arm-specific: 0.010 on trained-out A1 vs 0.029
    on epoch-1 raw vs 0.152 on C2-lo A2' [RES, CL].
  - *Budget*: C1 under-trained (patience fired at 15 on nine validation users); C2-hi censored at
    120/120 [RES].
  - *Selection on nine validation users* for Across-XR-only arms [RES].
- **Construct**
  - *Metric not matched* to the reference (template vs nearest-window kNN; 10 s vs 15 s; 20 Hz vs 30
    fps) - resolved only by PENDING P-1/P-2.
  - *10-min figure* is one decision per user per cell and has no CI [REG, RES].
  - *Within-application* A0 is same-recording split and carries placement under raw [CL].
- **External**
  - One corpus of 49 people, one sitting, one headset, standing [SCH 4].
  - Pretraining activity overlap with two of five applications [REG].
- **Comparison validity**
  - Sensors (head vs head + controllers) and architecture differ simultaneously; no sensor-set claim
    is available [RES].
  - The released per-user values [SCH-REL] must be verified for application numbering and user order
    before pairing (gaps G6, G7).
- **Reproducibility / durability**
  - The programme record is on an unmerged branch (gap G8); the 18+5 gated checkpoints exist on one
    disk (`runs/miami-alignment/`) and are BOXRR-derived under DUA clause 15 [RES].
  - All rows at one code identity and one stack; `code_identity` does not cover converters or
    dependency versions [CL].

## 7. Pending work (no numbers exist; do not invent any)

- **P-1. Formal paired per-user comparison against the reference's released per-user values**
  (zero-shot and C2-lo), one harness scoring both. Their `accuracy_values.json` holds per-user
  `precision_at_1` for all 17 test users in every cell and reproduces 0.8314 / 0.1804 / 0.3082
  [SCH-REL; verified 2026-09-15 while drafting]. **Outcome may upgrade the zero-shot framing from
  "placement" to a tested result, or downgrade it.** Until it lands the outline must not claim that no
  formal test is possible (it is now possible) nor claim a beat.
- **P-2. Scoring the reference's released model** (`evaluation/models/slm_model/max_precision_at_1.ckpt`,
  plus `embeddings.pkl`) under our harness - "scoring their released model", not "reproducing their
  architecture"; no retraining required [coordinator note; files present in
  `external_sota/schach2026/training-and-evaluation/evaluation/`]. Gives a single-decision-rule
  comparison on their inputs (head + controllers, BRV).
- **P-3. Rack et al. 2023 baseline reproduction** on Across-XR (architecture held in
  `external_sota/Versatile-XR-User-Identification`); training run not complete [PLAN, brief].
- **P-4. Merge the `worktree-across-xr-alignment` record to main** so every [RES]-tagged number is
  citable from the default branch.
- **P-5. Update [PLAN] OUTCOMES** (stale P2 "UNRUN", seed-1 P3 figures) - not done here by instruction.

## 8. Figures and tables to produce

| id | content | data source |
| --- | --- | --- |
| Fig 1 | Teaser: cross-app rank-1 @N=17, zero-shot / C2-lo / reference / chance, 1 window and 10 min | [AGG], [SCH] |
| Fig 2 | 5x5 cell heatmaps, ours (zero-shot, C2-lo) beside the reference | [AGG] `per_cell`, [SCH-REL] |
| Fig 3 | A1 by arm with user-bootstrap CIs, reference and chance lines | [AGG], [RES] |
| Fig 4 | P3 forest plot per held-out app, pooled, covered/uncovered, +0.030 threshold | [P3S], [P3SPLIT] |
| Fig 5 | Exposure x identity count 2x2 with dose annotation | [AGG], [DOSE] |
| Fig 6 | Paired per-user raw vs dyn, zero-shot | [AGG], raw seed JSONs |
| Fig 7 | A2' - A1 and A2 - A1 per checkpoint | [AGG], [P3J] |
| Fig 8 | Per-user cross-app rank-1 distribution, ours vs reference | [AGG] `per_user`, [SCH-REL] |
| Tab 1 | Datasets (users, applications, sessions, role, licence/DUA) | [SCH], [CL], DATASET_CATALOGUE |
| Tab 2 | Arms: composition, trained identities, dose, seeds | [RES], [REG] |
| Tab 3 | Protocol comparison with the reference | [SCH 5], [REG] |
| Tab 4 | Zero-shot A0/A1/10-min per seed vs reference | [AGG], [SCH] |
| Tab 5 | All arms A0/A1/10-min with CIs (incl. M-zero row) | [AGG], [RES], [MARGIN] |
| Tab 6 | P3 per held-out app | [P3J], [P3S] |
| Tab 7 | Four-arm ten-minute averaging table (raw/dyn x exposure) | [RES], [P2J] |
| Tab 8 | Alignment arms x instruments | [AGG], [P3J], [RES] |
| Tab 9 | Registered predictions and verdicts (P1-P3, amendments) | [PLAN], [REG], [RES] |

## 9. Citation plan (must-cite)

- Schach, Rack, McMahan, Latoschik 2026, Frontiers in VR, doi:10.3389/frvir.2026.1743491
  (arXiv:2509.08539); Across-XR data (CC BY-NC-SA 4.0).
- **Nair, Guo, Mattern, Wang, O'Brien, Rosenberg, Song 2023, "Unique Identification of 50,000+ Virtual
  Reality Users from Head & Hand Motion Data" (USENIX Security; arXiv:2302.08927) - required by the
  BOXRR-23 DUA** (text as in `prepare_boxrr.py` CITATION_TEXT).
- Nair et al. 2023, BOXRR-23 dataset paper, doi:10.1109/TVCG.2024.3372087 (arXiv:2310.00430).
- Rack, Fernando, Yalcin, Hotho, Latoschik 2023, "Who is Alyx?", Frontiers in VR,
  doi:10.3389/frvir.2023.1272234; dataset Zenodo 10.5281/zenodo.8379914.
- Rack et al. 2024, "Versatile User Identification in XR using Pretrained Similarity-Learning",
  arXiv:2302.07517.
- Rack, Hotho, Latoschik 2022, encodings (AIVR); Rack et al. 2024, Motion Learning Toolbox (VRW).
- Baldoni et al. 2025 (IEEE VR); Miller et al. 2020/2021; Rogers et al. 2015 (all in [SCH] references).
- AM-Softmax original paper for the objective - not in any project source; add the reference by hand.

---

## 10. Open questions and gaps found while drafting

- **G1. The alignment numbers are not in the local PDF.** `external_sota/schach2026.pdf` is arXiv v1;
  it has no section 6.2.5, no 52.3% / 94.3%, no "diagnostic upper bound" language, and its section 8
  does not propose train-user-only alignment (grep for "orthogonal|Procrustes|52.3" returns nothing).
  Those come from the Frontiers version (`frvir-7-1743491.pdf`, cited in [LIT]) which is not in the
  repo. Obtain it before quoting section numbers or the +0.34.
- **G2. P3 figures disagree between documents.** [PLAN] OUTCOMES: +0.053 [+0.022, +0.083], uncovered
  +0.052 (seed-1 only). [RES] claim 3, [P3S], [CL]: +0.049 [+0.021, +0.078], uncovered +0.046
  [+0.017, +0.074] (seed-averaged). Use the latter.
- **G3. "A2 - A1 <= 0 on every checkpoint" is false as written** (zero-shot +0.016/+0.011/+0.006, P3
  Superhot +0.005); counts also vary (14 / 18 / "all nine").
- **G4. "Run-dependent at identical configuration" is supported by the three C2-lo seeds only**; the
  "three of five P3 runs" figure is across different configurations.
- **G5. Identity-count naming** (4,096 vs 3,072 / 3,095; 676 vs 495) differs between [RES]/[PLAN]
  and [CL].
- **G6. Application numbering in [SCH-REL]** (`ref_comment`/`query_comment` 1-5) must be mapped to
  applications before pairing; [PLAN] warns that our `game_id` order is not play order.
- **G7. User order in [SCH-REL]'s 17-element arrays** must be verified to correspond to users 32-48
  in our order before any per-user pairing.
- **G8. The programme record is on an unmerged branch** (`origin/worktree-across-xr-alignment`).
- **G9. DUA citation identity.** [CL] says "Nair et al. 2023"; our `CITATION.txt` names the 50,000+
  identification paper, not the BOXRR-23 dataset paper. Confirm against the DUA text.
- **G10. [PLAN] OUTCOMES is stale**: P2 marked "UNRUN" but P2 ran (three raw seeds + raw C2-lo) [RES];
  [RES]'s own header (line 45) also still says P2 not run, contradicted by its close-out.
- **G11. The "sd 15.1" label.** [CL] and [RES] call it "across-cell sd"; [SCH] 6.2.2 and [SCH-REL]
  show it is the mean over cells of the across-user standard deviation (population sd: 0.1506).
- **G12. No CIs on any ten-minute figure** (0.357, 0.711, 0.497 ...), yet the 10-min contrast (0.711
  vs 0.308; 0.711 vs 0.497) is used in claims.
- **G13. The margin/scale reversal compares different metrics** (verification AUC at 419 vs rank-1 at
  4,096); state the sign reversal only.
- **G14. [SCH] internal inconsistency**: classification accuracy 43.2% (section 6.3) vs 43.5%
  (section 7.2). Minor; cite 43.2% if at all.
- **G15. Within-application protocol of the reference** (how gallery and query are separated inside
  one recording) is not specified in [SCH] v1; our half/half split may not match. Check [SCH-FV] or
  their `slm_compute_accuracies.py`.
- **G16. Rhythm-game cell transfer (0.459 / 0.406)** is seed-1 only [RES]; recompute over seeds from
  [AGG] if used.
- **G17. The earlier [RES] mechanism sentence** "the honest route still fails for the rank reason"
  (C2-lo seed-1 section) is superseded by "the reason is not the rank argument" (C1-full section);
  the paper must use the latter.

---

## RESOLUTIONS (coordinator, 2026-09-15) - verified against primary sources

- **G1 resolved.** The GOPA material is in the **Frontiers version** (doi:10.3389/frvir.2026.1743491, 16 pp.,
  `external_sota/papers/schach2026_frontiers.pdf`): section 6.2.5, aligned cross-application 52.3% and 94.3% at
  ten minutes, aligned within-application 82.9%, "post hoc diagnostic upper bound", and future work to "learn
  orthogonal transformations only on training/validation users and then apply them to unseen test users and
  applications". Cite the Frontiers version for all of it; do not assert a section number for the future-work
  text. Unaligned headline numbers (83.1 / 18.0 / 78.5 / 15.1) are identical in both versions.
- **G2, G3, G4, G10, G11 corrected at source** in `docs/PAPER_PLAN.md` and `CLAUDE.md`: P3 is +0.049 [+0.021,
  +0.078]; A2 - A1 is "never resolvably above zero", not "<= 0"; run-dependence rests only on C2-lo's three seeds;
  P2 ran (+0.117); the 15.1 is a mean per-cell across-user sd.
- **Margin/scale**: only the sign reversal is claimable (verification AUC at 419 vs rank-1 at 4,096).
- **BOXRR citation (gap 8) resolved - our CITATION.txt was right.** The DUA clause 5 names exactly one required
  citation: Nair et al., *Unique Identification of 50,000+ VR Users from Head & Hand Motion Data*,
  arXiv:2302.08927. Cite the BOXRR-23 dataset paper (arXiv:2310.00430) as well, but it does not discharge clause 5.
- **43.2% vs 43.5%** is an inconsistency inside Schach et al. itself (both versions); quote with a note.
- **Still open**: ten-minute figures have no CI; their JSON's application numbering (1-5) must be mapped before any
  per-user pairing (sent to New Gen); the programme record is still unmerged.

---

## RESOLUTIONS, ROUND 2 (coordinator, 2026-09-15) - from primary sources on disk

### G6 RESOLVED - `comment` is `game_id`, and their Readme names the applications

`dataset-preprocessing/src/cross-application/data_selection_slm.py:29` writes `'comment': game_id`
with **no reindexing**, and `dataset/Readme.md:17-21` gives the mapping outright:

| `comment` / `game_id` | application | in our pretraining corpus? |
| --- | --- | --- |
| 1 | Superhot VR | no |
| 2 | Half-Life: Alyx | yes (who_is_alyx) |
| 3 | Beat Saber | yes (BOXRR-23) |
| 4 | Synth Riders | **no** - the uncovered control |
| 5 | Social VR Scenario | no |

This agrees with the `take_id` ordering already recorded in `docs/COORDINATION.md`, so the two
independent readings match. **The caveat that `game_id` is not play order still stands** - the
paper's play order is Synth Riders, Superhot, Beat Saber, Alyx, Social VR (`game_id` 4,1,3,2,5) -
but it does not affect any cross-application cell, which is unordered by construction.

### G7 RESOLVED - index k of every 17-element array is user 32+k

Four links, each checked rather than assumed:

1. **Splits are one file per user, filename == `user_id`**: `train/` 0-22 (23 users), `valid/`
   23-31 (9), `test/` 32-48 (17). Confirmed by reading `user_id` inside `test/32.csv` - constant
   at 32 over 139,077 rows. *(This also confirms their 0.180 is user-disjoint: trained on 0-22,
   validated on 23-31, tested on 32-48 - the same regime as our C1/C2 arms.)*
2. **Label remap sorts**: `Dataset._remap_labels` builds its mapping from `torch.unique(labels)`,
   which returns ascending values, so class k is the k-th smallest `user_id`. Independent of the
   `glob("*")` order in which files are concatenated.
3. **The per-class array sorts the same way**: pytorch-metric-learning's `maybe_get_avg_of_avgs`
   calls `get_unique_labels` -> `torch.unique(labels, dim=0)` (ascending) and returns
   `average_per_class` in that order.
4. **Nothing was skipped**: their sequence code *does* drop a class with too few samples
   (`"Skipping class ... not enough samples"`), which would shorten an array and silently break the
   correspondence. It did not fire - **every one of the 44 per-user arrays is length 17 in all 35
   cells**, checked exhaustively.

**So the mapping is the identity in our numbering: array index 0 = user 32, index 16 = user 48.**
Link 4 is the one worth keeping: the hazard here is not a wrong mapping, it is a *variable-length*
one, and it is invisible unless counted.

### G12 RESOLVED FOR THE REFERENCE SIDE - their ten-minute figure has a CI now

Their JSON ships `sequence_top_1_accuracy_list_*_mins` per user, so the reference side needs no
re-run. Per-user means over cells, cluster bootstrap over the 17 users, 10,000 resamples:

| Schach et al. | single 15 s window | 10-minute sequence |
| --- | --- | --- |
| **cross-application** (20 off-diagonal cells) | **0.1804** [0.1396, 0.2246] | **0.3082** [0.2061, 0.4169] |
| within-application (5 diagonal cells) | 0.8314 [0.8102, 0.8520] | **1.0000** [1.0000, 1.0000] |

All four point estimates reproduce the published figures exactly. Three consequences:

1. **The zero-shot placement is now clearly the right call, and quantifiably so.** Their
   cross-application interval reaches **0.2246**, and our zero-shot 0.234 [0.181, 0.292] overlaps it
   heavily. "Every seed sits above their reported mean" remains true and remains the only sentence
   available; anything stronger is now refuted rather than merely unsupported.
2. **The ten-minute contrast survives with room to spare.** Their 0.308 tops out at **0.417**, and
   our exposed arm reads 0.66-0.711 - outside their interval entirely. This is the cleanest
   separation anywhere in the comparison, and it is now an interval statement rather than two points.
3. **Their within-application ten-minute figure is 1.0000 for every one of the 17 users.** The
   metric is saturated there, which is the outline's own argument for why the ten-minute number
   separates methods only across applications - now demonstrated on their data rather than asserted.

**And their own per-user spread makes our metric contribution for us.** Cross-application per-user
rank-1 runs **0.068 to 0.371** on a single window and **0.043 to 0.818** at ten minutes - on the
same 17 people. A risk assessment reported as a mean conceals a person identified four-fifths of the
time behind a population figure of 0.31. That is precisely the "report the distribution, not only
the mean" contribution, and it can now be made **using the reference's own published numbers**,
which is far stronger than making it only on ours.

### G15 RESOLVED - and the answer is that they do not separate gallery from query

The within-application protocol is `ref = embeddings[comments == q][::150]` against
`query = embeddings[comments == q]` - the reference is a **subset of the queries**, from the same
unbroken recording, with `ref_includes_query=False` so the kNN never excludes the identical vector.
Full derivation and the graded overlap in `CLAUDE.md`. **Our half/half split does not match theirs
and should not be made to**: the correct handling is not to pair A0 against their 0.831 at all.

### G8 RESOLVED - both branches are merged

`worktree-across-xr-alignment` (83 commits) and `miami-server` (5) are on `origin/main` as of
2026-09-15 (PRs #9, #10). The programme record is no longer on an unmerged branch.

### G5 RESOLVED - the two numbers are the pool and the list, and only one is the treatment

`4,096` and `676` are **identity pools before the 25% validation draw**; `3,072` and `495` are the
**training lists the loader actually held**. 4,096 x 0.75 = 3,072 exactly, and C2-lo's 3,095 is that
same 3,072 plus the 23 Across-XR training users - which is why C2-lo's window arithmetic closes at
540,107 - 20,896 = 519,211 against the zero-shot set. **Quote the trained-identity count, never the
pool**, per this project's own rule that a matched count is a claim about whichever list you
counted. Give the pool only where the validation draw itself is under discussion.

### Still open, with what each needs

- **G12, our side.** Our ten-minute figures (0.357, 0.66, 0.711, 0.497) still have no CI. The
  reference side is done; ours needs the same per-user bootstrap from `[AGG]` - New Gen's harness
  already produces per-user arrays, so this is an aggregation, not a re-run.
- **G16.** Rhythm-game cell transfer (0.459 / 0.406) is seed-1 only; recompute over the three C2-lo
  seeds before it appears in the paper.
- **G17.** Use "the reason is not the rank argument" (C1-full) and strike the superseded
  "the honest route still fails for the rank reason". A wording decision, not a measurement.

### One environment fact the reproductions must not share

**Rack et al. require `pytorch-metric-learning==1.7.3`; Schach et al. require 2.x.** Schach's
`MotionAccuracyCalculator.get_accuracy` is declared `(query, query_labels, reference,
reference_labels, ref_includes_query=...)` - the 2.x signature and the 2.x keyword - while Rack's
code dies on 2.x with `unexpected keyword argument 'embeddings_come_from_same_source'` and is pinned
to the last 1.x. Since 1.x and 2.x also **reorder the positional arguments**, a single environment
serving both would silently pass reference embeddings as query labels and return a plausible number.
**The two SOTA reproductions need separate virtualenvs**, and this is a correctness matter rather
than a convenience one.
