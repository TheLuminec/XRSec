# Progress report: identifying XR users across applications from head motion alone

Prepared 2026-09-16 for Dr. Feng. This file is the only source for the presentation. Every number
in it is copied from a named file in the repository, and each result carries a status label.

**Source key** (the tag in square brackets after a number names the file it came from):

| tag | file |
| --- | --- |
| [PLAN] | `docs/PAPER_PLAN.md` (where a later section corrects an earlier one, the later figure is used here) |
| [OUT] | `docs/PAPER_OUTLINE.md`, including its RESOLUTIONS blocks |
| [CL] | `CLAUDE.md` (the project's running record of results) |
| [CAT] | `docs/DATASET_CATALOGUE.md` |
| [PAIR] | `docs/acceptance/schach_paired.json` (the settled paired comparison) |
| [GATE] | `docs/acceptance/schach_release_gate.json` (reproduction of the published numbers) |
| [QG] | `docs/acceptance/questset_geometry.json` |
| [QL] | `docs/acceptance/questset_static_lookup.json` |
| [P3S] | `docs/acceptance/across_xr_alignment_p3_stability.json` |
| [RES] | `docs/acceptance/across_xr_alignment_RESULTS.md` |
| [COORD] | `docs/COORDINATION.md` (Questset arm registrations) |
| [CSV] | `docs/acceptance/schach_per_user.csv` (per-user values behind the existing figure) |

**Status labels used throughout:**

| label | meaning |
| --- | --- |
| **SETTLED** | gated (each checkpoint reproduces its own recorded score before being used) and a certificate is committed |
| **REGISTERED-AND-RUNNING** | prediction written down before the run; the run is executing or queued on the lab GPU server; **no number exists yet** |
| **EXPLORATORY** | observed after the fact, not registered, not claimed |
| **PENDING** | not yet run, or not yet computed |

Numbers taken from a certificate are shown to three or four decimals. A number with fewer decimals is
quoted exactly as the source text gives it.

---

# Part A - Summary for the presenter

## A.1 Executive summary

The student set a five-step goal on 2026-09-10: publish a paper on how well biometric identification
generalises **across XR applications**. All five steps are now complete. The public dataset is
Across-XR: 49 users, each recorded in 5 applications [PLAN]. The state of the art (SOTA) is Schach,
Rack, McMahan and Latoschik (2026, Frontiers in VR). That paper reports 83.1% rank-1 identification
within an application, which falls to **18.0%** across applications among 17 unseen users (chance
5.88%). The system uses head motion plus both hand controllers [PLAN]. We reproduced their published
numbers bit-exactly from their released model and evaluation code (35 cells, max abs diff 0.0)
[PLAN, GATE]. We then compared our system with theirs user by user, on their 17 test users and under
their own metric. Our system uses **head motion only**, a scope chosen so that it runs on AR glasses.
It was pretrained on 3,072 identities from other corpora and trained on the same 23 Across-XR
training users they used. That system (the "C2-lo" arm) **beats their released model**: **+0.119
[+0.050, +0.192]** under their metric (0.299 vs 0.180, 15 of 17 users better) and **+0.176 [+0.092,
+0.260]** under ours (0.375 vs 0.199) [PLAN, PAIR]. Our zero-shot system never saw Across-XR. Its
comparison is **unresolved** (+0.025 [-0.031, +0.080]), so it is not reported as a beat [PLAN, PAIR].
Four further results come with the beat:
- **A static-cue audit of our own result.** A model that keeps absolute head position reaches 0.351
  after one training epoch, +0.117 over our behaviour-only headline. So a behaviour-only risk
  assessment understates the risk [PLAN, OUT].
- **The SOTA's own proposed fix fails, with a mechanism.** This is train-user-only orthogonal
  alignment [PLAN].
- **Exposure carries to an unseen application** (+0.049) [PLAN, P3S].
- **The per-user distribution.** It shows that the population mean hides highly exposed individuals
  [PLAN].

The plan recommends writing the paper now, framed as a rigorous re-assessment of cross-application
XR biometric risk rather than as "our model is better" [PLAN].

## A.2 The five-step goal

| step | status | evidence |
| --- | --- | --- |
| 1. Public dataset | **DONE** | Across-XR, 49 users x 5 applications, CC BY-NC-SA 4.0, converted and on three machines [PLAN] |
| 2. SOTA: run their code, understand their story | **DONE** | Schach et al. 2026. We cloned all three of their repositories and re-ran their released model's evaluation code; it reproduces their published numbers bit-exactly (35 cells, max abs diff 0.0) [PLAN, GATE] |
| 3. Evaluation metric | **DONE** | Rank-1 at N=17 on their test users 32-48, gallery from one application and probe from another, averaged over 20 ordered pairs, matched to theirs. Our addition is reporting the per-user distribution [PLAN] |
| 4. Our algorithm | **DONE** | `dyn` encoding, `identity_softmax` objective, head only, 10 s windows, with exposure to the corpus's *other* participants. The orthogonal-alignment component was registered, run and closed negative [PLAN] |
| 5. Beat SOTA | **DONE for the exposed arm** | C2-lo +0.119 [+0.050, +0.192] against their 0.180, on their people, under their metric, gate bit-exact. Zero-shot is **UNRESOLVED** [PLAN] |

## A.3 Headline results

| # | result | number | interval | status |
| --- | --- | --- | --- | --- |
| 1 | Exposed head-only system (C2-lo) vs their released head-plus-controllers model, paired per user, **their** metric, single window | **+0.119** (0.299 vs 0.180), 15/17 users | [+0.050, +0.192] | **SETTLED - BEAT** [PLAN, PAIR] |
| 1b | Same contrast, **our** metric | **+0.176** (0.375 vs 0.199), 16/17 users | [+0.092, +0.260] | **SETTLED - BEAT** [PLAN, PAIR] |
| 2 | Zero-shot head-only system (never trained on Across-XR) vs theirs, their metric | +0.025 (0.206 vs 0.180), 10/17 users | [-0.031, +0.080] | **SETTLED - UNRESOLVED** (not a beat) [PLAN, PAIR] |
| 3 | Static-cue audit: `raw` minus `dyn`, zero-shot, cross-application rank-1 | **+0.117** (0.351 vs 0.234); every `raw` seed selected epoch 1 of 16 | [+0.042, +0.192] | **SETTLED** [PLAN, OUT] |
| 4 | Exposure carries to an application held out of training (P3 minus Z-676, pooled over five) | **+0.049** | [+0.021, +0.078] | **SETTLED** (the stricter registered threshold, CI lower bound above +0.030, is **not met**) [PLAN, P3S] |
| 5 | Train-user-only orthogonal alignment (the SOTA's future-work proposal), zero-shot, A2 minus A1 | +0.011 | [-0.020, +0.041] | **SETTLED - NEGATIVE** (registered band +0.05 to +0.20 excluded) [OUT] |

Secondary metric, always shown beside result 1 and never instead of it: over a ten-minute sequence,
C2-lo beats theirs by **+0.355 [+0.202, +0.499]** under their sequence metric and **+0.423 [+0.293,
+0.546]** under our vote [PLAN, PAIR].

---

# Part B - Suggested slide sequence (20 slides)

### Slide 1 - Title
- **Key message:** Head motion alone identifies unseen XR users across applications better than the
  published head-plus-controllers model, once the model has been exposed to the corpus's other people.
- **Content:** Title: "Identifying XR users across applications from head motion alone: progress
  report". Student name, advisor Dr. Feng, date 2026-09-16. One line: "Five-step goal complete; paired
  beat of the SOTA settled."
- **Graphic:** none.
- **Speaker notes:** This report covers the five-step goal set on 10 September. All five steps are
  done. The main result is a paired, per-user beat of the published state of the art on its own
  data, its own test users and its own metric. Two things travel with it: a zero-shot result that is
  unresolved, and a set of negative results that I think are as important as the beat.

### Slide 2 - Motivation and research question
- **Key message:** Motion identification in XR is close to solved within one application and
  collapses across applications. That collapse is the gap this work addresses.
- **Content:**
  - XR systems must stream head and hand motion to work, so motion is shared by necessity. That makes
    identification both an authentication tool and a privacy risk [OUT].
  - SOTA (Schach et al. 2026): **83.1%** rank-1 within an application and **18.0%** across
    applications, with N=17 unseen users and chance 5.88% [PLAN].
  - Their own abstract: identification "across different XR applications remains limited" [PLAN].
  - **Research question:** can a head-only model identify people it has never seen, in an
    application other than the one they enrolled in, and how should that risk be measured?
- **Graphic:** none (or a two-bar teaser from C1).
- **Speaker notes:** The SOTA story fits in one line. Within an application, motion identification is
  close to solved. Across applications it drops to about three times chance. That 18.0% is the number
  every comparison in this report is made against. It is a clean cross-application figure, and I
  explain on slide 15 why their 83.1% within-application figure is not used for comparison.

### Slide 3 - The five-step goal: status
- **Key message:** All five steps are done; step 5 is done for the exposed arm, and the zero-shot
  comparison is unresolved.
- **Content:** the table from Part A.2.
- **Graphic:** none (a table).
- **Speaker notes:** Step 2 went further than reading the paper. We re-ran their released evaluation
  code on their released embeddings and reproduced every per-user value in all 35 cells exactly. So
  our comparison is against their actual model, not against a reimplementation. Step 5 is marked done
  only for the exposed arm, on purpose.

### Slide 4 - Why head-only
- **Key message:** Head-only is a deliberate scope decision. It lets the method run on the whole
  device class, including AR glasses, which track the head and have no hand controllers.
- **Content:**
  - Target is all of XR, including AR glasses that have head tracking and no hands; a model that needs
    controller channels cannot run there [CL].
  - Input: head orientation (quaternion) and head position, 7 channels [CL].
  - Every published comparison, including Schach et al., uses head **plus both controllers** [CL,
    PLAN]. Their 18 features per frame are 4 HMD rotation features plus 14 from the two controllers
    [PLAN].
  - Consequence: our results are head-only measured against head-plus-both-controllers. Where we win,
    we win with less sensor input.
- **Graphic:** none.
- **Speaker notes:** Using only the head is a design choice, not a constraint we are working around.
  AR glasses track the head and have no hand controllers, so a head-only model covers the whole
  device class. It also means that when we beat the SOTA, we beat a system that uses both hand
  controllers as well, with 10-second windows against their 15.

### Slide 5 - The dataset: Across-XR
- **Key message:** Across-XR is the only fully crossed cross-application corpus we know of. Every user
  appears in every application, and we use the authors' exact user split.
- **Content:**
  - 49 users x 5 applications: Superhot VR, Half-Life: Alyx, Beat Saber, Synth Riders, Social VR
    [PLAN, OUT]. All 245 (user, application) cells present [OUT].
  - Licence CC BY-NC-SA 4.0 [PLAN]. 90.9 Hz native, positions in centimetres, y-up [CL].
  - Split read directly from the corpus's `split` column: **train 0-22 (23), validation 23-31 (9),
    test 32-48 (17)** [PLAN].
  - Played in one sitting, in a fixed order, 10-15 minutes each plus breaks. Cross-application pairs
    are up to about an hour apart, never on different days [PLAN].
  - Every application is played standing by the population. Per-person median height span across the
    five is 0.068 m, and 2 of 49 people shift by more than 0.20 m [CL].
  - Pretraining corpora: BOXRR-23 (Beat Saber) and who-is-alyx (Half-Life: Alyx). So Beat Saber and
    Alyx are pretraining *activities* with different people, and Superhot VR, Synth Riders and Social
    VR are not [OUT].
- **Graphic:** C-D1 (dataset split diagram, optional).
- **Speaker notes:** The split is read from the file, not approximated. Users 0 to 22 train, 23 to 31
  validate, 32 to 48 test, which is exactly the protocol their paper uses. We also checked that BOXRR-23,
  the largest XR motion corpus, has zero users recorded in two applications. So a fully crossed corpus
  like this one is genuinely scarce.

### Slide 6 - The SOTA and its story
- **Key message:** Schach et al. published the number to beat (18.0%) and also published a proposed
  fix that they themselves marked as not yet legitimate.
- **Content:**
  - Schach, Rack, McMahan, Latoschik 2026, Frontiers in VR, doi:10.3389/frvir.2026.1743491 [PLAN].
  - Preprocessing: resample to 30 FPS, BRV (body-relative velocity) encoding, 15 s windows (450
    frames). Model: Transformer followed by GRU, 480-d embedding, cosine similarity, nearest-reference
    matching [PLAN].
  - Results at N=17 (Table B6 below): within 83.1%, cross **18.0%**, cross ten-minute 30.8% [PLAN].
  - Their BRV encoding removes head position by construction, so **their numbers are not inflated by
    placement or height** [PLAN].
  - Their section 6.2.5 (Frontiers version only): fitting an orthogonal alignment **on the test users**
    lifts cross-application accuracy from 18.0% to 52.3% (single window) and from 30.8% to 94.3% (ten
    minutes). They call this "a diagnostic upper bound, not a deployable, generalizing solution" and
    propose fitting the alignment on training/validation users as future work [PLAN].
  - Their "cross-application" test uses applications seen in training. **A held-out application is
    never tested**, and we test it [PLAN].
- **Table B6:**

  | metric (N=17, chance 5.88%) | within-application | cross-application |
  | --- | --- | --- |
  | single 15 s window | 83.1% | **18.0%** |
  | 10-minute sequence | 100% | 30.8% |
  | top-3, 10-minute sequence | 100% | 56.0% |

- **Graphic:** C2 (their per-user confidence intervals).
- **Speaker notes:** Their paper is framed as a risk assessment, and it contains an invitation.
  Aligning the embedding spaces lifts cross-application accuracy dramatically, but only when the
  alignment is fitted on the test users. They say so and name the honest version as future work. We
  ran the honest version; the answer is on slide 14. Their released model and code let us reproduce
  their numbers exactly. Their architecture itself is not in the public training code, so we score
  their released model rather than retrain it.

### Slide 7 - The metric
- **Key message:** Primary metric: single-window rank-1 at N=17, matched to theirs. The ten-minute
  figure is secondary. Our addition is reporting the distribution over users.
- **Content:**
  - **Primary:** rank-1 nearest-embedding accuracy at **N=17** on test users 32-48. Gallery from
    application A, probe from application B, averaged over the 20 ordered off-diagonal pairs. Always
    quoted with N; chance = 1/17 = 0.0588 [PLAN].
  - **Secondary:** ten-minute sequence accuracy. It saturates within an application (their
    within-application ten-minute figure is 1.0000 for all 17 users), so it separates methods only
    across applications [PLAN].
  - **Our addition:** report the distribution over users, not only the mean. "The model identifies
    users at 18%" and "a user has an 18% chance of being identified" are different claims [PLAN].
  - **Two harnesses, both directions:** D1 = our embeddings scored by their calculator; D2 = their
    embeddings scored by our template harness [PLAN].
  - **Power, set before running:** at N=17 the binomial sd of a rank-1 near 0.18 is 0.093, so a
    single split cannot resolve differences below about 0.09. A beat must be paired across users and
    seeds [PLAN].
- **Graphic:** none.
- **Speaker notes:** I fixed the primary metric before any result existed, and it stays primary even
  though the secondary metric separates us from them more clearly. Promoting whichever metric looks
  better after the fact would amount to moving the goalposts. The distribution argument matters for a
  risk assessment, because risk falls on individuals, not on the mean.

### Slide 8 - Our approach
- **Key message:** A standard, well-understood pipeline: head pose, 10 s windows, a
  static-cue-free encoding, a BiLSTM trained with an angular-margin identity loss, and template
  matching. The contribution is in the protocol and the measurements, not the architecture.
- **Content:**
  - Pipeline (C12): 10 s windows at 20 Hz, stride 5 s; `dyn` encoding; `bilstm` backbone, 128-d
    embedding; `identity_softmax` (AM-Softmax, margin 0.35, scale 30); cosine scoring [OUT].
  - `dyn`: pose relative to the window's mean pose, gravity kept. It removes height, seat and placement
    and is invariant to rigid transforms of the capture frame [OUT].
  - Gallery: a per-user template, the renormalised mean of all enrolment windows from application A.
    Probe: a single 10 s window from application B [OUT].
  - Arms (C5): **zero-shot** (BOXRR-23 + who-is-alyx, 3,072 trained identities, no Across-XR) and
    **C2-lo** (the same plus Across-XR training users 0-22, 3,095 trained identities, 3.87% of
    training windows) [OUT].
  - Composition check: C2-lo 540,107 windows minus Across-XR 20,896 = 519,211, exactly the zero-shot
    training set [PLAN].
- **Graphic:** C12 (pipeline diagram).
- **Speaker notes:** C2-lo trains on Across-XR users 0 to 22 and tests on 32 to 48. That is exactly
  their protocol, since their model also trained on users 0 to 22, so it is matched, not advantaged.
  The difference is that we also pretrain on about three thousand people from other corpora. The
  architecture is not novel, and the paper will not claim it is.

### Slide 9 - Headline: the paired comparison
- **Key message:** On their 17 test users, the exposed head-only system beats their released model
  under both metrics. The zero-shot system is unresolved.
- **Content:** the table below and chart C1.

  | contrast (paired per user, bootstrap over 17 users) | measured | 95% CI | users better | outcome |
  | --- | --- | --- | --- | --- |
  | zero-shot minus theirs, **their** metric | +0.025 (0.206 vs 0.180) | [-0.031, +0.080] | 10/17 | UNRESOLVED |
  | **C2-lo minus theirs, their metric** | **+0.119** (0.299 vs 0.180) | [+0.050, +0.192] | 15/17 | **BEAT** |
  | zero-shot minus theirs, **our** metric | +0.035 (0.234 vs 0.199) | [-0.050, +0.122] | 10/17 | UNRESOLVED |
  | **C2-lo minus theirs, our metric** | **+0.176** (0.375 vs 0.199) | [+0.092, +0.260] | 16/17 | **BEAT** |

  Status: **SETTLED** [PLAN, PAIR, GATE].
- **Graphic:** C1.
- **Speaker notes:** This is the result the paper can carry. Exposure to the corpus's other
  participants, plus about three thousand pretraining identities, beats their released similarity
  model. It holds on their people, under their metric and ours, single-window and ten-minute, with a
  head-only model against head plus both controllers and 10-second windows against 15. Zero-shot is
  unresolved under both metrics: every zero-shot seed sits above their reported mean, and that is all
  we claim. With 17 users we cannot resolve a difference of +0.05.

### Slide 10 - Secondary metric: ten-minute sequence
- **Key message:** Over ten minutes of probe data the same pattern holds more strongly: C2-lo beats
  them resolvably, and zero-shot stays unresolved.
- **Content:**

  | arm | ten-minute, their metric | paired vs theirs (their metric) | ten-minute, our vote | paired vs theirs (our vote) |
  | --- | --- | --- | --- | --- |
  | theirs | 0.308 [0.208, 0.420] | - | 0.288 | - |
  | zero-shot | 0.414 [0.323, 0.509] | +0.106 [-0.039, +0.256] UNRESOLVED | 0.357 [0.257, 0.474] | +0.069 [-0.091, +0.240] UNRESOLVED |
  | **C2-lo** | **0.663** [0.571, 0.753] | **+0.355 [+0.202, +0.499] BEAT** | **0.711** [0.635, 0.785] | **+0.423 [+0.293, +0.546] BEAT** |

  Status: **SETTLED**, secondary metric [PLAN, PAIR].
- **Graphic:** C3.
- **Speaker notes:** This is the secondary metric and is shown next to the primary one, not instead
  of it. It gives the widest separation in the comparison: their interval tops out at 0.420 and ours
  starts at 0.571. The agreement is what matters: both metrics give the same verdict, unresolved for
  zero-shot and a beat for the exposed arm, and they could have disagreed.

### Slide 11 - Per-cell results
- **Key message:** Among the 20 ordered application pairs, C2-lo beats theirs resolvably in 11 and
  loses none. Zero-shot resolves in only 3 of the 20 (and loses none) - which is why its overall
  contrast stays UNRESOLVED. Do not quote the "3" as zero-shot beating SOTA.
- **Content:**
  - Chart C4a: 5x5 heatmaps under their metric (theirs, zero-shot, C2-lo).
  - Chart C4b: the paired per-cell differences with intervals.
  - The two largest gains are the rhythm-game pair (Beat Saber and Synth Riders, both directions). That
    affinity is already present in the zero-shot arm, which never saw Across-XR, so it belongs to the
    activity pair and is not created by exposure [OUT].
  - Status: **SETTLED** [PLAN, PAIR].
- **Graphic:** C4a, C4b.
- **Speaker notes:** The 20 cells share the same 17 users, so they are not 20 independent samples. They
  are better read as 20 quasi-replications inside one corpus. Beat Saber is also our main pretraining
  activity, so cells involving it are the ones where our pretraining is most relevant.

### Slide 12 - What makes the difference: exposure, identity count, and an unseen application
- **Key message:** Exposure is the lever. Identity count helps only when combined with exposure, and
  exposure carries partially to an application held out of training.
- **Content:**
  - Arm table (C5): zero-shot 0.234; Z-676 0.218; C2-hi 0.307; C2-lo 0.375; C1 (their 23 users only)
    0.131, C1-full 0.164 [CL, OUT].
  - Without exposure, identity count is flat: Z-676 minus zero-shot -0.013 [-0.039, +0.013] [OUT].
  - With exposure it is not: C2-hi minus C2-lo -0.061 [-0.099, -0.026]. Dose cannot explain this:
    halving in-domain windows costs -0.028 [-0.062, +0.009], and the higher-dose arm is the one that
    loses [PLAN, OUT].
  - Our model trained on their 23 users alone (C1-full, 0.164) sits below 0.180. The gain comes from
    pretraining combined with exposure, not from our model on their data [OUT].
  - **Unseen application (P3), C6:** holding one application out of training still carries +0.049
    [+0.021, +0.078] pooled. Applications absent from all pretraining carry +0.046 [+0.017, +0.074]
    against +0.055 for covered ones. Synth Riders, which is in no pretraining corpus, carries (seed
    average +0.065 [+0.034, +0.095]). The stricter registered threshold (CI lower bound above +0.030)
    is **not met** [PLAN, P3S, OUT].
  - The registered prediction P3 (unseen-application cell below seen-application cell) **held**:
    -0.036 [-0.054, -0.018] [PLAN].
  - Status: **SETTLED**. Several arms are single-seed; see C5.
- **Graphic:** C5, C6.
- **Speaker notes:** This is the first data-side lever we have measured to cross an activity boundary.
  Identity count alone was flat across one, and activity diversity was null (next slide). The
  held-out-application cell is the deployment-realistic one, and the SOTA never tested it. I report
  that the stricter threshold was not met, even though the carry itself excludes zero.

### Slide 13 - Static-cue audit of our own result (P2)
- **Key message:** Keeping absolute head pose adds +0.117 cross-application, and a model reaches that
  one epoch after initialisation. So behaviour-only risk assessments understate the risk.
- **Content:**
  - `raw` zero-shot cross-application rank-1 **0.351** (seeds 0.364 / 0.353 / 0.335) against `dyn`
    0.234: **+0.117 [+0.042, +0.192]** [PLAN, OUT].
  - Within-application: `raw` 0.723 against `dyn` 0.500, +0.223 [+0.184, +0.263]. This cell carries
    placement and is never quoted as a biometric [OUT, RES].
  - **Every `raw` seed selected epoch 1 of 16.** The cue sits on the surface of the input; the model
    does not need to learn it [OUT, CL].
  - Across applications, `raw` reads height (P(within<between) 0.754), not placement (0.527), **across
    applications that share a posture** (see slide 17) [CL, PLAN].
  - With exposure the static advantage is unresolved: `raw` C2-lo 0.404 vs `dyn` 0.375, +0.029
    [-0.068, +0.134], 1 seed. Over ten minutes the behavioural model leads: `dyn` 0.711 vs `raw`
    0.497 [OUT].
  - The headline stays on `dyn`, and that was decided before any `raw` number existed. BRV discards
    head position by design, so a `raw` comparison would win partly on a cue their method excludes on
    purpose [CL, OUT].
  - Status: **SETTLED** (3 gated `raw` seeds) [PLAN].
- **Graphic:** C7.
- **Speaker notes:** We ran on our own headline the same audit that this project has run on other
  people's numbers. Static anthropometry and posture, available after one epoch with no behaviour
  needed, get a head-only model to 0.351 across applications. That is close to the 0.375 of our
  trained, exposed behavioural model. This does not contradict Schach et al.: their encoding
  deliberately measures behavioural risk only, and this shows that a behaviour-only assessment
  understates the total. Their numbers are not inflated by this cue.

### Slide 14 - What did not work
- **Key message:** Several registered ideas failed, and each failure is reported with its mechanism.
  These negatives are part of the contribution.
- **Content:**
  - **The SOTA's future-work alignment is closed on `dyn`.** Train-user-only fit, zero-shot: A2 minus A1
    +0.011 [-0.020, +0.041], against a registered band of +0.05 to +0.20. The test-fitted ceiling on our
    embedding is +0.026 [+0.000, +0.051], against their +0.34, which is 13x smaller [OUT, CL]. On C2-lo,
    A2 minus A1 is -0.003 [-0.011, +0.005] [OUT]. Mechanism: the corpus caps correspondences at **32**
    people recorded in two or more applications [PLAN]. The test-fitted ceiling is itself
    run-dependent: C2-lo seeds read +0.148 / -0.004 / +0.001 at identical configuration [OUT].
    Scope: on `raw`, seed 1 reads +0.032 [+0.007, +0.057], so this is a `dyn` result [PLAN].
  - **Activity diversity does not transfer (earlier programme).** Swapping 50 BOXRR identities for 50
    Nymeria identities (daily life on AR glasses) at a fixed 419 identities: -0.0012, 95% CI [-0.0045,
    +0.0020], against a registered band of +0.005 to +0.03. The whole band sits above the whole
    interval [CL].
  - **Identity count is flat across a domain boundary without exposure** (earlier programme, C10) [CL].
  - **A registered mechanism failed.** Template averaging was predicted to lift their embedding into
    0.20-0.32 under our metric. It lifted it by only +0.019 (0.180 to 0.199), missing the band's lower
    edge by 0.001. The averaging gain turns out to be model-specific [PLAN].
  - **A hyperparameter reversed at scale.** Margin/scale 0.1/15 gave +0.016 at 419 identities
    (verification AUC) and costs -0.028 [-0.045, -0.012] in rank-1 at the larger scale. Only the sign
    reversal is claimed, because the two metrics differ [OUT].
  - Status: all **SETTLED**.
- **Graphic:** C13 (alignment); activity-diversity table inline.
- **Speaker notes:** Their paper names train-user-only alignment as the way forward. We ran it and it
  does not carry on our static-free encoding. The reason is concrete: the corpus has only 32 people in
  two or more applications to fit on. That is an actionable specification for a future corpus, not
  just a null result. The template-averaging miss is 0.001 from its band edge. I don't argue it away,
  just as I don't count a 0.001 margin in our favour as a win.

### Slide 15 - A note on their within-application figure
- **Key message:** Their 83.1% within-application figure and their 18.0% cross-application figure are
  not on the same footing. That is a structural property of the corpus. Their 18.0% is clean, and it
  is the figure we compare against.
- **Content:**
  - Each (user, application) cell in Across-XR is one unbroken recording, with no second take [PLAN, CL].
  - So in their within-application evaluation, the reference windows are a subset of the query windows
    from the same recording. Every within-application query shares frames with some reference window:
    0.67% are the identical vector, and 59% share at least half their frames [PLAN].
  - **Their cross-application 18.0% is unaffected**, and every comparison in this report is made
    against it [PLAN].
  - Consequence for us: we never pair our within-application figure with their 83.1%. The drop from
    83.1 to 18.0 overstates the cross-application collapse by whatever the overlap is worth [PLAN].
  - Status: **SETTLED** (read from their released evaluation code) [PLAN].
- **Graphic:** none.
- **Speaker notes:** I want to frame this carefully, because it is not an error on their part. Their
  design has no way to avoid it: the corpus holds only one recording per user and application, so the
  gallery and probe must come from the same recording. It does not affect their cross-application
  number or our comparison. The paper will state it once, structurally.

### Slide 16 - The per-user distribution
- **Key message:** The population mean hides individuals who are identified far more often than
  average. Separately, and only as an exploratory observation, their model and ours find different
  people easy.
- **Content:**
  - Their own per-user cross-application rank-1 ranges from **0.068 to 0.371** on a single window and
    **0.043 to 0.818** at ten minutes, on the same 17 people. One person is identified four-fifths of
    the time behind a population figure of 0.31 [PLAN].
  - Per-user spread (sd): theirs 0.090, zero-shot 0.069, C2-lo 0.110; ranges 0.068-0.371,
    0.105-0.363, 0.148-0.528 [PLAN].
  - **EXPLORATORY, not registered, not claimed:** the two systems' per-user rankings are **not strongly
    correlated**: Spearman theirs vs C2-lo -0.010, 95% CI [-0.49, +0.47]. Our own two arms agree at
    +0.767 [+0.45, +0.91] [PLAN].
  - Example: user 32 is their worst (0.068) and among our best (0.295 zero-shot, 0.487 C2-lo) [PLAN].
  - The cause cannot be attributed. The systems differ in sensor set, architecture and encoding at the
    same time [PLAN].
  - What survives, as ordering claims only: a per-person risk audit under one system need not transfer
    to another, and a defence that protects the top-k most identifiable users has no stable target
    [PLAN].
- **Graphic:** C11 (the existing figure `docs/acceptance/schach_per_user.png`).
- **Speaker notes:** The first point is on their own published numbers, which makes it strong: a risk
  assessment reported as a mean hides the most exposed person. The second point is a hypothesis. With
  17 users the correlation interval is about plus or minus 0.5, so I can say "not strongly
  correlated" and nothing stronger. The cause is confounded. It needs a registered replication on a
  second corpus, and Questset now makes that possible.

### Slide 17 - Questset and the posture finding
- **Key message:** A second cross-application corpus now exists and gives a second gallery size.
  Measuring it showed that head height survives an application change only across applications that
  share a posture.
- **Content:**
  - Questset (Padova, MMSys '24), CC BY 4.0, 60 complete users (70 recruited, 10 withdrew because of
    cybersickness). Four commercial titles, two per user: group 1 Beat Saber + Cooking Simulator,
    group 2 Medal of Honor: Above and Beyond + Forklift Simulator [CL, CAT].
  - Not fully crossed (each user plays 2 of 4 titles), one sitting per user. Its main value is a
    second gallery size: N=30 as well as N=17 [PLAN, CL].
  - **Posture finding (bounded, one item):** head height survives an application change **only across
    applications that share a posture**. Table C8: group 1 (both standing) height P = 0.718; group 2
    (standing vs seated) height P = 0.493, at chance, with all 30 people moving more than 0.20 m
    [QG, CL].
  - The training-free height lookup agrees: group 2 is at chance at both gallery sizes (N=17 0.059;
    N=30 0.033) [QL, CL].
  - Lateral placement is at chance in both groups (0.526, 0.549), which replicates Across-XR's 0.527
    on a second corpus [CL].
  - Status: **SETTLED** (certificates committed), single-corpus.
- **Graphic:** C8 (a two-row table/bar) and C9 (lookup table). Per the recorded decision, the posture
  result gets **no figure of its own in the paper**. In this progress report C8 and C9 are summary
  tables.
- **Speaker notes:** Questset was acquired today. The posture result came from a prediction registered
  before the run, and its falsifier fired. It matters because it adds a qualifier to a claim we already
  make: height survives a change of application only when both applications share a posture. It is
  one corpus, so it stays a bounded note inside the static-cue audit, not a headline. Note that
  Questset's existing identification paper measures a different quantity: its test users are seen in
  training.

### Slide 18 - Methodology and rigour
- **Key message:** The predictions were registered before the runs, every checkpoint is gated, and
  negatives are reported as prominently as positives.
- **Content:**
  - Registered predictions with falsifiers, dated before the runs: P1 (head-only `dyn` in band 0.18 to
    0.35, held at 0.234), P2 (`raw` minus `dyn` positive, held at +0.117), P3 (unseen application below
    seen, held at -0.036) [PLAN]. Table C14.
  - Amendments are kept in place and never edited away [OUT].
  - Every checkpoint is gated against its own recorded score before use: 23 gates passed, with gaps
    from 5.3e-8 to 2.9e-4 [OUT].
  - The SOTA reproduction is bit-exact: 463,996 embeddings accounted for across 85 (user, application)
    cells, and 35 cells reproduced at max abs diff 0.0 [PLAN, GATE].
  - The label-to-user mapping was reconstructed, not assumed. The 17 per-user window-count vectors are
    all distinct [PLAN, GATE].
  - Registered outcomes that went against us are reported: zero-shot unresolved, P3's stricter
    threshold not met, the template-averaging mechanism failed, the ensemble corollary failed [PLAN].
  - Bands are read against confidence intervals, not p-values [OUT].
- **Graphic:** C14 (registered-predictions table).
- **Speaker notes:** Most "we beat X" claims rest on a reimplementation. This one rests on their
  released model, reproduced exactly. Where a registered prediction failed, it is written down as a
  failure. The one post hoc choice was recommending C2-lo as the headline; that was made after P1 was
  registered, and the paper will say so. The C2-lo contrast itself was registered before it ran.

### Slide 19 - Limitations
- **Key message:** The main limits are 17 test users, one fully crossed corpus, and no evidence
  across days.
- **Content:** the list in Part E, in short form:
  - 17 test users, so intervals are wide.
  - One fully crossed corpus; Questset is only partly crossed.
  - No temporal persistence: every user is one sitting, in both corpora.
  - The architecture is not novel.
  - Their within-application figure contains self-matches, so it is never paired with ours.
  - The per-user rank-disagreement finding is exploratory.
  - Several arms are single-seed.
- **Graphic:** none.
- **Speaker notes:** The temporal point comes first because a reviewer will raise it first. Nothing in
  either corpus separates two sessions by a day. Cross-application pairs in Across-XR are at most
  about an hour apart. The 17-user limit has no fix inside this corpus. We mitigate it with intervals
  everywhere, per-user and per-cell results, and Questset's second gallery size.

### Slide 20 - Next steps and decisions needed
- **Key message:** The recommendation is to write the paper now. Venue and the expected level of
  novelty are Dr. Feng's decisions.
- **Content:** Part F in short form: work running now; recommendation and framing; the two decisions;
  the recorded posture decision.
- **Graphic:** none.
- **Speaker notes:** The result set is complete and coherent. Questset strengthens it but does not
  need to delay it. The Questset predictions will be registered before those runs, so if they finish
  before submission they are a real test. The two questions I need your judgement on are the venue and
  how much novelty a first paper needs, given that the architecture is standard.

---

# Part C - Chart and graphic specifications

Chance at N=17 is 1/17 = **0.0588**. Chance at N=30 is 1/30 = **0.0333**. Every chart that shows
rank-1 at N=17 carries a dashed chance line at 0.0588.

## C1 - Headline paired comparison
- **Type:** grouped horizontal bar (or dot-and-whisker) chart with 95% CI whiskers, two panels side
  by side.
- **Title:** "Cross-application rank-1 at N=17, single window: their released model vs ours".
- **Axes:** x = rank-1 accuracy (0 to 0.5, unitless proportion); y = system. Panel 1 = "their metric
  (nearest reference window)"; panel 2 = "our metric (mean template)".
- **Data (levels; bootstrap over the 17 users; seeds averaged within user) [PAIR]:**

  | panel | system | sensors | rank-1 | 95% CI low | 95% CI high |
  | --- | --- | --- | --- | --- | --- |
  | their metric | Schach et al. released model | head + both controllers | 0.1804 | 0.1402 | 0.2251 |
  | their metric | ours, zero-shot (3 seeds) | head only | 0.2059 | 0.1745 | 0.2397 |
  | their metric | ours, C2-lo (3 seeds) | head only | 0.2993 | 0.2494 | 0.3538 |
  | our metric | Schach et al. released model | head + both controllers | 0.1988 | 0.1460 | 0.2579 |
  | our metric | ours, zero-shot (3 seeds) | head only | 0.2336 | 0.1817 | 0.2915 |
  | our metric | ours, C2-lo (3 seeds) | head only | 0.3745 | 0.3214 | 0.4333 |

- **Paired differences to annotate beside the bars [PLAN, PAIR]:**

  | panel | contrast | difference | 95% CI | users better | outcome |
  | --- | --- | --- | --- | --- | --- |
  | their metric | zero-shot minus theirs | +0.025 | [-0.031, +0.080] | 10/17 | UNRESOLVED |
  | their metric | C2-lo minus theirs | +0.119 | [+0.050, +0.192] | 15/17 | BEAT |
  | our metric | zero-shot minus theirs | +0.035 | [-0.050, +0.122] | 10/17 | UNRESOLVED |
  | our metric | C2-lo minus theirs | +0.176 | [+0.092, +0.260] | 16/17 | BEAT |

- **Emphasise:** the C2-lo bars and the word BEAT. Show zero-shot in a neutral colour, labelled
  UNRESOLVED.
- **Reference lines:** chance 0.0588 (dashed).
- **Caption to print:** "Paired per user on Schach et al.'s 17 test users (32-48); bootstrap over
  users. Head-only (ours) vs head plus both controllers (theirs); 10 s vs 15 s windows. Zero-shot is
  unresolved, not a beat. Status: SETTLED."
- **Note for the graphics author:** every interval in this report now comes from one source, the
  settled paired certificate `schach_paired.json`, so the same quantity never shows two different
  intervals on two slides. Some project notes print slightly different edges (e.g. zero-shot
  [0.181, 0.292]) - those are separate bootstrap runs of the same quantity and differ only in the
  third decimal. Use the values in this report; do not mix in others.

## C2 - Schach et al.'s own per-user confidence intervals
- **Type:** dot-and-whisker, 2x2 layout (rows: within / cross; columns: single window / ten minutes).
- **Title:** "Schach et al., per-user 95% CIs from their released values (N=17)".
- **Axes:** x = rank-1 accuracy (0 to 1); y = condition.
- **Data (per-user means over cells, cluster bootstrap over 17 users, 10,000 resamples) [PLAN]:**

  | condition | metric | mean | 95% CI low | 95% CI high |
  | --- | --- | --- | --- | --- |
  | cross-application (20 off-diagonal cells) | single 15 s window | 0.1804 | 0.1402 | 0.2251 |
  | cross-application | 10-minute sequence | 0.3082 | 0.2080 | 0.4199 |
  | within-application (5 diagonal cells) | single 15 s window | 0.8314 | 0.8102 | 0.8520 |
  | within-application | 10-minute sequence | 1.0000 | 1.0000 | 1.0000 |

- **Per-user ranges, to annotate [PLAN]:** cross single window 0.068 to 0.371; cross ten-minute
  0.043 to 0.818.
- **Emphasise:** the cross-application single-window row (the comparison target). The within
  ten-minute row is saturated at 1.0000 for all 17 users.
- **Reference lines:** chance 0.0588.
- **Caption:** "All four point estimates reproduce the published figures exactly. Within-application
  values include self-matches by design (one recording per user and application) and are not used for
  comparison. Status: SETTLED."

## C3 - Ten-minute secondary comparison
- **Type:** grouped bar with CI whiskers, two groups (their sequence metric / our vote).
- **Title:** "Secondary metric: ten-minute sequence, cross-application, N=17".
- **Axes:** x = system; y = accuracy (0 to 1).
- **Data [PLAN, PAIR]:**

  | metric | system | accuracy | 95% CI low | 95% CI high |
  | --- | --- | --- | --- | --- |
  | their sequence metric | Schach et al. | 0.308 | 0.208 | 0.420 |
  | their sequence metric | ours, zero-shot | 0.414 | 0.323 | 0.509 |
  | their sequence metric | ours, C2-lo | 0.663 | 0.571 | 0.753 |
  | our vote | Schach et al. | 0.2882 | 0.1912 | 0.3941 |
  | our vote | ours, zero-shot | 0.357 | 0.257 | 0.474 |
  | our vote | ours, C2-lo | 0.711 | 0.635 | 0.785 |

  Paired differences: zero-shot +0.106 [-0.039, +0.256] (their metric) and +0.069 [-0.091, +0.240]
  (our vote), both UNRESOLVED; C2-lo +0.355 [+0.202, +0.499] and +0.423 [+0.293, +0.546], both BEAT.
- **Emphasise:** C2-lo's interval lies entirely above theirs.
- **Reference lines:** chance 0.0588.
- **Caption:** "SECONDARY metric, shown beside the single-window primary (C1), never instead of it.
  Their within-application ten-minute figure is 1.0000, so this metric separates methods only across
  applications. Status: SETTLED."

## C4a - Per-cell heatmaps (their metric)
- **Type:** three 5x5 heatmaps side by side, same colour scale (0 to 0.9). Rows = gallery
  (reference) application; columns = probe (query) application.
- **Title:** "Cross-application rank-1 per application pair, their metric, N=17".
- **Application order:** 1 Superhot VR, 2 Half-Life: Alyx, 3 Beat Saber, 4 Synth Riders, 5 Social VR
  [OUT].
- **Data, Schach et al. released model (their per-cell means, reproduced by the gate) [GATE]:**

  | gallery \ probe | Superhot | Alyx | Beat Saber | Synth Riders | Social VR |
  | --- | --- | --- | --- | --- | --- |
  | Superhot | *0.8069* | 0.2222 | 0.1975 | 0.1822 | 0.1977 |
  | Alyx | 0.2262 | *0.7234* | 0.1327 | 0.1518 | 0.1819 |
  | Beat Saber | 0.1995 | 0.1840 | *0.8799* | 0.2086 | 0.1889 |
  | Synth Riders | 0.1893 | 0.1540 | 0.2259 | *0.8667* | 0.1728 |
  | Social VR | 0.1342 | 0.1777 | 0.1051 | 0.1765 | *0.8800* |

  Diagonal (italic) = within-application. It includes self-matches, so hatch or grey it out.
  Off-diagonal mean = 0.1804; diagonal mean = 0.8314.
- **Data, ours zero-shot (3 seeds), their metric, off-diagonal only [PAIR]:**

  | gallery \ probe | Superhot | Alyx | Beat Saber | Synth Riders | Social VR |
  | --- | --- | --- | --- | --- | --- |
  | Superhot | - | 0.2126 | 0.2208 | 0.1808 | 0.1992 |
  | Alyx | 0.1862 | - | 0.1725 | 0.1760 | 0.1761 |
  | Beat Saber | 0.2060 | 0.1878 | - | 0.3870 | 0.1799 |
  | Synth Riders | 0.1859 | 0.1666 | 0.3809 | - | 0.2016 |
  | Social VR | 0.1572 | 0.1689 | 0.1829 | 0.1894 | - |

- **Data, ours C2-lo (3 seeds), their metric, off-diagonal only [PAIR]:**

  | gallery \ probe | Superhot | Alyx | Beat Saber | Synth Riders | Social VR |
  | --- | --- | --- | --- | --- | --- |
  | Superhot | - | 0.2667 | 0.3317 | 0.2777 | 0.2601 |
  | Alyx | 0.2839 | - | 0.3066 | 0.2883 | 0.2599 |
  | Beat Saber | 0.3053 | 0.2698 | - | 0.4821 | 0.3236 |
  | Synth Riders | 0.2740 | 0.2381 | 0.5013 | - | 0.2966 |
  | Social VR | 0.2282 | 0.2429 | 0.2925 | 0.2569 | - |

- **Emphasise:** the rhythm-game pair (Beat Saber and Synth Riders) is the brightest off-diagonal pair
  in both of our arms.
- **Caption:** "Values are from the certificates, rounded to four decimals. Beat Saber is our main
  pretraining activity (BOXRR-23); Half-Life: Alyx is the other (who-is-alyx). Status: SETTLED."

## C4b - Per-cell paired differences (C2-lo minus theirs, their metric)
- **Type:** forest plot, 20 rows (ordered pairs), point and 95% CI; zero line.
- **Title:** "C2-lo beats their model in 11 of 20 application pairs and loses none".
- **Axes:** x = paired difference in rank-1 (-0.15 to +0.40); y = gallery -> probe pair.
- **Data (rounded to three decimals from [PAIR]); "resolved" = interval excludes zero:**

  | gallery -> probe | C2-lo minus theirs | 95% CI | resolved | zero-shot minus theirs | 95% CI | resolved |
  | --- | --- | --- | --- | --- | --- | --- |
  | Superhot -> Alyx | +0.044 | [-0.040, +0.125] | no | -0.010 | [-0.092, +0.072] | no |
  | Superhot -> Beat Saber | +0.134 | [+0.026, +0.255] | **yes** | +0.023 | [-0.058, +0.114] | no |
  | Superhot -> Synth Riders | +0.095 | [-0.029, +0.228] | no | -0.001 | [-0.102, +0.105] | no |
  | Superhot -> Social VR | +0.062 | [-0.037, +0.153] | no | +0.002 | [-0.084, +0.081] | no |
  | Alyx -> Superhot | +0.058 | [-0.034, +0.150] | no | -0.040 | [-0.113, +0.031] | no |
  | Alyx -> Beat Saber | +0.174 | [+0.060, +0.280] | **yes** | +0.040 | [-0.045, +0.121] | no |
  | Alyx -> Synth Riders | +0.136 | [-0.002, +0.266] | no | +0.024 | [-0.096, +0.134] | no |
  | Alyx -> Social VR | +0.078 | [-0.012, +0.169] | no | -0.006 | [-0.107, +0.107] | no |
  | Beat Saber -> Superhot | +0.106 | [+0.000, +0.202] (lower bound positive before rounding) | **yes** | +0.006 | [-0.079, +0.078] | no |
  | Beat Saber -> Alyx | +0.086 | [-0.026, +0.188] | no | +0.004 | [-0.106, +0.093] | no |
  | Beat Saber -> Synth Riders | +0.273 | [+0.170, +0.377] | **yes** | +0.178 | [+0.071, +0.287] | **yes** |
  | Beat Saber -> Social VR | +0.135 | [+0.048, +0.223] | **yes** | -0.009 | [-0.103, +0.087] | no |
  | Synth Riders -> Superhot | +0.085 | [-0.012, +0.177] | no | -0.003 | [-0.071, +0.063] | no |
  | Synth Riders -> Alyx | +0.084 | [+0.006, +0.161] | **yes** | +0.013 | [-0.066, +0.084] | no |
  | Synth Riders -> Beat Saber | +0.275 | [+0.162, +0.387] | **yes** | +0.155 | [+0.031, +0.272] | **yes** |
  | Synth Riders -> Social VR | +0.124 | [+0.009, +0.231] | **yes** | +0.029 | [-0.087, +0.150] | no |
  | Social VR -> Superhot | +0.094 | [+0.026, +0.164] | **yes** | +0.023 | [-0.030, +0.076] | no |
  | Social VR -> Alyx | +0.065 | [+0.006, +0.124] | **yes** | -0.009 | [-0.074, +0.065] | no |
  | Social VR -> Beat Saber | +0.187 | [+0.103, +0.278] | **yes** | +0.078 | [+0.015, +0.140] | **yes** |
  | Social VR -> Synth Riders | +0.080 | [-0.028, +0.187] | no | +0.013 | [-0.083, +0.103] | no |

  Totals [PLAN]: C2-lo resolved beats 11 of 20, losses 0; zero-shot resolved beats 3 of 20, losses 0.
- **Caption:** "The 20 cells share the same 17 users and are not independent. Status: SETTLED."

## C5 - Programme arms: exposure and identity count
- **Type:** horizontal bar chart with CI whiskers, one bar per arm, plus a reference line for Schach et
  al.
- **Title:** "Cross-application rank-1 (our metric) by training composition, N=17".
- **Axes:** x = rank-1 (0 to 0.5); y = arm.
- **Data [CL, OUT, PLAN]:**

  | arm | training data | Across-XR exposure | trained identities | dose (share of training windows) | seeds | rank-1 (A1) | 95% CI |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | zero-shot | BOXRR-23 + who-is-alyx | none | 3,072 | 0 | 3 | 0.234 | [0.182, 0.292] |
  | Z-676 | BOXRR 600 + who-is-alyx | none | 495 | 0 | 1 | 0.218 | [0.168, 0.271] |
  | C2-hi | Z-676 minus 23 BOXRR training users, plus Across-XR 0-22 | yes | 495 | 14.1% | 1 | 0.307 | [0.263, 0.354] |
  | **C2-lo** | zero-shot corpus + Across-XR 0-22 | yes | 3,095 | 3.87% | 3 (range 0.010) | **0.375** | [0.321, 0.433] |
  | C1 (their protocol, our model) | Across-XR 0-22 only | yes | 23 | 100% | 1 | 0.131 | [0.088, 0.177] |
  | C1-full (budget-matched C1) | Across-XR 0-22 only | yes | 23 | 100% | 1 | 0.164 | [0.128, 0.205] |

  Contrasts to annotate: C2-lo minus zero-shot +0.141 [+0.100, +0.183] (3 paired seeds); Z-676 minus
  zero-shot -0.013 [-0.039, +0.013]; C2-hi minus C2-lo -0.061 [-0.099, -0.026]; C2-hi minus Z-676
  +0.089 [+0.048, +0.131]; halving C2-lo's in-domain windows -0.028 [-0.062, +0.009] (dose 1.96%)
  [OUT, PLAN].
- **Reference lines:** Schach et al. 0.180 (solid, labelled "published, head + controllers"); chance
  0.0588 (dashed).
- **Emphasise:** the two exposed arms against their unexposed counterparts (C2-lo vs zero-shot; C2-hi
  vs Z-676).
- **Caption:** "Trained identities are counted after the 25% validation draw (pools of 4,096 and 676).
  C1 was under-trained (patience fired at epoch 15); C1-full is the budget-matched version.
  Single-seed arms are marked. Status: SETTLED."

## C6 - Exposure carries to an unseen application (P3)
- **Type:** forest plot.
- **Title:** "Leave-one-application-out: gain over the unexposed control (P3 minus Z-676)".
- **Axes:** x = paired difference in rank-1 (-0.02 to +0.14); y = held-out application.
- **Data [P3S, OUT]:**

  | held-out application | in pretraining? | seeds | P3 minus Z-676 | 95% CI |
  | --- | --- | --- | --- | --- |
  | Superhot VR | no | 1 | +0.034 | [-0.004, +0.070] |
  | Half-Life: Alyx | yes | 1 | +0.026 | [-0.004, +0.059] |
  | Beat Saber | yes | 1 | +0.084 | [+0.044, +0.122] |
  | Synth Riders | no | 2 (+0.077 / +0.053) | +0.065 | [+0.034, +0.095] |
  | Social VR | no | 2 (+0.044 / +0.032) | +0.038 | [+0.001, +0.073] |
  | **pooled, all five** | - | - | **+0.049** | **[+0.021, +0.078]** |
  | uncovered (Superhot, Synth Riders, Social VR) | no | - | +0.046 | [+0.017, +0.074] |
  | covered (Beat Saber, Alyx) | yes | - | +0.055 | [+0.024, +0.089] |

- **Reference lines:** zero (solid); the registered headline threshold at **+0.030** (dashed,
  labelled "registered: CI lower bound must exceed this - NOT MET").
- **Emphasise:** pooled and uncovered rows; Synth Riders, which is absent from all pretraining.
- **Caption:** "Registered band +0.02 to +0.07 held at the mean; falsifier (<= 0) excluded; stricter
  threshold not met. Related registered prediction P3: unseen application below seen, -0.036 [-0.054,
  -0.018], held. Status: SETTLED."

## C7 - Static-cue audit (P2)
- **Type:** paired bar chart (`dyn` vs `raw`) in three groups, with seed dots on the single-window
  group.
- **Title:** "Keeping absolute head pose: raw vs dyn, zero-shot, N=17".
- **Axes:** x = condition; y = rank-1 (0 to 0.8).
- **Data [PLAN, OUT, RES, PAIR]:**

  | condition | `dyn` (behaviour only) | `raw` (keeps head pose) | raw minus dyn | 95% CI |
  | --- | --- | --- | --- | --- |
  | cross-application, single window (zero-shot, 3 seeds) | 0.234 | 0.351 (seeds 0.364 / 0.353 / 0.335) | **+0.117** | [+0.042, +0.192] |
  | within-application, single window (zero-shot) | 0.500 | 0.723 (seeds 0.730 / 0.732 / 0.707) | +0.223 | [+0.184, +0.263] |
  | cross-application, ten-minute (zero-shot) | 0.357 | 0.434 | +0.077 | not reported |
  | cross-application, single window, **with exposure (C2-lo; raw 1 seed)** | 0.375 | 0.404 | +0.029 | [-0.068, +0.134] (unresolved) |
  | cross-application, ten-minute, with exposure | 0.711 | 0.497 | -0.214 | not reported |

  Annotation: "every `raw` seed selected epoch 1 of 16". Static-geometry context for Across-XR,
  across applications: lateral P(within<between) 0.527, height 0.754; within an application, lateral
  0.7525 [CL].
- **Emphasise:** the first row (the audit result) and the epoch-1 annotation.
- **Reference lines:** chance 0.0588.
- **Caption:** "The headline stays on `dyn`, decided before any `raw` number existed. Schach et al.'s
  BRV encoding discards head position, so their numbers are not affected by this cue. Within-application
  `raw` carries placement and is not a biometric figure. Height survives across applications that
  share a posture. Status: SETTLED."

## C8 - Questset: the static cue splits on posture
- **Type:** a small two-row table rendered as a graphic, or grouped bars (lateral P, height P) for the
  two groups. Per the recorded decision, keep it small.
- **Title:** "Questset: head height survives only across applications that share a posture".
- **Axes (if bars):** x = group; y = P(within < between) (0 to 1).
- **Data [QG, CL]** (30 people per group; 30 within-person pairs, 870 between-person pairs):

  | group | applications | all P | lateral P | height P | median height change | people moving > 0.20 m |
  | --- | --- | --- | --- | --- | --- | --- |
  | 1 | Beat Saber / Cooking Simulator (both standing) | 0.537 | 0.526 | **0.718** | 0.051 m | 0 / 30 |
  | 2 | Medal of Honor / Forklift Simulator (standing vs seated) | 0.552 | 0.549 | **0.493** (chance) | 0.438 m | 30 / 30 |

- **Registered before running [QG]:** height above 0.65 means the band holds; below 0.60 the falsifier
  fires; 0.60-0.65 means "weakened". Group 1 is in the band. Group 2 fired the falsifier.
- **Reference lines:** 0.5 (chance for P).
- **Emphasise:** the height column.
- **Caption:** "Single corpus; a bounded scoping note inside the static-cue audit. For comparison,
  Across-XR (all five applications standing) reads height P 0.754, lateral 0.527. Status: SETTLED."

## C9 - Questset: training-free static lookup
- **Type:** table (or small grouped bars per axis).
- **Title:** "Training-free mean-position lookup, Questset, 10 s windows".
- **Data (rank-1, mean of both directions) [QL, CL]:**

  | group | axis | N=17 (chance 0.0588) | N=30 (chance 0.0333) |
  | --- | --- | --- | --- |
  | 1 (both standing) | y only (height) | **0.153** | **0.092** |
  | 1 | xyz | 0.108 | 0.073 |
  | 1 | xz only (lateral) | 0.054 | 0.034 |
  | 2 (standing / seated) | y only (height) | **0.059** | **0.033** |
  | 2 | xyz | 0.071 | 0.033 |
  | 2 | xz only (lateral) | 0.074 | 0.046 |

- **Emphasise:** group 2 height sits at chance at both gallery sizes.
- **Reference lines:** chance values per column.
- **Caption:** "Gallery = per-user mean position over game A; probe = individual game B windows; both
  directions; per-axis standardised. The largest training-free static figure anywhere in Questset is
  0.153 at N=17. Status: SETTLED. Registered prediction HALF HELD: group 2 height at chance as
  predicted (0.033, registered < 0.10); group 1 height above chance but BELOW its registered level
  (0.092, registered > 0.15 at N=30)."
- **Say it this way, and only this way:** "group 2 at chance as predicted; group 1 above chance but
  below its registered level." Never "the prediction held". The groups still separate on height
  (group 1 at 2.8x chance, group 2 exactly at chance), which is what the posture reading rests on.

## C10 - Identity-count curve under `dyn` (earlier programme, a different metric)
- **Type:** line chart with markers and error bars, two series (5 s and 10 s windows).
- **Title:** "Training identities vs transfer to seven held-out corpora (`dyn`)".
- **Axes:** x = training identities (log scale: 419, 1000, 2096, 4096); y = pooled verification AUC
  (0.55 to 0.65; chance 0.50).
- **Data [CL]:**

  | window | identities | pooled transfer AUC | spread | seeds |
  | --- | --- | --- | --- | --- |
  | 5 s | 419 | 0.582 | +-0.001 | 5 |
  | 5 s | 1000 | 0.600 | +-0.001 | 2 |
  | 5 s | 2096 | 0.598 | - | 1 |
  | 10 s | 419 | 0.600 | - | 5 |
  | 10 s | 2096 | 0.6179 | +-0.0005 | 2 |
  | 10 s | 4096 | 0.6156 | +-0.0040 | 2 |

  Context: under `raw`, transfer is flat, 0.672 at 419 and 0.671 at 2096 [CL]. In domain, at 10 s, the
  same checkpoints read BOXRR 0.845 -> 0.962 -> 0.970 (419 / 2096 / 4096) [CL].
- **Emphasise:** the curve flattens after the first step. The second doubling (2096 to 4096) moves
  transfer by nothing.
- **Caption:** "Verification AUC (chance 0.50) on seven held-out corpora, not rank-1 on Across-XR. Do
  not place it beside a rank-1 figure. Identity count pays within the training activity and very little
  across one. Status: SETTLED (earlier programme, recorded runs)."

## C11 - Per-user figure (existing file)
- **File:** `docs/acceptance/schach_per_user.png` (also `.svg`). It already exists and should be used
  as-is.
- **What it shows:** two panels (left: their nearest-reference metric; right: our template metric).
  One row per test user 32-48, in the same order in both panels, sorted by their model's value under
  their metric. Each row has three dots: their released model (blue), our zero-shot (orange), our
  C2-lo (green). Dashed verticals mark population means, and a line marks chance at 1/17.
- **Data behind it [CSV]:**

  | user | theirs, their metric | zero-shot, their metric | C2-lo, their metric | theirs, our metric | zero-shot, our metric | C2-lo, our metric |
  | --- | --- | --- | --- | --- | --- | --- |
  | 40 | 0.3706 | 0.1968 | 0.4065 | 0.4202 | 0.1998 | 0.5092 |
  | 38 | 0.3312 | 0.2787 | 0.3081 | 0.3282 | 0.3649 | 0.3723 |
  | 35 | 0.3297 | 0.1742 | 0.1478 | 0.4737 | 0.2056 | 0.2137 |
  | 47 | 0.2592 | 0.1748 | 0.2674 | 0.2445 | 0.1840 | 0.3620 |
  | 44 | 0.2081 | 0.1538 | 0.3250 | 0.2629 | 0.1525 | 0.4426 |
  | 33 | 0.2068 | 0.2069 | 0.2918 | 0.2682 | 0.1858 | 0.3995 |
  | 46 | 0.1732 | 0.2462 | 0.2492 | 0.1289 | 0.2130 | 0.2716 |
  | 45 | 0.1719 | 0.1083 | 0.1753 | 0.1587 | 0.0762 | 0.2206 |
  | 48 | 0.1609 | 0.1048 | 0.1836 | 0.2672 | 0.1087 | 0.3109 |
  | 41 | 0.1482 | 0.3086 | 0.4750 | 0.1550 | 0.4186 | 0.5241 |
  | 34 | 0.1325 | 0.2100 | 0.2958 | 0.0746 | 0.2706 | 0.3815 |
  | 42 | 0.1236 | 0.3634 | 0.5283 | 0.1125 | 0.4916 | 0.6809 |
  | 39 | 0.1099 | 0.1815 | 0.2775 | 0.1100 | 0.1792 | 0.3816 |
  | 36 | 0.0978 | 0.1671 | 0.2069 | 0.1203 | 0.1558 | 0.2810 |
  | 43 | 0.0920 | 0.1772 | 0.2138 | 0.1010 | 0.1847 | 0.2918 |
  | 37 | 0.0842 | 0.1532 | 0.2488 | 0.0917 | 0.1565 | 0.2703 |
  | 32 | 0.0677 | 0.2949 | 0.4873 | 0.0614 | 0.4243 | 0.4535 |

- **Rank-agreement numbers for an annotation box (EXPLORATORY) [PLAN]:**

  | pair | Spearman rho | 95% CI (Fisher) | permutation p |
  | --- | --- | --- | --- |
  | theirs vs our C2-lo (their metric) | -0.010 | [-0.49, +0.47] | 0.97 |
  | theirs vs our zero-shot (their metric) | -0.037 | [-0.51, +0.45] | 0.89 |
  | control: our zero-shot, their metric vs ours | +0.939 | - | 0.000 |
  | control: our C2-lo, their metric vs ours | +0.926 | - | 0.000 |
  | control: our two arms vs each other | +0.767 | [+0.45, +0.91] | 0.001 |

- **Caption:** "EXPLORATORY and not registered. The systems' per-user orderings are not strongly
  correlated (n=17, CI about +-0.5). The cause cannot be attributed: sensor set, architecture and
  encoding all differ. Do not add an ensemble or coverage claim; that corollary was tested and failed
  (C2-lo already catches all but 2 of 17 users better than their model)."

## C12 - Method pipeline diagram (to draw)
- **Type:** left-to-right flow diagram, six boxes and a matching stage.
- **Boxes, in order:**
  1. **Head-tracking data**: head orientation (quaternion x, y, z, w) and head position (x, y, z),
     7 channels. No controllers, by design.
  2. **Windows**: 10 s at 20 Hz (200 frames), a new window every 5 s.
  3. **Encoding `dyn`**: each window re-expressed relative to its own mean pose, gravity kept. This
     removes height, seat and placement.
  4. **Model**: BiLSTM backbone, trained with an angular-margin identity loss (AM-Softmax) on about
     3,000 pretraining identities (BOXRR-23 Beat Saber, who-is-alyx Half-Life: Alyx), plus Across-XR
     users 0-22 for the exposed arm.
  5. **Embedding**: 128-dimensional vector per window, compared by cosine similarity.
  6. **Matching**: gallery = one template per enrolled user, the mean of all their windows in
     application A. Probe = one 10 s window from application B. The prediction is the nearest template
     among N=17 users, scored as rank-1.
- **Side note on the diagram:** "Schach et al.: head + both controllers, BRV encoding, 15 s at 30 FPS,
  Transformer + GRU, 480-d embedding, nearest reference window."
- **Sources:** [OUT] section 4.4-4.5, [PLAN], [CL].

## C13 - Alignment route closed
- **Type:** dot-and-whisker plot with reference lines.
- **Title:** "Orthogonal alignment of application embeddings: honest fit vs test-fitted ceiling".
- **Axes:** x = gain in cross-application rank-1 over the unaligned model (-0.15 to +0.40); y =
  variant.
- **Data [OUT, PLAN]:**

  | variant | arm | gain | 95% CI | registered expectation |
  | --- | --- | --- | --- | --- |
  | A2' minus A1 (fit on test users; diagnostic ceiling) | zero-shot, 3 seeds | +0.026 | [+0.000, +0.051] | >= +0.15 |
  | A2 minus A1 (fit on users 0-31; honest) | zero-shot, 3 seeds | +0.011 | [-0.020, +0.041] | band +0.05 to +0.20 |
  | A2-null minus A1 (permuted correspondences) | zero-shot | -0.074 | [-0.126, -0.029] | - |
  | A2-full minus A2 (unrestricted 128-d fit) | zero-shot | -0.055 | [-0.086, -0.029] | - |
  | A2 minus A1 (honest) | C2-lo, 3 seeds | -0.003 | [-0.011, +0.005] | - |
  | A2' minus A1, per seed | C2-lo | +0.148 / -0.004 / +0.001 | per seed | - |
  | A2 minus A1 (honest), `raw` | zero-shot `raw`, seed 1 | +0.032 | [+0.007, +0.057] | - |
  | Schach et al. test-fitted ceiling (their model) | published | +0.34 (18.0% -> 52.3%) | - | - |

- **Reference lines:** zero; +0.34 labelled "Schach et al., fitted on test users".
- **Emphasise:** the honest fit on `dyn` sits at zero. The correspondence cap is 32 people.
- **Caption:** "The honest fit never resolvably carries on `dyn` (the headline encoding). On `raw`,
  seed 1 is resolvably positive, which fits a static frame that a rotation can act on. The test-fitted
  ceiling is run-dependent at identical configuration, so it was never a target. Status: SETTLED."

## C14 - Registered predictions and verdicts
- **Type:** table.
- **Scope (review pass, 2026-09-17):** every band or falsifier registered in
  `docs/acceptance/across_xr_alignment_REGISTERED.md` (original text and Amendments 1-8), in
  `questset_geometry.json`, `questset_static_lookup.json` and `across_xr_within_application.json`
  has a row, including the ones whose outcome went against the prediction. Where a band and its
  falsifier did not meet, the unnamed region is stated in the "registered" column. The paper's
  Table 3 may print the subset that bears on a reported claim, but every row here is a registered
  outcome and no row may be dropped silently - an omitted negative is the failure mode this table
  exists to prevent. Verdicts are read against the interval, not against p < 0.05.
- **Data [PLAN, OUT, RES, P3S, QG, QL, CL]:**

  | prediction | registered (source) | measured | verdict |
  | --- | --- | --- | --- |
  | **P1**: head-only `dyn` cross-application rank-1 at N=17 | band 0.18-0.35; falsifier < 0.12; [0.12, 0.18) and > 0.35 unnamed (original registration) | 0.234 [0.182, 0.292], 3 seeds | HELD, inside the band |
  | **P2**: `raw` minus `dyn`, zero-shot, cross-application (R-zero A1 - dyn A1) | band **+0.00 to +0.06**; falsifier < -0.03; (-0.03, 0.00) and > +0.06 unnamed (Amendment 6) | **+0.117 [+0.042, +0.192]**, 3 seeds, every seed at epoch 1 of a 120-epoch budget (patience 15) | direction HELD and the falsifier excluded by 0.07; **the registered SIZE was under-predicted**: the mean sits at twice the band's upper edge and the interval spans that edge (lower bound inside the band, upper bound above it), so "inside" against "above" is not resolved and is not argued. `p2.json` verdict: "interval spans a registered edge: unresolved" |
  | P2, exposed: `raw` minus `dyn`, C2-lo (R-C2-lo A1 - dyn C2-lo A1) | band +0.00 to +0.06; falsifier < -0.03 (Amendment 6) | +0.029 [-0.068, +0.134], 1 `raw` seed against 3 `dyn` seeds | UNRESOLVED: the interval spans both band edges and zero |
  | P2, within-application: `raw` A0 gain exceeds `raw` A1 gain | directional (Amendment 6): the within cell carries placement (P=0.7525) and should gain more | A0 +0.223 [+0.184, +0.263] against A1 +0.117 | HELD; the within-application `raw` figure is never quoted as a biometric |
  | **P3**: unseen-application cell below seen-application cell (P3 - C2-hi on X-cells) | directional (original registration; Amendment 4) | -0.036 [-0.054, -0.018] | HELD (dose-confounded by 20% less in-domain data; a direction, not a measurement) |
  | Exposure carries to an application held out of training (P3 - Z-676 on X-cells, pooled) | band +0.02 to +0.07 at the mean; falsifier <= 0; (0, +0.02] named "unresolved"; headline "crosses an activity boundary" needs CI lower bound > +0.030 (Amendment 4) | +0.049 [+0.021, +0.078], five applications, seed-averaged | inside the band; falsifier excluded; **headline threshold NOT MET** (lower bound 0.021 against 0.030) |
  | P3 control: removing one application leaves the seen cells where they were (P3 non-X - C2-hi non-X within +-0.03) | band +-0.03 (Amendment 4) | -0.009 [-0.025, +0.008] | HELD: 20% less in-domain data cost nothing measurable on seen cells, so the P3 runs are comparable to C2-hi |
  | P3 mechanism: the two rhythm games carry best | ordering prediction, not a band (Amendment 4 iii) | Beat Saber +0.084, Synth Riders +0.065 are the top two of five | HELD |
  | P3 mechanism: Social VR carries least, below +0.03 | ordering prediction (Amendment 4 iii; second addendum) | Social VR +0.038 [+0.001, +0.073], 2 seeds; **Alyx is least at +0.026** | **FAILED**: Social VR carries about as much as the mean; the task-structure ordering holds at the top and not at the bottom |
  | P3: Synth Riders carries at >= +0.04 on its cells | second addendum to Amendment 4, registered with two runs outstanding | +0.065 [+0.034, +0.095], 2 seeds (+0.077 / +0.053) | HELD |
  | P3 embeddings: test-fitted alignment ceiling A2' - A1 below +0.03 | Amendment 4 | +0.003 / +0.039 / +0.135 / +0.136 / +0.131 (Superhot / Alyx / Beat Saber / Synth Riders / Social VR) | **FAILED on three of five**: the ceiling is present on three held-out applications and absent on two at identical configuration - the run-dependence C2-lo's seeds showed, on a second arm |
  | **C2-lo beats theirs, their metric** (paired per user) | point in +0.08 to +0.20; outcome partition BEAT / LOSS / UNRESOLVED by the interval (Amendment 8) | +0.119 [+0.050, +0.192], 15/17 users | **BEAT**, inside the band |
  | **C2-lo beats theirs, our metric** | point in +0.05 to +0.17 (Amendment 8) | +0.176 [+0.092, +0.260], 16/17 users | **BEAT**; 0.006 past the band's upper edge, declared and not argued |
  | Zero-shot vs theirs, their metric | point in -0.05 to +0.06, predicted UNRESOLVED (Amendment 8; power note: MDD 0.081 at N=17) | +0.025 [-0.031, +0.080], 10/17 users | UNRESOLVED, as predicted; not a beat and not parity |
  | Zero-shot vs theirs, our metric | predicted UNRESOLVED (Amendment 8) | +0.035 [-0.050, +0.122], 10/17 users | UNRESOLVED, as predicted |
  | Template averaging lifts their model to 0.20-0.32 under our metric | band (Amendment 8) | 0.199 [0.146, 0.258] | **FAILED** by 0.001 below the edge, not argued; the averaging gain is model-specific (+0.028 / +0.075 on ours, +0.019 on theirs) |
  | C2-lo - zero-shot (exposure at a 3.87% dose) | registered as a dose statement: above +0.05 is informative, a null is a result about the dose (Amendment 1) | +0.141 [+0.101, +0.182], 3 paired seeds (`schach_paired_review_addenda.json`; the aggregate certificate's own draw reads [+0.100, +0.183]) | above +0.05: a 3.87% dose carries |
  | C2-hi - Z-676 (exposure at a 14.1% dose, 495 identities) | band +0.05 to +0.20; falsifier < +0.03; [+0.03, +0.05) unnamed (Amendment 1) | +0.089 [+0.047, +0.131], 1 seed | band held at the mean; the lower edge sits 0.003 below +0.05 and is not argued either way |
  | Z-676 - zero-shot (identity count without exposure) | within +-0.03 (Amendment 1) | -0.013 [-0.039, +0.013], seed 1 (three-seed per-user means: -0.016 [-0.040, +0.007]) | mean inside the band; the interval extends past -0.03, so flatness holds at the mean and is not resolved at the edge |
  | C2-hi - C2-lo (dose effect, predicted positive) | directional (Amendment 1) | -0.068 [-0.099, -0.037] against the three-seed per-user mean (`schach_paired_review_addenda.json`); the aggregate's seed-1 pairing reads -0.061 [-0.099, -0.026] | **FAILED**: sign reversed; the dose reading is withdrawn and both arms are read against their own controls (Amendment 5) |
  | C2-lo-half - C2-lo (dose at fixed identities) | within +-0.03; below -0.03 dose binds; above +0.03 regularisation (Amendment 5) | -0.028 [-0.062, +0.009], 1 seed | mean inside the band; the interval extends below -0.03: "dose may bind" is not excluded and is reported as unresolved |
  | C1-full - C1 (budget-matched their-protocol arm) | [0, +0.08]; < 0 named "capacity-limited" (Amendment 3 and addendum) | +0.033 | HELD: C1 was under-trained by about 0.03 |
  | C1-full stays below the zero-shot 0.234 | falsifier: at or above 0.234 (Amendment 3) | 0.164 [0.128, 0.205] | HELD: their protocol on our model, trained out, is still 0.07 below zero-shot |
  | Test-fitted alignment ceiling A2' - A1, zero-shot | band >= +0.15; programme falsifier CI upper < +0.05; [+0.05, +0.15) unnamed (original registration; defect recorded in Amendment 2) | +0.026 [+0.000, +0.051], 3 seeds | band excluded by a factor of 2.9 at the interval's upper end; the falsifier missed by 0.001 and the result lands in the unnamed gap - not argued; the verdict rests on where the interval fell |
  | **Honest alignment A2 - A1, zero-shot** (fit on users 0-31) | band +0.05 to +0.20; falsifier < +0.05 (original registration) | +0.011 [-0.020, +0.041], 3 seeds | **band excluded, falsifier FIRED** (negative result, reported as such) |
  | Honest alignment, exposed and smaller arms | same band | C2-lo -0.003 [-0.011, +0.005]; C2-hi -0.003; Z-676 -0.005; C1 -0.002 | band excluded on every `dyn` arm |
  | Alignment guard: permuted correspondences A2-null <= A1 + 0.03 | guard (original registration) | -0.074 [-0.126, -0.029] | HELD: the fit is person-specific |
  | Alignment guard: unrestricted 128-d fit A2-full <= A2 | guard (original registration) | -0.055 [-0.086, -0.029] | HELD |
  | Alignment seed check: A2 - A1 seeds within 0.05 | original registration | seed range 0.010 | HELD |
  | Margin/scale screen: M-zero - zero-shot | band -0.02 to +0.04; above +0.04 the screen fires; below -0.02 the sign-flip reading (Amendment 7; a SCREEN resolving about +-0.037) | -0.028 [-0.045, -0.012], 1 seed | below -0.02: **sign reversal at 4,096 identities**; size against the edge not resolved; no seeds and no M-C2-lo, as registered |
  | Activity diversity (Nymeria swap, earlier programme) | +0.005 to +0.03; falsifier < +0.005 | -0.0012 [-0.0045, +0.0020], 5 paired seeds | band excluded [CL] |
  | Across-XR within-application lateral placement P(within<between) | band 0.80-0.95; falsifier < 0.65; [0.65, 0.80) unnamed (`across_xr_within_application.json`) | 0.7525 [0.7114, 0.7748] | lands in the unnamed gap: a registration defect, recorded; the design consequence (a same-application arm is not placement-free) holds under either mechanism |
  | Questset lateral P at chance | band 0.45-0.60 [QG] | 0.526 (group 1) / 0.549 (group 2) | HELD in both groups |
  | Questset height P, group 2 (standing vs seated) | band > 0.65; falsifier < 0.60; 0.60-0.65 "weakened" [QG] | 0.493 | **FALSIFIER FIRED** |
  | Questset height P, group 1 (both standing) | band > 0.65 [QG] | 0.718 | HELD |
  | Questset height-only lookup, group 2, N=30 | < 0.10 holds; 0.10-0.15 weakened; > 0.15 falsifier [QL] | 0.033 | HELD (exactly chance) |
  | Questset height-only lookup, group 1, N=30 | > 0.15 [QL]; no outcome named below the band | 0.092 | **NOT MET** (2.8x chance, below the registered level) |

- **Say it this way:** a row that reads FAILED or NOT MET stays in the table and in the paper. The
  registered size of P2 was wrong by a factor of two in the direction that makes the static cue
  larger; the P3 ordering held at the top and failed at the bottom; the P3 ceilings show the
  test-fitted rotation on three of five held-out applications. None of these changes a headline,
  and all of them are reported.

## C-D1 - Across-XR split (optional)
- **Type:** simple bar split into three segments.
- **Data [PLAN]:** users 0-22 train (23), users 23-31 validation (9), users 32-48 test (17). Label:
  "identical to Schach et al.'s protocol; read from the corpus's `split` column".

---

# Part D - Reference tables

## D.1 Datasets held

| dataset | users | applications / activity | licence / terms | role in this work |
| --- | --- | --- | --- | --- |
| **Across-XR** (Schach et al. 2026) | 49 (split 23 / 9 / 17) | 5: Superhot VR, Half-Life: Alyx, Beat Saber, Synth Riders, Social VR | CC BY-NC-SA 4.0 | Evaluation instrument; training users 0-22 in the exposed arms [PLAN, CAT] |
| **Questset** (Baldoni et al., MMSys '24) | 60 complete (70 recruited) | 4 titles, 2 per user: Beat Saber + Cooking Simulator; Medal of Honor + Forklift Simulator | CC BY 4.0 | Second cross-application corpus; second gallery size (N=30); arms registered and queued [CL, CAT] |
| **BOXRR-23** (Nair et al.) | 4,020 users converted (release has 105,852) | Beat Saber | CC BY-NC-SA 4.0 plus a signed Data Use Agreement; ethics approval in place | Pretraining (head track only) [CL, CAT] |
| **who-is-alyx** (Rack et al. 2023) | 76 players, mostly two sessions on different days | Half-Life: Alyx | Zenodo 10.5281/zenodo.8379914 | Pretraining [CL, CAT] |
| **Nymeria** (Ma et al. 2024) | 50 participants held | 17 scripts of daily life, real AR glasses | CC BY-NC 4.0 | Earlier activity-diversity experiment (null) [CL, CAT] |
| Seven seated / navigation corpora (Head_and_Gaze 100, PanoSaliency 99, VR_User_Behavior 48, ViewGauss 35, EyeNavGS 22, Panonut360 21, NJIT_6DOF 18) | 343 in total | 360-degree video, navigation, room-scale walking | per dataset (see catalogue) | Held-out corpora for the earlier transfer and identity-count work [CL, CAT] |

## D.2 Glossary

| term | meaning |
| --- | --- |
| **rank-1** | Fraction of probes whose nearest enrolled user is the correct one (closed-set identification) |
| **N** | Gallery size, the number of candidate users. Always quoted with rank-1 |
| **chance** | 1/N: 0.0588 at N=17, 0.0333 at N=30 |
| **gallery** | Enrolment data: here, a user's data in application A |
| **probe** | The query sample: here, a 10 s window (or ten minutes) from application B |
| **verification vs identification** | Verification: are these two samples the same person (two classes, chance 0.50, reported as AUC). Identification: which of N people is this (chance 1/N, rank-1). Never compared with each other |
| **cross-application** | Gallery and probe come from different applications; averaged over the 20 ordered pairs |
| **`dyn`** | Our encoding: pose relative to the window's own mean pose, gravity kept. Removes height, seat and placement, so any signal left is behavioural |
| **`raw`** | Pose as recorded, including absolute head position (height, placement, posture) |
| **BRV** | Body-relative velocity, Schach et al.'s encoding: pose relative to the HMD, then differentiated. Head position never reaches their model |
| **zero-shot** | Our model trained only on other corpora (BOXRR-23, who-is-alyx) and never on any Across-XR data |
| **exposure / C2-lo** | Zero-shot corpus plus Across-XR training users 0-22 (3.87% of training windows). The same training users as Schach et al.; the test users 32-48 are never seen |
| **Z-676 / C2-hi** | The smaller-identity pair (495 trained identities) without and with exposure (C2-hi dose 14.1%) |
| **C1 / C1-full** | Our model trained on Across-XR users 0-22 only (their protocol); C1-full is the budget-matched run |
| **P3** | Leave-one-application-out training: an application unseen in training and users unseen in training |
| **D1 / D2** | D1 = our embeddings scored by their calculator; D2 = their embeddings scored by our harness |
| **P(within < between)** | The probability that a person's own two per-application mean positions are closer together than two different people's. 0.5 = no static cue; 1.0 = a perfect per-person constant |
| **registered prediction** | A prediction written down and dated before the run, with a band |
| **falsifier** | The pre-stated outcome that would refute the prediction |
| **gate** | A check that a checkpoint or harness reproduces a known recorded value before any new number from it is used |
| **self-match** | A query window that shares frames with a reference window from the same recording |

## D.3 What each certificate proves

| file | proves |
| --- | --- |
| `docs/acceptance/schach_release_gate.json` | Their released `embeddings.pkl` is hashed and loaded safely (11 whitelisted globals). All 463,996 embeddings are accounted for across 85 (user, application) cells, with 17 distinct per-user count vectors (so label i = user 32+i). Their calculator, run verbatim, reproduces every per-user value in all 35 cells at max abs diff 0.0. Within mean 0.8314, cross mean 0.1804 [GATE, PLAN] |
| `docs/acceptance/schach_paired.json` | The paired per-user comparison in both directions (D1, D2) for zero-shot and C2-lo, with levels, paired intervals, users won, per-cell results and ten-minute figures, plus the `raw` arms' levels. Each of our seeds is rechecked against its certificate (max abs diff 0.0) [PAIR] |
| `docs/acceptance/schach_ours_*_gate_cpu.json` | Each of our checkpoints, rescored on CPU, reproduces its own recorded verification AUC within tolerance (for example, zero-shot seed 1 gap 1.97e-5) [PAIR-adjacent certificates] |
| `docs/acceptance/questset_geometry.json` | The registered prediction and the per-group P(within < between) for all / lateral / height, plus per-person height change. Gated first on a synthetic fixture that must return 1.000, and 0.000 when inverted [QG] |
| `docs/acceptance/questset_static_lookup.json` | The registered prediction and the training-free mean-position lookup at N=17 and N=30 per group and axis, both directions [QL] |
| `docs/acceptance/schach_per_user.png` / `.csv` | The per-user figure and its values, read from `schach_paired.json` with nothing recomputed [CSV] |
| `docs/acceptance/across_xr_alignment_p3_stability.json` | Seed-averaged P3 results per held-out application and pooled [P3S] |

---

# Part E - Limitations and threats to validity

1. **17 test users.** User-level uncertainty dominates, and intervals are wide. There is no fix inside
   the corpus. It is mitigated by intervals everywhere, the per-user figure, per-cell results (11 of 20
   resolved, none lost), and Questset's second gallery size (N=30) [PLAN].
2. **One fully crossed corpus.** Across-XR is the only one known. Questset gives each user 2 of 4
   titles in disjoint groups, so it is two 2-application corpora of about 30 users, not a crossed
   design. BOXRR-23 has zero users in two applications [PLAN].
3. **No temporal persistence.** Every (user, application) cell in Across-XR is one unbroken recording.
   Cross-application pairs are up to about an hour apart within one sitting and never on different
   days. Questset is also one sitting per user, one session per game [PLAN, CL].
4. **The architecture is not novel** (BiLSTM with AM-Softmax). The paper is not a modelling paper and
   will not be framed as one [PLAN].
5. **Self-match in their within-application figure.** Their 83.1% cannot be paired with our
   within-application figure. The within-application gap remains unresolved and confounded between
   sensor set and architecture [PLAN].
6. **The per-user rank-disagreement finding is exploratory.** It is post hoc, has n=17 and a CI of
   about +-0.5, and its attribution is confounded (sensor set, architecture and encoding all differ).
   The ensemble/coverage corollary failed [PLAN].
7. **Single-seed arms.** C1, C1-full, Z-676, C2-hi, the half-dose arm, three P3 applications, `raw`
   C2-lo and the margin screen are single-seed. Seed variance is specific to each arm (0.010 on
   trained-out rank-1 against 0.029 for epoch-1 `raw`) [OUT].
8. **Matched metric is not matched decision rule.** Theirs is nearest reference window with 15 s at 30
   FPS; ours is a mean template with 10 s at 20 Hz. This is addressed by scoring both directions with
   both harnesses (D1 and D2) [OUT, PLAN].
9. **The headline arm choice is post hoc relative to P1.** P1 targeted zero-shot. C2-lo's contrast was
   registered before it ran [OUT].
10. **The pretraining activities overlap two of the five applications** (Beat Saber, Half-Life: Alyx).
    The coverage control in P3 addresses this [OUT].
11. **The posture finding is single-corpus**, and its registration was written the same day it ran. It
    is reported small for that reason [PLAN].

---

# Part F - Next steps and decisions for Dr. Feng

## F.1 Running now

- **Rack et al. 2023 reproduction** on the lab GPU server (the second baseline, whose architecture is
  public). Status: **RUNNING**. Measured cost: 21.5 minutes per epoch including validation, about 36
  hours per seed at 100 epochs [CL]. Seed 1 is expected around 2026-09-17 11:00 [COORD].
- **Questset arms A1 / A2 / A3**, registered before the corpus reached the server and queued behind
  the Rack run. Status: **REGISTERED-AND-RUNNING (queued; no numbers yet)** [COORD]:
  - **A1:** zero-shot cross-application at matched N=17 and at N=30. Band 0.15-0.40, falsifier below
    0.10, outcomes partitioned.
  - **A2:** group 2 (standing/seated) as the clean behavioural arm against group 1 as the contrast
    (`raw` minus `dyn` predicted smaller on group 2).
  - **A3:** covered/uncovered control (Beat Saber is in our pretraining; the other three titles are
    not).
- The per-user rank-disagreement prediction is to be **registered before** the Questset arms run, so
  that the replication is a real test [PLAN]. Status: **PENDING**.

## F.2 Recommendation (from `docs/PAPER_PLAN.md`)

**Write the paper now**, framed as a **rigorous re-assessment of cross-application XR biometric
risk**, not as "our model is better" [PLAN]. The contribution set under that framing:
1. The beat on the SOTA's own data, people and metric, using their released model, while head-only
   against head plus both controllers and with 10 s windows against 15 s.
2. The static-cue audit, showing that behaviour-only assessments understate the risk.
3. Their proposed fix (train-user-only alignment) closed with a mechanism.
4. The per-user distribution as a metric argument.

Questset does not change the recommendation. It makes the per-user hypothesis testable and adds a
second gallery size. If its arms land before submission they strengthen the paper, and if not,
nothing is lost [PLAN].

Two cautions recorded in the plan: frame the self-match observation structurally and never as an
error, since Schach et al. are plausible reviewers; and make the protocol match (train 0-22, test
32-48, the same as theirs) impossible to miss, because "you beat them by training on their corpus" is
the attack that matters most [PLAN].

## F.3 Decisions that belong to Dr. Feng

1. **Venue.** The natural targets are where the SOTA published (Frontiers in Virtual Reality), or a
   privacy venue if the risk-assessment framing leads [PLAN].
2. **How much novelty a first paper needs** in this programme, given that the architecture is
   standard and the contribution is protocol, measurement and negative results [PLAN].

## F.4 Recorded decision on the posture finding (2026-09-16)

> *"I think we should include the cross task needing the same posture but we won't focus on it too much"*

As recorded [PLAN]:
- **The qualifier is obligatory.** Wherever the paper says head height survives a change of
  application, it says "across applications that share a posture".
- **The finding is bounded:** one subsection in the static-cue audit (one paragraph plus the two-row
  table), one sentence in limitations, and nothing in the abstract, the contribution list or the
  figures.
- **Questset's role stays the same:** the second gallery size.

---

# Part G - References

1. L. Schach, C. Rack, R. P. McMahan, M. E. Latoschik. *Motion-Based User Identification across XR
   and Metaverse Applications by Deep Classification and Similarity Learning.* Frontiers in Virtual
   Reality, 2026. doi:10.3389/frvir.2026.1743491. Preprint arXiv:2509.08539. The alignment analysis
   (section 6.2.5, 52.3% / 94.3%) appears only in the Frontiers version. Data (Across-XR), CC BY-NC-SA
   4.0: https://go.uniwue.de/identification-across-xr-applications. Code, data and models:
   gitlab.informatik.uni-wuerzburg.de/hci/software/research-prototypes/2025-frontiers-identification-across-xr-applications/
2. **V. Nair, W. Guo, J. Mattern, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song. *Unique
   Identification of 50,000+ Virtual Reality Users from Head & Hand Motion Data.* USENIX Security
   2023. arXiv:2302.08927.** Citation required by clause 5 of the BOXRR-23 Data Use Agreement.
3. V. Nair, W. Guo, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song. *Berkeley Open Extended Reality
   Recordings 2023 (BOXRR-23): 4.7 Million Motion Capture Recordings from 105,852 Extended Reality
   Device Users.* IEEE TVCG, 2024. doi:10.1109/TVCG.2024.3372087. arXiv:2310.00430.
4. C. Rack, T. Fernando, M. Yalcin, A. Hotho, M. E. Latoschik. *Who is Alyx? A new behavioral
   biometric dataset for user identification in XR.* Frontiers in Virtual Reality, 2023.
   doi:10.3389/frvir.2023.1272234. Dataset: Zenodo doi:10.5281/zenodo.8379914.
5. C. Rack et al. *Versatile User Identification in XR using Pretrained Similarity-Learning.*
   arXiv:2302.07517 (2024).
6. C. Rack, A. Hotho, M. E. Latoschik. Motion encodings (SR/BR/BRV/BRA), IEEE AIVR 2022; and
   C. Rack et al., *Motion Learning Toolbox*, IEEE VRW 2024.
7. Baldoni et al. *Questset: A VR Dataset for Network and QoE Studies.* ACM MMSys '24.
   doi:10.1145/3625468.3652187. Data (CC BY 4.0): researchdata.cab.unipd.it/1239/
   (doi:10.25430/researchdata.cab.unipd.it.00001179).
8. S. Baldoni, S. Benhamadi, F. Chiariotti, M. Zorzi, F. Battisti. *Movement- and Traffic-based User
   Identification in Commercial Virtual Reality Applications: Threats and Opportunities.* arXiv:2501.16326,
   2025 (author list and title verified on arXiv). On Questset. Its test users are seen during training
   (first 8 minutes train, next 2 minutes test, within each participant), and it uses head plus both
   controllers. It therefore measures **a different quantity** from ours (recognising a known,
   enrolled user). It reports >95% within-game on Beat Saber and Forklift Simulator, ~80% on Medal of
   Honor and Cooking Simulator, and no model above 0.3 cross-game at N=30 [CAT].
9. L. Ma, Y. Ye, F. Hong, et al. *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion
   in the Wild.* ECCV 2024. arXiv:2406.09905. CC BY-NC 4.0.
10. Baldoni et al. 2025 (IEEE VR), ~30% cross-application identification across two applications,
    classification [OUT]; and Rogers et al. 2015 and Miller et al. 2020/2021, as cited in Schach et
    al. Full bibliographic entries to be taken from Schach et al.'s reference list.
11. AM-Softmax (additive margin softmax), the objective used here. The reference is not in any project
    source and must be added by hand.

---

# Part H - Numbers to confirm

**All items below are RESOLVED.** Each was raised while drafting and checked against the sources before
this report was finalised. They are kept so the reasoning behind each figure is visible.

1. **Level intervals for zero-shot and C2-lo - RESOLVED: use the settled certificate everywhere.**
   Project notes quote zero-shot 0.234 [0.181, 0.292] and C2-lo 0.375 [0.321, 0.435]; the settled
   paired certificate `schach_paired.json` gives 0.2336 [0.1817, 0.2915] and 0.3745 [0.3214, 0.4333].
   Same quantity, separate bootstrap runs, differing only in the third decimal. **This report uses the
   certificate values throughout** - zero-shot [0.182, 0.292], C2-lo [0.321, 0.433] - so no quantity
   appears with two different intervals on two slides.
2. **Their per-user interval edges - RESOLVED the same way.** Certificate values are used everywhere:
   single window 0.1804 [0.1402, 0.2251], ten-minute 0.3082 [0.2080, 0.4199] (3 dp: [0.208, 0.420]).
   The ten-minute separation is unaffected: their upper edge 0.420 against our lower edge 0.571.
3. **Questset lookup, group 1 - RESOLVED: the registered prediction HALF HELD.** Registered at N=30:
   group 2 height-only below 0.10 (measured **0.033** - held) and group 1 height-only **above 0.15**
   (measured **0.092** - **not met**; it clears 0.15 only at N=17, 0.153). An earlier project note
   called the whole prediction "held" because its analysis script scored group 2 alone; that note and
   the script have been corrected, and the script now scores both bands. **The only acceptable wording
   is: "group 2 at chance as predicted; group 1 above chance but below its registered level."** The
   group separation on height survives (2.8x chance against exactly chance) and is supported
   independently by the geometry result (0.718 against 0.493). Both lookup rows are now in chart C14.
4. **Within-application `raw` A0 - RESOLVED: 0.723.** 0.730 was seed 1 alone, printed in a project
   note beside a three-seed difference. The three-seed mean is **0.723** (0.730 / 0.732 / 0.707), and
   0.723 - 0.500 = +0.223 [+0.184, +0.263]. The note has been corrected.
5. **Attribution of arXiv:2501.16326 - RESOLVED: Baldoni et al.** Verified on arXiv: S. Baldoni,
   S. Benhamadi, F. Chiariotti, M. Zorzi, F. Battisti, 2025. It is **not** a Schach et al. paper; it
   shares four authors with the Questset dataset paper. Full citation in Part G.
6. **Questset A1-A3 registrations - ACCEPTED.** They are recorded in the project's coordination log
   and are quoted exactly as registered there (A1 band 0.15-0.40, falsifier below 0.10).
7. **Two ten-minute figures present only in the certificate - ACCEPTED.** Their model under our vote,
   0.2882 [0.1912, 0.3941], and zero-shot's paired ten-minute difference under our vote,
   +0.069 [-0.091, +0.240], are taken from `schach_paired.json`, the settled source.
8. **Schach et al.'s test-fitted alignment gain - ACCEPTED as "+0.34".** The sources give it at that
   precision only (18.0% -> 52.3%); do not add digits.
