# Paper plan: generalisation of motion biometrics across XR applications

Written 2026-09-10 on AVALON, after obtaining the SOTA paper and its authors' public code.
This is the durable statement of the goal the user set. Predictions here are REGISTERED:
they are dated, they have falsifiers, and they were written before any of the runs.

## The five steps, and where each actually stands

| step | status |
| --- | --- |
| 1. public dataset | **DONE** - Across-XR, 49 users x 5 applications, CC BY-NC-SA 4.0, converted and on three machines |
| 2. SOTA work: run their code, understand the story | **DONE** - paper read (Frontiers version), all three of their repos cloned from `gitlab.informatik` (not the auth-gated `gitlab2`), and **their released model's own evaluation code re-run to reproduce their published numbers bit-exact** (35 cells, max abs diff 0.0) |
| 3. evaluation metric | **defined below**, matched to theirs, with one addition of our own |
| 4. our algorithm | **`dyn` + `identity_softmax`, head-only, 10 s windows, with exposure to the corpus's *other* participants.** The orthogonal-alignment component was registered, run and **closed negative** - it never resolvably carries on `dyn`, and the corpus caps the correspondences at 32. Reported as a finding, not dropped |
| 5. beat SOTA | **DONE for the exposed arm, on their people, under their metric** - C2-lo +0.119 [+0.050, +0.192] against their 0.180, gate bit-exact. Zero-shot is **UNRESOLVED** and is reported as such |

## The SOTA: Schach, Rack, McMahan, Latoschik 2026 (Frontiers in VR, doi:10.3389/frvir.2026.1743491; preprint arXiv:2509.08539)

> **CITATION CORRECTION, 2026-09-15.** The GOPA alignment analysis - section 6.2.5, the 52.3% / 94.3% aligned
> figures, the "post hoc diagnostic upper bound" wording, and the future-work proposal to learn the
> transformations on training/validation users - exists **only in the Frontiers version**. The arXiv v1
> preprint held at `external_sota/schach2026.pdf` has none of it; this file cited it anyway, from a
> briefing that had read the Frontiers PDF. Verified against the Frontiers text
> (`external_sota/papers/schach2026_frontiers.pdf`). The unaligned headline numbers are identical in both.

Same group as who-is-alyx and the BOXRR conversion scripts. `external_sota/schach2026.pdf`.

**Their protocol, and we can match it exactly.** 49 users split into three disjoint user
groups: **first 23 train, next 9 validation, final 17 test**. Our converted copy carries a
`split` column that reproduces that partition digit-exactly - train users 0-22, valid 23-31,
test 32-48 - so their split is not something we approximate, it is something we read off the
file. Preprocessing: resample to 30 FPS, **BRV** (body-relative velocity) via their
Motion-Learning-Toolbox. Window 450 frames at 30 FPS = **15 s**. Architecture: Transformer
followed by a GRU, cosine similarity on a 480-d embedding, nearest-reference-embedding
matching with majority voting over a sequence.

**Their headline numbers, at N=17 test users, chance = 1/17 = 5.88%:**

| metric | within-application (diagonal) | **cross-application (off-diagonal)** |
| --- | --- | --- |
| nearest embedding accuracy (single 15 s window) | 83.1% (72.3-88.0) | **18.0%** (10.5-22.6), mean per-cell sd 15.1 (across users) |
| accuracy on a 10-minute sequence | 100% | 30.8% (9.0-57.7) |
| top-3 on a 10-minute sequence | 100% | 56.0% (29.1-76.4) |
| classification model, test accuracy | 43.2% | not supported by that model |

The parenthesised ranges above are **across-cell** (the spread over application pairs, as their
paper reports it). **The comparison we need is across USERS, and it is now computable** - their JSON
ships `precision_at_1` and `sequence_top_1_accuracy_list_*` as 17 per-user values in every one of the
35 cells. Per-user means over cells, cluster bootstrap over the 17 users, 10,000 resamples
(2026-09-15; all four point estimates reproduce the published figures exactly):

| Schach et al., **per-user 95% CI** | single 15 s window | 10-minute sequence |
| --- | --- | --- |
| **cross-application** (20 off-diagonal cells) | **0.1804** [0.1396, **0.2246**] | **0.3082** [0.2061, **0.4169**] |
| within-application (5 diagonal cells) | 0.8314 [0.8102, 0.8520] | 1.0000 [1.0000, 1.0000] |

**This settles the zero-shot claim against us, and that is the correct outcome.** Their
single-window interval reaches **0.2246**; our zero-shot 0.234 [0.181, 0.292] overlaps it heavily.
So "no beat is available" is no longer a caution about an unpublished distribution - it is a
measured overlap. The only sentence that survives is the one already registered: **every seed sits
above their reported mean, head-only and zero-shot.** Do not strengthen it.

**The ten-minute contrast, by contrast, separates cleanly.** Their 0.3082 tops out at **0.4169**
and our exposed arm reads 0.66-0.711 - outside their interval entirely. That is the widest
separation anywhere in the comparison and it is now interval-against-interval rather than
point-against-point. **It is nevertheless the SECONDARY metric and is reported beside the registered
single-window contrasts, never in place of them** - see the emphasis note below, which is a
correction to my own framing rather than a caveat on the number. **Their within-application ten-minute figure is 1.0000 for all 17 users**, so
that metric is saturated there and separates methods only across applications - the reason it is our
secondary metric, now demonstrated on their data rather than argued.

**And their own per-user spread makes our metric contribution for us.** On the same 17 people,
cross-application per-user rank-1 runs **0.068 to 0.371** on a single window and **0.043 to 0.818**
at ten minutes. A risk assessment reported as a mean conceals a person identified four-fifths of the
time behind a population figure of 0.31. **Make the distribution argument on the reference's own
published numbers first, then on ours** - it is far stronger than making it only on ours, and it
costs nothing because the arrays are in their release.

**The diagonal column contains self-matches; the off-diagonal column does not.** Verified in their
code (`slm_compute_accuracies.py:48-57`, `slm_compute_embeddings.py:15`): embeddings are computed at
stride **5 frames**, and a cell's reference set is `embeddings[comments == ref][::150]` - so on the
diagonal the reference is a strict subset of the queries, drawn from the same unbroken recording,
with `ref_includes_query=False` so the kNN never excludes the identical vector. References sit 750
frames apart against a 450-frame window, so **every diagonal query window shares frames with some
reference window**; 0.67% are the identical vector, 59% share at least half their frames. Their
all-five-reference cell (0.802) is affected the same way because `embeddings[::150]` spans the
query's own application.

**Consequence for this paper, and it is narrow:** every number we place against them goes against
**18.0%**, which is cross-application by construction and clean, so **the comparison is unaffected**.
What we must not do is pair our within-application A0 against their 83.1% - and what we should say,
once, without accusation, is that their within- and cross-application figures are **not on the same
footing as each other**, so the 83.1 -> 18.0 drop overstates the collapse by whatever the self-match
is worth. It is structural in their design rather than an error: gallery and probe are the same
recording, and the corpus holds no second take per (user, application) cell.

**The story of their paper, in one line: within an application motion identification is
close to solved, and across applications it collapses to roughly three times chance.** Their
abstract says so outright - "their ability to identify users across different XR applications
remains limited". That collapse is the gap our paper exists to close, and 18.0% is the number
to beat.

**Two things about their setup that shape ours.**

1. **Their encoding already removes static cues, so their baseline is clean.** BR referencing
   fixes the HMD at its own origin and BRV then differentiates, so head position never reaches
   their model and static offsets do not survive the derivative. Their 18 features per frame
   are HMD **rotation only** (4) plus both controllers' position and rotation (14). We must
   NOT claim their numbers are inflated by placement or height - they are not, and this
   project's static-cue audit does not apply to them. It applies to the *field*, and to our
   own `raw` rows.
2. **Their "cross-application" is unseen USERS, not an unseen APPLICATION.** The model is
   trained on all five applications and tested on held-out people, with gallery drawn from one
   application and probe from another. **The unseen-users-AND-unseen-application cell is never
   tested by them**, and it is the deployment-realistic one: a new application appears that
   was not in training. That cell is free for us to run and is a genuine contribution.

## The metric (step 3)

**Primary, matched to theirs so the comparison is like-for-like:** rank-1 nearest-embedding
accuracy at **N=17**, on **their test users 32-48**, gallery from application A and probe from
application B, averaged over the 20 ordered off-diagonal pairs. Quoted with N always, per the
standing rule. The 10-minute sequence figure is the secondary, because it is the one that
saturates within-application and therefore separates methods only across applications.

**Our addition, and it is the metric contribution:** report the **distribution over users, not
only the mean.** Their paper is framed as a risk assessment of unwanted identification; a risk
assessment is a claim about the most exposed person, not the average one. This project has
already measured that per-user rank-1 spreads **24x wider than its score model implies**, with
15 of 94 users above 0.80 and 13 below 0.10 where a Gaussian null produces neither. Their own
code already contains a `MinAccuracy` (worst-class) metric and their Figure 3 shows user-wise
box plots, so the ingredients are theirs; nobody reports the worst case as a headline. **"The
model identifies users at 18%" and "a user has an 18% chance of being identified" are
different claims and only the first is supported.**

## The algorithmic contribution: the SOTA published its own ceiling and disqualified it

**Schach et al. section 6.2.5 is the most important paragraph in their paper and it is an
invitation.** They find that the embedding spaces learned for different applications differ by
an **orthogonal transformation only** - rotation and reflection, no scaling, no translation -
and that aligning them moves cross-application accuracy **18.0% -> 52.3%** on a single window
and **30.8% -> 94.3%** at ten minutes. They then disqualify their own result in section 9,
because the rotations were fitted **on the test users**: "a diagnostic upper bound, not a
deployable, generalizing solution." Their future-work paragraph (Frontiers version only) names the fix as future work in as many
words - learn the orthogonal transformations on training/validation users only, then apply them
to unseen test users.

**So the ceiling is published, the illegitimacy of their route to it is published, and the
protocol that would make it legitimate is published as an open problem.** We hold the corpus,
and our converted copy carries their exact 23/9/17 split, so the training/validation users to
fit on are the ones their own paper names.

**52.3% is simultaneously the target and the leakage detector.** It is a test-fitted upper
bound, so an honest train-user-only result must land below it. **Exceeding 52.3% is evidence of
leakage rather than of success** - a registered prediction carrying a built-in falsifier for its
own best outcome, which is rare enough to be worth stating as the design's main virtue.

**The prior negative is real, is scoped, and its own stated mechanism sets the scope.**
`docs/GENERALISATION_PROPOSAL.md` 5.5 lists GOPA-style adaptation under "what not to spend runs
on", measured at zero on alyx over five checkpoints. It does not govern this experiment, for two
reasons the proposal itself supplies. Everything tested there - centring, CORAL, donor
statistics - is **label-free and correspondence-free**, fitted across corpora with **disjoint
populations**, and a rotation cannot be fitted without paired points. Schach's alignment works
because **the same 49 people appear in all five applications**. And the proposal's stated reason
for its null is an embedding "mostly a position lookup", which is a `raw` property; under `dyn`,
and under their equally static-free BRV, it does not hold - and that is the regime Schach
measured the orthogonal difference in. **A negative with a stated mechanism can be checked for
scope; that is what makes stating the mechanism worth the words.**

Related, and second in line rather than first: section 5.3 rejected DANN because "with two or
three source domains the dataset classifier is trivial". Across-XR is **five applications over
one population**, so the stated reason no longer applies unexamined. Not an endorsement - a note
that it needs re-deciding rather than inheriting.

## OUTCOMES - 2026-09-11, after the programme. Scored against the registrations below.

**The three predictions registered in this file, scored.**

| | registered | measured | verdict |
| --- | --- | --- | --- |
| **P1** head-only `dyn` cross-application rank-1 @ N=17 | band 0.18-0.35, falsifier < 0.12 | **0.234** [0.181, 0.292], 3 seeds | **HELD**, inside the band |
| **P2** `raw` minus `dyn` positive | (no band) | **+0.117** [+0.042, +0.192], 3 seeds, epoch-1 selection (run 2026-09-11; this row said UNRUN until 2026-09-15) | **HELD** |
| **P3** unseen-application cell below the seen-application cell | directional | **-0.036** [-0.054, -0.018] | **HELD** |

**The five claims the programme ends on.**

1. **Head-only, zero-shot, never trained on the corpus: 0.234** cross-application at N=17 over
   three gated seeds, against a published **0.180** that used head **plus both controllers**.
   Reported as a **placement against a published mean, not a beat** - the interval excludes
   0.180 by 0.001, and their figure is a mean whose per-cell distribution was never published,
   so no formal test is available at any margin.
2. **With in-domain exposure: 0.375** (three seeds, range 0.010), **+0.141** over zero-shot;
   ten-minute sequence **0.711** against their 0.308.
3. **Exposure carries to an UNSEEN application: +0.049** [+0.021, +0.078] (seed-averaged; this read +0.053 on seed 1 until 2026-09-15), interval excluding
   zero, falsifier excluded. The coverage control is what makes it stand - applications absent
   from every pretraining corpus read **+0.046** against **+0.055** for those present, so the
   carry is not pretraining leaking through the hold-out, and **Synth Riders (+0.077) has no
   pretraining coverage at all**. The stricter registered threshold for the phrase "crosses an
   activity boundary" (CI lower above +0.030) read +0.022 and is **reported as not met**.
   **This is the first data-side lever this project has measured to cross an activity
   boundary** - identity count is flat across one, activity diversity was null.
4. **Schach et al.'s future-work proposal (Frontiers version) is answered negatively, with a mechanism.** The honest
   train-user-only orthogonal fit **never carries on `dyn`**, the headline encoding (A2 - A1 never
   resolvably above zero there; zero-shot seeds read +0.016/+0.011/+0.006, every interval spanning
   zero - the earlier "<= 0 on 14 checkpoints" was false, and the correction that replaced it
   omitted the encoding scope). **On `raw`, seed 1 reads +0.032 [+0.007, +0.057], resolvably above
   zero**, which the write-up must state rather than absorb: a static frame is exactly what an
   orthogonal map can rotate, so the negative is a `dyn` result and is reported as one. The
   correspondences available for fitting are **capped at 32 by the corpus** - the number of
   people recorded in two or more applications - and no amount of pretraining raises it; and
   **the test-fitted ceiling that motivates the whole idea is itself run-dependent**, present in
   one of three seeds at identical configuration (C2-lo) - the five P3 runs are five *different* configurations and do not bear on run-dependence. So it was never a
   target, and a single-run diagnostic bound of that kind is not evidence that application
   embeddings differ by a rotation. **That raises the evidential bar for every claim of this
   shape, including the published +0.34 this programme set out to reproduce.**
5. **Identity count is flat without exposure and not flat with it.** Dose is a small,
   roughly linear effect (halving in-domain windows costs -0.028; a 20% cut, -0.009) and cannot
   account for the pair - the higher-dose arm loses by 0.061, so **correcting for dose widens
   the scale effect rather than narrowing it**.

**Recorded as unresolved, and staying that way:** the within-application gap (confounded between
sensor set and architecture, and not isolable without their architecture head-only); the C2-hi
alignment dip; and how often the orthogonal structure appears.

**The composition is provable rather than asserted.** The loaders' own window counts close
exactly: C2-lo 540,107 minus Across-XR 20,896 = **519,211**, which is the zero-shot arm's
training set to the window. So the treatment arm is the control's corpus plus Across-XR 0-22 and
nothing else, and the dose is **3.87%** with the half arm at **1.96%**, exactly halved.

## Registered predictions - 2026-09-10, before any cross-application run

**P1. Head-only `dyn`, cross-application rank-1 at N=17 on users 32-48: band 0.18 to 0.35;
falsifier below 0.12.**
Below 0.12 says head-only cannot approach controller-based cross-application identification
and that our scope is a real cost on this axis, which we would report as such rather than
bury. Registered on `dyn` deliberately: `dyn` removes height, seat and placement, so a win
there is unambiguously behavioural. A win under `raw` would have two explanations - behaviour,
or the height cue this project measured at P=0.754 across these five applications - and per
the outcome-asymmetry rule the ambiguous outcome must be named in the registration, not in the
write-up.

**P2. `raw` minus `dyn` on the same folds is positive**, and its size is the anthropometric
contribution to cross-application identification. This is a measurement we want either way; it
is not a horse race.

**P3. The unseen-application cell (leave-one-application-out, unseen users) is below the
seen-application cross-application cell.** If it is not, their training-on-all-five design was
buying nothing and that is worth knowing.

**Power, settled before running rather than after.** At N=17 with 17 test users the effective
sample size is users, not windows: the binomial sd on a rank-1 near 0.18 is
`sqrt(0.18*0.82/17)` = **0.093**. So a single split cannot resolve a difference below about
0.09, and "we beat 18.0% with 21%" would be noise reported as a result. **A claim of beating
SOTA must be either a margin above ~0.10 absolute, or established paired across the 20
application pairs and several seeds.** The 20 off-diagonal pairs are correlated because they
share the same 17 users, so they are not 20 independent samples and must not be treated as
such. This is the same failure the Nymeria registration made - a band inside its own design's
noise floor - and it is one line of arithmetic to avoid.

## Blocked on the user - nothing here can be routed around

1. ~~**The Across-XR evaluation code is auth-gated.**~~ **RESOLVED 2026-09-15, and the blocker
   was a hostname.** The code, data and trained models are public at
   `gitlab.informatik.uni-wuerzburg.de/hci/software/research-prototypes/2025-frontiers-identification-across-xr-applications/`
   - `gitlab.informatik`, **not** the `gitlab2.informatik` that returned 404 unauthenticated and
   blocked this for days. Cloned to `external_sota/schach2026/` (`dataset` @ `565a3f39`,
   `dataset-preprocessing` @ `92222c24`, `training-and-evaluation` @ `4ec4106a`). Nobody logged
   in or created an account. **The lesson is worth more than the unblocking**: "the resource is
   gated" was an inference from one hostname, carried for days as a fact, and the check that
   settled it was trying the other host.
2. **Miami's GitHub deploy key** - still outstanding from 2026-09-09, still the only reason
   Miami's certificates need a human relay.

## What we already hold, verified today on AVALON

`external_sota/`, cloned from GitHub and pinned by commit:

| repo | commit | what it is |
| --- | --- | --- |
| `cschell/Versatile-XR-User-Identification` | `97f054baf04141ccf0943d207f2ff24a8e8bd1aa` | Rack et al. 2023. Holds the two training PARADIGMS (`similarity_module.py`, `classification_module.py`) and **not Schach's architecture** - see the correction below |
| `cschell/Motion-Learning-Toolbox` | `b8189e6c6250527b974c0aa5ccae309964eefe5f` | their preprocessing library - the BR/BRV encodings the paper cites |
| `cschell/Who-Is-Alyx-Code` | `2e28e22beada1fe92e5b51d088fce55d0a38244a` | same pipeline on the Alyx dataset, which we hold, so it is gateable |

**AVALON's IP is not rate-limited where DESKTOP-C's was.** The 429/403 that blocked this
acquisition for days was never about the repository; the same URLs return 200 here. Fetch
external material from AVALON.

## CORRECTION, 2026-09-10, same day: Schach's model is NOT in the public code

My first brief said the Versatile repo "holds BOTH model families the Across-XR paper
evaluates". **That is wrong and Miami caught it before spending hours on an install.** Verified
here independently: `transformer`, `Transformer`, `nhead` and `MultiheadAttention` return
**zero hits** across the whole clone; `machine_learning/src/models/` contains exactly
`cnn_model.py` and `rnn_model.py`; and `similarity_module.py` takes `model: nn.Module`, so it
is an architecture-agnostic wrapper. Schach's Transformer-into-GRU on a 480-d embedding is
absent.

**The error was conflating a training PARADIGM with an ARCHITECTURE.** The abstract names
"similarity learning and classification models", two files carry exactly those names, and I
read the match as the thing itself. That is this project's recurring bug once more: a stand-in
that looks like the thing being checked, and the match was on the *words in the abstract*
rather than on the contents. Filenames are a claim like any other.

**Consequence, stated so the paper does not overclaim.** "We reproduced the SOTA" and "we
measured ourselves against a number the SOTA published" are different claims, and on currently
available code **only the second is available for Schach et al.** That does not block anything
- 18.0% is a published target and their protocol is matchable exactly via the `split` column -
but the paper must say which of the two it is doing. The auth-gated GitLab repo is the likely
home of the missing architecture.

**What IS reproducible, and it is worth having.** Rack et al. 2023 on who_is_alyx is this
repo's own paper, and Miami verified the protocol reconstructs exactly rather than assuming it:
`01_aggregate.py` selects players with exactly 2 sessions, which takes AVALON's 76 players to
**63**, matching the config filename `15_fps-63_subjects-metric_learning_movement.hdf5` to the
digit; and the shipped config equals the paper's Table IV cell for cell (GRU x3, layer 450,
dropout 0.28, lr 2e-5, ArcFace, embedding 192). So the honest plan is **two baselines**: Rack's
architecture, which we hold and can run end to end on Across-XR under our own control, and
Schach's published 18.0% as an external reference we match protocol with but do not re-run.
Two baselines is stronger than either alone, and it converts the missing code from a blocker
into a stated limitation.

## Corpus fact recorded so nobody infers it wrongly

**`takeN` in our converted filenames is `game_id`, and `game_id` is NOT play order.** The
mapping is identical on all 49 users: take1 Superhot VR, take2 Half-Life: Alyx, take3 Beat
Saber, take4 Synth Riders, take5 Social VR. The paper's play order is Synth Riders, Superhot
VR, Beat Saber, Half-Life: Alyx, Social VR - i.e. game_ids **4, 1, 3, 2, 5**. Anyone treating
the take number as a session index gets the temporal ordering wrong on four of five
applications.

**And this partially reopens what the `take_id` finding closed.** That finding stands - each
(participant, game) cell is one unbroken recording, so there is no *within*-application
temporal separation. But the five applications were played **in sequence in one sitting, 10-15
minutes each plus breaks**, so a cross-application pair carries real temporal separation of up
to roughly an hour, and the separation is ordered identically for every participant. It is not
a different day and cannot pay the cross-session cost. It is also not nothing, which is what
"one sitting" alone implies.

### Ten-minute figures now have intervals on BOTH sides (New Gen, 2026-09-15) - PROVISIONAL

Cluster bootstrap over the same 17 users, seeds averaged inside users, computed the same way on
both sides. Under their sequence metric with parameters translated to our grid (118 windows = 600 s,
step one window = 5 s, asserted in the script) and **paired against their per-user ten-minute values**:

| arm | 10-min, their metric | paired vs their 0.308 | verdict |
| --- | --- | --- | --- |
| zero-shot (3 seeds) | 0.414 [0.323, 0.509] | **+0.106 [-0.039, +0.256]** | **UNRESOLVED** |
| C2-lo (3 seeds) | 0.663 [0.571, 0.753] | **+0.355 [+0.202, +0.499]** | **BEAT** |

Under our own harness the same arms read 0.357 [0.257, 0.474] and 0.711 [0.635, 0.785]; the `raw`
levels (0.434 and 0.497) are recorded for the audit and the headline stays on `dyn`. **The pattern
matches the single-window pairing exactly** - unresolved zero-shot, resolved for the exposed arm -
which is the agreement worth having, because the two metrics could have disagreed and did not.

**An independent cross-check fell out of this.** New Gen's aggregation of *their* per-user
ten-minute values returns **0.308 [0.206, 0.417]**, matching the interval computed here from the
same JSON by a different implementation to every printed digit. Two implementations over one
published array is a weaker check than two corpora, but it is the check that was available and it
passed.

**EMPHASIS NOTE, and it is a correction to the coordinator rather than to a number.** Having seen
that the ten-minute contrast separates and the single-window one does not, I told New Gen it was
"worth prioritising in the write-up over the single-window comparison". **The registration lists the
ten-minute figure as SECONDARY**, and New Gen pushed back correctly: it is reported *beside* the
registered single-window contrasts, never in place of them. Promoting the metric that happened to
give the better result - after seeing which one did - is moving the line by another route, and it is
harder to catch than moving a band because nothing numeric changes and every step looks like
editorial judgement. This file already carries the rule in its mirror form ("a 0.001 near-miss that
goes in favour gets the same treatment or the rule is not a rule"). **The registered primary metric
is the primary metric in the write-up, whichever way the secondary falls.**

Both rows stay **PROVISIONAL** until their calculator reproduces the JSON from the pickle; nothing
here is a certificate and nothing is relayed to the user as settled.

## SETTLED (2026-09-15): the paired comparison, no longer provisional

The calculator gate **passed bit-exact** and supersedes every "provisional" label above.
Certificates on `origin/main`: `docs/acceptance/schach_release_gate.json`, `schach_paired.json`.

**What the gate established, in the order the chain runs.** Their `embeddings.pkl` hashes equal on
two machines (`42a668ac...`); `pickletools` lists 11 globals, all numpy/pandas/builtins, loaded
through a whitelisting `Unpickler`; the pickle's **463,996** embeddings equal
`len(range(0, rows-450, 5))` summed over the released test CSVs in all **85** (user, application)
cells; and their `MotionAccuracyCalculator`, run verbatim on their embeddings, reproduces **every**
per-class `precision_at_1` and ten-minute list in all 35 cells at **max absolute difference 0.0**.

**Verified independently here (coordinator, from the CSVs alone, different implementation):** 85
cells, **463,996** windows, and **17 distinct per-user count vectors**. The last is the load-bearing
one - it is what makes `label i = user 32+i` a *reconstruction* rather than an assumption, because
each user's five-application window-count fingerprint is unique.

**Results, paired per user on their 17 test users, bootstrap over users.** D1 = our embeddings
through their metric; D2 = their embeddings through ours.

| contrast | registered | measured | outcome |
| --- | --- | --- | --- |
| zero-shot - theirs, **their** metric | UNRESOLVED | +0.025 [-0.031, +0.080] (0.206 vs 0.180), 10/17 users | **UNRESOLVED** |
| **C2-lo - theirs, their metric** | BEAT +0.08..0.20 | **+0.119 [+0.050, +0.192]** (0.299 vs 0.180), 15/17 | **BEAT** |
| zero-shot - theirs, **our** metric | UNRESOLVED | +0.035 [-0.050, +0.122] (0.234 vs 0.199), 10/17 | **UNRESOLVED** |
| **C2-lo - theirs, our metric** | BEAT +0.05..0.17 | **+0.176 [+0.092, +0.260]** (0.375 vs 0.199), 16/17 | **BEAT** |

Ten-minute, secondary and reported beside: C2-lo **+0.355 [+0.202, +0.499]** (their sequence metric)
and **+0.423 [+0.293, +0.546]** (our vote), BEAT under both; zero-shot unresolved under both. Per
cell, C2-lo beats resolvably in **11 of 20** ordered cells and **loses none**; zero-shot beats in 3
and loses none.

**THE SENTENCE THE PAPER CAN CARRY.** *Zero-shot against their model on their own people is
unresolved under both metrics, and 17 users cannot resolve a +0.05. Exposure to the corpus's other
participants, plus 4,096 identities, beats their released similarity model on their people, under
their metric and ours, single-window and ten-minute - head-only against head plus both controllers,
and 10 s against 15 s.* **Neither asymmetry qualifies the unresolved contrast**: they are stated
where we win, not borrowed as an excuse where we do not.

**A REGISTERED MECHANISM FAILED and is recorded as a failure, not folded into the beat.** New Gen
registered that template averaging would lift their embedding into 0.20-0.32 under our metric, from
the BOXRR k-curve argument. It lifts their model by **+0.019 only** (0.180 -> 0.199), against +0.028
for our zero-shot and +0.076 for C2-lo, so the level band is missed at its lower edge by **0.001**.
Two readings, and the second is the useful one: the miss is an edge and is **not argued** - the same
treatment this file gave the 0.001 near-miss that went in our favour - and **the averaging gain is
model-specific**, costing their nearest-reference embedding almost nothing while buying ours
0.03-0.08. That is a mechanism worth a sentence in the paper, and it was found by registering a
prediction that then failed.

**Two edges, both declared rather than argued.** Their model under our metric lands 0.001 below the
registered band; C2-lo under our metric lands 0.006 past its upper edge. Neither is claimed as the
band holding or failing.

**What is still NOT compared:** their 0.831 is never paired with our A0 (self-match, see above);
their 0.180 is clean and is what everything above is paired against.

## Per-user distribution: the figure, and an UNREGISTERED observation
*(Headed "…that may be the better paper" when written. After the two checks below it is a better
**hypothesis**, not a better paper - see the verdict at the end of this section.)*

`docs/acceptance/schach_per_user.{svg,png,csv}` from `schach_per_user_figure.py`, every value read
from `schach_paired.json` and nothing recomputed. Two panels (their nearest-reference metric; our
template metric), one row per test user 32-48 in the same order on both sides, three dots per row,
population means dashed, chance at 1/17.

**The observation: their model and ours rank the same 17 people in UNRELATED orders.** Verified here
independently from the certificate's per-user arrays, with a permutation test the original did not
run:

| | Spearman | permutation p |
| --- | --- | --- |
| their model vs our **zero-shot**, their metric | **-0.037** | 0.89 |
| their model vs our **C2-lo**, their metric | **-0.010** | 0.97 |
| *control*: our zero-shot, D1 vs D2 (two metrics) | **+0.939** | 0.000 |
| *control*: our C2-lo, D1 vs D2 | **+0.926** | 0.000 |
| *control*: our two arms vs each other (D1) | **+0.767** | 0.001 |

**The confound that would have killed it is excluded.** A near-zero correlation is uninformative if
either side is flat; neither is. Per-user spread is sd **0.090** (theirs), **0.069** (zero-shot),
**0.110** (C2-lo), over ranges 0.068-0.371, 0.105-0.363 and 0.148-0.528. So the near-zero reading is
a real disagreement about *who*, not an absence of variation - and the self-consistency controls rule
out noise in the per-user values themselves. **User 32 is their worst (0.068, barely above chance)
and among our best (0.295 zero-shot, 0.487 C2-lo); user 35 is their third best and near our floor.**

**One qualification the original framing did not carry, and it matters for what may be claimed.**
"The head-plus-controllers model and the head-only model find different people easy" attributes this
to the **sensor set**, but their model and ours differ in sensor set **and** architecture
(transformer+GRU vs BiLSTM) **and** encoding (BRV vs `dyn`) **and** training data, all at once. The
+0.767 between our two arms says training exposure alone does not scramble the order - same
architecture, same encoding, same sensors, different exposure - so exposure is largely excluded. The
remaining three move together and **this corpus cannot separate them.** The supportable sentence is
*"two systems that differ in sensor set, architecture and encoding rank the same people in unrelated
orders"*; naming the sensor set as the cause is a hypothesis, and it is the interesting one, but it
is not what was measured.

**Why it would matter if it held.** CLAUDE.md already holds that every rank-1 here is a population
mean over a 24x-wider-than-Gaussian per-user distribution, and that the exposed and protected
individuals are the substance of a biometric claim. This adds the sharper half: **if who is exposed
depends on the system rather than on the person, then a risk assessment cannot say "these people are
at risk" at all - only "this system exposes these people".**

**The difference in KIND is what makes it attractive, and it is worth naming precisely.** The beat is
a ranking claim - *we built a better instrument* - and a reviewer can accept it entirely and still say
"and next year someone beats you". It expires. This would be a claim about **what the field's central
number means**, which does not expire when a better model arrives. Two practical consequences follow
if it holds: **a per-person risk audit under one system does not transfer to another**, so measuring
yourself safe under one model is no evidence of safety; and **a defence that protects the top-k most
identifiable users has no stable target**, because the list changes with the system.

**Status: EXPLORATORY. Not registered, not paired, not claimed.** It is a figure caption and a
hypothesis. Before it is more, it needs registering in advance and a second corpus - and the
second corpus is the part that does not exist yet, since Across-XR is the only fully crossed
cross-application corpus we hold. **Do not let it into the abstract on the strength of one corpus
and a post-hoc correlation**, which is precisely the shape this file has been burned by before.

### The ensemble corollary FAILED, and "uncorrelated" was too strong (coordinator, 2026-09-15)

Two corrections to the entry above, both against my own enthusiasm and both found by checking before
propagating rather than after.

**The attacker-ensemble argument does not hold on this data and is struck.** I expected that because
the two systems rank people differently, running both would expose far more people than either alone
- the "the real risk is worse than either paper reports" reading. Users at or above a per-user
rank-1 threshold, their model against C2-lo against either:

| threshold | theirs | C2-lo | either | best single | union gain |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 6 | 14 | 15 | 14 | **+1** |
| 0.25 | 4 | 10 | 11 | 10 | +1 |
| 0.30 | 3 | 6 | 7 | 6 | +1 |
| 0.35 | 1 | 4 | 4 | 4 | **+0** |

**C2-lo nearly dominates their model outright** - only **2 of 17** users are caught better by theirs -
so the union buys at most one person and nothing at the high thresholds. The disagreement is in the
*ordering*, not in coverage. **Ordering-based arguments survive; coverage-based ones do not**, and
the distinction is worth keeping because they sound alike.

**"Unrelated orders" was too strong for n=17.** Fisher intervals on the Spearman:

| | rho | 95% CI |
| --- | --- | --- |
| theirs vs C2-lo | -0.010 | **[-0.49, +0.47]** |
| theirs vs zero-shot | -0.037 | [-0.51, +0.45] |
| *control*: our two arms | +0.767 | [+0.45, +0.91] |

At 17 users the interval is about +/-0.5 wide. It **excludes** the 0.77-0.94 the internal controls
show - the systems genuinely do not agree the way our own arms agree with each other - and it
**cannot exclude a moderate correlation**. The supportable phrase is **"not strongly correlated"**,
never "unrelated" or "uncorrelated". I wrote the stronger version into a message before computing the
interval; that is the error, not the estimate.

**What survives, precisely:** a defence that protects the top-k most identifiable users has no stable
target across systems, and a per-person risk audit under one system does not transfer to another.
Both are ordering claims. The coverage claim is withdrawn.

**VERDICT on this section (coordinator, 2026-09-15).** After the ensemble check failed and the
Fisher interval came back +/-0.5 wide, the honest ranking is: **the beat is the paper; this is a
figure plus a clearly-labelled observation with the n=17 caveat attached.** It is a better
*hypothesis*, not a better paper. If it replicates on a second corpus under a registration written
in advance, that is the follow-up, and it is the bigger one. Two sub-arguments survive (ordering:
no stable audit, no stable defence target) and one is withdrawn (coverage/ensembling).

---

## IS THIS ENOUGH FOR A PAPER? Assessment for the user's first paper - 2026-09-15

Asked directly by the user. Recorded because a judgement made once and left in chat is a judgement
nobody can check later, and because the answer shapes what gets written next.

**Short answer: yes, and without much hedging.** The result set is complete and coherent. The
binding constraint on strengthening it further - a second fully crossed cross-application corpus -
**does not exist**, so waiting is waiting indefinitely.

### What this work has that the field's norm does not

The typical XR biometrics paper is one dataset, 15-100 users, an architecture, and a table of
benchmark numbers. This has that plus four things that are rare:

1. **A beat on the SOTA's own data, own people, own metric, using their released model** - not a
   reimplementation. The gate reproduces their published per-user arrays at **max abs diff 0.0**
   across 35 cells. Most "we beat X" claims rest on a reimplementation and are arguable; this is not.
2. **The win carries a stated handicap**: head-only against head **plus both controllers**, 10 s
   against 15 s. A win under a handicap is a stronger claim than a straight win.
3. **Registered predictions with falsifiers, and negatives reported as prominently as positives.**
   Zero-shot is UNRESOLVED and says so. The alignment route - **their own published future work** -
   was tested and closed with a mechanism and an actionable corpus specification (32 correspondences).
   A registered mechanism (template averaging) failed and is recorded as a failure.
4. **The static-cue audit was turned on our own headline.** P2: a model **one epoch from
   initialisation** reaches 0.351 cross-application on `raw`. This is the second-strongest
   contribution after the beat and arguably the most interesting - it says a behaviour-only risk
   assessment **understates** the risk, which lands on Schach et al.'s framing rather than
   contradicting it, and their method cannot produce it.

### The three attacks a reviewer will make, and whether they land

| attack | lands? | the answer |
| --- | --- | --- |
| *"You beat them by training on their corpus."* | **No - but only if made unmissable** | C2-lo trains on participants **0-22** and tests on **32-48**. **That is exactly their protocol** - their model trained on 0-22 too. Matched, not advantaged. Put the split table early and explicitly; this is the attack that matters most |
| *"17 test users."* | **Yes, and there is no fix** - it is the corpus | Mitigate: intervals everywhere, per-user distribution figure, and per-cell results (**11 of 20 cells resolvably, losing none**) as 20 quasi-replications inside one corpus |
| *"One dataset."* | **Partially** | Across-XR is the **only** fully crossed cross-application corpus in existence, and we verified BOXRR-23 has **zero** users in two applications. That is a finding about the field's data, not an excuse |

### What is genuinely missing, and must be stated as limitation rather than found by a reviewer

- **No temporal persistence.** Every (participant, application) cell is one unbroken recording;
  `take_id` carries nothing. Cross-application pairs carry up to ~an hour of separation within one
  sitting, and **nothing about a different day**. Say so first.
- **The architecture is not novel** - BiLSTM + AM-Softmax. **Do not frame this as a modelling
  paper**; it is not one, and claiming otherwise invites exactly the wrong review.

### Recommendation

**Write it now**, framed as a **rigorous re-assessment of cross-application XR biometric risk** -
not as "our model is better". The contribution set is coherent under that framing: the beat under a
sensor handicap; the static-cue audit showing behaviour-only assessments understate risk; their
proposed fix closed with a mechanism; and the per-user distribution as a metric argument.

**Two cautions.**

- **Handle the self-match observation carefully.** This is a small field and Schach et al. are
  plausible reviewers. Frame it **structurally** - their design cannot avoid it, the corpus has no
  second take per cell - **never as an error**, and state in the same breath that their
  cross-application number is unaffected and is what we compare against.
- **Venue is the advisor's call, not the coordinator's.** The natural targets are where the SOTA
  published, or a privacy venue if the risk-assessment framing leads. **One thing this file cannot
  assess is how much novelty the user's specific programme expects of a first paper** - that is a
  question for the advisor, asked with this result set in front of them.
