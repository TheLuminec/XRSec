# Paper plan: generalisation of motion biometrics across XR applications

Written 2026-09-10 on AVALON, after obtaining the SOTA paper and its authors' public code.
This is the durable statement of the goal the user set. Predictions here are REGISTERED:
they are dated, they have falsifiers, and they were written before any of the runs.

## The five steps, and where each actually stands

| step | status |
| --- | --- |
| 1. public dataset | **DONE** - Across-XR, 49 users x 5 applications, CC BY-NC-SA 4.0, converted and on three machines |
| 2. SOTA work: run their code, understand the story | **paper obtained and read; their two models' code obtained; their own wrapper repo is auth-gated** (see Blocked) |
| 3. evaluation metric | **defined below**, matched to theirs, with one addition of our own |
| 4. our algorithm | `dyn` + `identity_softmax`, head-only, **plus train-user-only orthogonal embedding alignment** - see "The algorithmic contribution" below. Unrun |
| 5. beat SOTA | **the target is 18.0% and it is written down before we run** |

## The SOTA: Schach, Rack, McMahan, Latoschik 2026 (Frontiers in VR; arXiv:2509.08539)

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
| nearest embedding accuracy (single 15 s window) | 83.1% (72.3-88.0) | **18.0%** (10.5-22.6), sd 15.1 |
| accuracy on a 10-minute sequence | 100% | 30.8% (9.0-57.7) |
| top-3 on a 10-minute sequence | 100% | 56.0% (29.1-76.4) |
| classification model, test accuracy | 43.2% | not supported by that model |

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
deployable, generalizing solution." Their section 8 names the fix as future work in as many
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
| **P2** `raw` minus `dyn` positive | (no band) | **UNRUN** - the programme ran `dyn` only | not tested |
| **P3** unseen-application cell below the seen-application cell | directional | **-0.036** [-0.054, -0.018] | **HELD** |

**The five claims the programme ends on.**

1. **Head-only, zero-shot, never trained on the corpus: 0.234** cross-application at N=17 over
   three gated seeds, against a published **0.180** that used head **plus both controllers**.
   Reported as a **placement against a published mean, not a beat** - the interval excludes
   0.180 by 0.001, and their figure is a mean whose per-cell distribution was never published,
   so no formal test is available at any margin.
2. **With in-domain exposure: 0.375** (three seeds, range 0.010), **+0.141** over zero-shot;
   ten-minute sequence **0.711** against their 0.308.
3. **Exposure carries to an UNSEEN application: +0.053** [+0.022, +0.083], interval excluding
   zero, falsifier excluded. The coverage control is what makes it stand - applications absent
   from every pretraining corpus read **+0.052** against **+0.055** for those present, so the
   carry is not pretraining leaking through the hold-out, and **Synth Riders (+0.077) has no
   pretraining coverage at all**. The stricter registered threshold for the phrase "crosses an
   activity boundary" (CI lower above +0.030) read +0.022 and is **reported as not met**.
   **This is the first data-side lever this project has measured to cross an activity
   boundary** - identity count is flat across one, activity diversity was null.
4. **Schach et al.'s section 8 is answered negatively, with a mechanism.** The honest
   train-user-only orthogonal fit **never carries** (A2 - A1 <= 0 on 14 checkpoints); the
   correspondences available for fitting are **capped at 32 by the corpus** - the number of
   people recorded in two or more applications - and no amount of pretraining raises it; and
   **the test-fitted ceiling that motivates the whole idea is itself run-dependent**, present in
   one of three runs at one configuration and three of five at another. So it was never a
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

1. **The Across-XR evaluation code is auth-gated.** `gitlab2.informatik.uni-wuerzburg.de`
   returns "404 Project Not Found" to an unauthenticated API call. The paper says the code
   will be published "upon publication" and the arXiv version is a preprint, so it is
   plausibly not released yet rather than withheld. I did not attempt to log in or create an
   account. Either the user asks the authors for it, or we proceed on their two models'
   public code, which we already have.
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
