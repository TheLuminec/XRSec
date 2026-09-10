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
| 4. our algorithm | `dyn` + `identity_softmax`, head-only. Design fixed; the cross-application arm is unrun |
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
| `cschell/Versatile-XR-User-Identification` | `97f054baf04141ccf0943d207f2ff24a8e8bd1aa` | Rack et al. 2023. Holds BOTH model families the Across-XR paper evaluates: `similarity_module.py` and `classification_module.py` |
| `cschell/Motion-Learning-Toolbox` | `b8189e6c6250527b974c0aa5ccae309964eefe5f` | their preprocessing library - the BR/BRV encodings the paper cites |
| `cschell/Who-Is-Alyx-Code` | `2e28e22beada1fe92e5b51d088fce55d0a38244a` | same pipeline on the Alyx dataset, which we hold, so it is gateable |

**AVALON's IP is not rate-limited where DESKTOP-C's was.** The 429/403 that blocked this
acquisition for days was never about the repository; the same URLs return 200 here. Fetch
external material from AVALON.

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
