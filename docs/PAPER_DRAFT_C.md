# Paper draft C: how far a head-motion biometric travels

First draft, 2026-10-05, written from thesis C in `docs/PAPER_THESES.md` (agreed by the user and the
Coordinator). Target: conference length, about 6,500 words of body text. It supersedes nothing:
`docs/PAPER_DRAFT.md` (the Schach-comparison draft) stays as it is, and much of its verified wording on
the reference study is reused here in shorter form.

Scope decisions taken with the user on 2026-10-05:
- Included beyond the core: the reference study's self-match structure (one paragraph), the per-user
  distribution, and the Questset posture finding.
- Left out: the orthogonal-alignment negative, the identity-count and dose arms, the leave-one-application-
  out table, test-time adaptation, the margin reversal, the Rack et al. reproduction. (The Rack re-run
  read FAIL under its registered rule on 2026-10-05, cause not established; see `docs/COORDINATION.md`.
  Rack et al. 2024 is cited only as related work, and the word "reproduced" must never be attached to
  it. If a Rack row is ever added, it carries the Result section's sentence verbatim.)
- The Nymeria row is rank-1 from the registered run of 2026-10-05
  (`docs/acceptance/nymeria_rank1_REGISTERED.md`, Result; read output `nymeria_rank1_read.json`). It is
  always quoted with the control, k (about an hour of other-script enrolment), single 10 s probes and
  one sitting. The leave-script-out row stays as verification AUC; no rank-1 was computed for it.

Every number below is quoted from a registered, gated result. The source file is given in a comment at
the end of each results subsection, for checking; comments are removed before submission.

---

## Title options

1. **Placement, Height, Behaviour: How Far a Head-Motion Biometric Travels Across Applications, Tasks,
   Days and Headsets**
2. **What Head Motion Identifies, and for How Long: Separating Session Cues from Behaviour in XR
   Identification**
3. **Who Is Still You Tomorrow? Persistence and Portability of Head-Only Identification in Extended
   Reality**

---

## Abstract

Head motion in extended reality (XR) identifies the wearer, and published figures suggest it does so
well within a single application and poorly across applications. Those figures share two limits. Nearly
all come from one recording sitting per person, so they cannot say what persists to another day. And a
figure computed on recorded head pose mixes three cues that behave differently: where the headset sat in
the tracking space, how tall the wearer is and how they stand, and how they move. Using head pose alone,
so that the method applies to AR glasses as well as VR headsets, we separate the three with a
training-free three-number baseline and a behaviour-only encoding, and measure how far each travels
across people, applications, tasks, days and headsets on seven corpora. Placement identifies people
within a sitting and is at chance across days. Head height survives a day but not a change of posture.
The learned behavioural signature transfers to people never seen, to applications held out of training,
and to tasks no training identity performed on real AR glasses. It loses a sixth to a third of its
value across a day, depending on the activity, and more across a change of headset. Anchored against the
released model of the reference cross-application study, on its own corpus, test users and metric, a
head-only behavioural model with exposure to the corpus's other participants identifies unseen users
across applications at 0.299 rank-1 against 0.180 (paired +0.119 [+0.050, +0.192], N=17). Two
consequences follow for risk assessment. A behaviour-only figure understates same-sitting risk by the
static term. A same-sitting figure overstates persistent risk. All predictions were registered with
falsifiers before the runs.

---

## 1 Introduction

A head-mounted display has to report its pose many times a second for rendering to work, and that
stream leaves the device by necessity. It is also a biometric: a decade of results shows that how a
person moves in XR identifies them. This is an opportunity for frictionless authentication and a privacy
exposure, because the identifying signal reaches every application the user runs.

The literature reports this exposure as a number, usually rank-1 identification accuracy among N enrolled
users. The most complete recent study, Schach et al. (2026), reports 83.1% within an application and
18.0% across applications among seventeen unseen users. Such a number answers a narrower question than
it appears to. Two properties of the data behind it decide what it can mean.

**First, almost every corpus records each person in one sitting.** Gallery and probe then come from the
same day, the same headset fit and the same spot in the room. A figure computed that way says nothing
about whether the identification would succeed next week, which is the question both an authentication
system and a privacy assessment actually face.

**Second, recorded head pose carries three different cues.** The mean head position of a window encodes
where the headset sat in the tracking space that day (*placement*). Its vertical component also encodes
the wearer's height and posture. The motion around that mean encodes behaviour. The three need not
travel together, and a figure computed on recorded pose cannot say which of them it is reporting.

This paper separates the three and measures how far each travels. Its scope is head motion only, with
no hand controllers, because AR glasses track the head and have no controllers; a method that needs
controller channels cannot run on that device class. We state that scope as a design decision. Where it
costs us against controller-based work, we say so; where we do well despite it, we do not borrow it as a
handicap.

**Contributions.**

1. **A decomposition with a training-free baseline.** A three-number lookup (the window's mean head
   position) matches a trained model on pooled corpora, 0.726 against 0.723 verification AUC. Restricted
   to one axis at a time, it shows the lookup reads placement within a sitting and height across days.
   On the one corpus with a posture change between applications, the height cue goes to chance.
2. **An anchor against the released state of the art.** On the reference study's corpus, test users and
   metric, paired per user against its released model, a head-only behavioural model with exposure to
   the corpus's other participants reaches +0.119 [+0.050, +0.192] rank-1 at N=17. Without exposure the
   comparison is unresolved, and we report it so. A static-cue audit of the same pipeline shows absolute
   pose adds +0.117 inside one sitting after a single training epoch.
3. **Measured persistence.** On two cross-day corpora, the learned signature identifies unseen people
   across days at 0.483 (N=12–17) and 0.693 (N=41), losing 35% and 16% of its same-session value. A
   change of headset costs a further −0.235. On the cross-day corpus the static cue that helped within a
   sitting reverses: absolute pose costs −0.149 across days and collapses to 0.133 across headsets. On
   real AR glasses, within one sitting, it identifies 46 unseen people across activities at 0.555 rank-1
   (N=17) against 0.172 for a model that never saw the device, and across tasks no training identity
   performed,
   at a cost of −0.043.
4. **Consequences for risk assessment.** A behaviour-only figure, such as the reference study's,
   understates same-sitting risk by the static term. A same-sitting figure, which is nearly every
   published one, ours included, overstates persistent risk by a sixth to a third. And the population
   mean conceals a spread across individuals that the reference study's own released per-user values
   show to run from 0.068 to 0.371.

The architecture is a standard recipe and is not a contribution. The contribution is the protocol, the
measurements and the decomposition.

---

## 2 Related work

**Motion identification in XR.** Head and hand trajectories have been shown to identify XR users since
Rogers et al. (2015) and Miller et al. (2020). Nair et al. (2023) identified users among more than 50,000
from head and hand motion in the BOXRR-23 corpus. Rack et al. (2023) released who-is-alyx, seventy-six
players of *Half-Life: Alyx* recorded mostly over two sessions on different days, and reported
cross-session identification under a protocol in which test users are seen during training. Rack et al.
(2024) moved to pretrained similarity learning with disjoint test subjects, the family our model belongs
to.

**Cross-application identification.** Schach et al. (2026) are the first to evaluate identification
across five applications for the same population with a disjoint user split, and they release their
corpus (Across-XR), code, trained model and per-user results. That release is what makes a paired,
per-user comparison possible, and it is the comparison we anchor on. Baldoni et al. (2025) report
within-game identification above 95% and cross-game below 0.30 on Questset, with test users seen during
training and both controllers; it measures a different quantity, and its use here is corroborative: the
cross-application collapse appears even under a more favourable protocol.

**Static cues.** Height is a known contributor to head-based identification, and body-relative
encodings (Rack et al., 2022) discard absolute position by construction. What has not been measured, to
our knowledge, is how the static part divides between placement and height, and how each behaves
across days. Persistence itself has been studied on who-is-alyx with seen users; we measure it on unseen
users and on a second corpus that also changes the headset.

---

## 3 Data

Table 1 lists the seven corpora and the role each plays. Two properties decide what each can say: how
many sittings a person contributes, and whether head position is a real position.

| corpus | people | structure | sittings per person | role here |
| --- | --- | --- | --- | --- |
| BOXRR-23 (Nair et al., 2024) | 4,020 converted | Beat Saber, head track only | many, across days | pretraining |
| who-is-alyx (Rack et al., 2023) | 76 | *Half-Life: Alyx*, mostly 2 sessions | **2, different days** | pretraining; cross-day test (held-out players) |
| Nymeria (Ma et al., 2024) | 236 | **Project Aria AR glasses**, 20 daily-life scripts | 1 | training and in-domain test (48 held out) |
| Across-XR (Schach et al., 2026) | 49 | 5 applications, fully crossed | 1 | cross-application test (users 32–48) |
| Questset (Baldoni et al., 2024) | 60 | 2 of 4 titles per person, two disjoint groups | 1 | posture contrast |
| Ball-throwing (T. A. R. S., 2025) | 41 | 3 headsets × 2 days, 1–30 days apart | **6 sessions on separate days** | cross-day and cross-headset test |
| seated 360° corpora (VR_User_Behavior, Head_and_Gaze) | 48, 100 | video viewing | 1 | static decomposition only |

*Table 1. Corpora. "Sittings" counts recording occasions on different days.*

**Across-XR's user split is read from the corpus.** Its split column reproduces the reference study's
partition exactly: training users 0–22, validation users 23–31, test users 32–48. Every Across-XR figure
in this paper is on users 32–48, and no arm ever trains on them. The exposed arm of Section 5.2 trains on
users 0–22, as the reference model did.

**The cross-day corpora are the only ones that can speak to persistence**, and they are small: who-is-alyx
contributes 12–17 held-out players per checkpoint, ball-throwing 41 people. Ball-throwing ships one
2-second window per throw and an unstated sample rate, which we take as 45 Hz; its headset order was
fixed (Quest, then Vive, then Cosmos), so headset and elapsed time are not fully separable.

**Nymeria is the only AR-glasses corpus we found**, and it records each participant in one sitting. Its
recorded position is a SLAM-map coordinate shared within a sitting rather than a height: mean head height
correlates with participants' measured height at 0.057. We therefore never report a static-cue figure on
Nymeria as a biometric.

**Data statement.** Across-XR is CC BY-NC-SA 4.0, Questset CC BY 4.0, Nymeria CC BY-NC 4.0 and
ball-throwing Apache-2.0. BOXRR-23 is used under a signed Data Use Agreement with the University of
California, Berkeley, with ethics approval in place before use; clause 5 of that agreement requires any
public disclosure to cite Nair et al. (2023), "Unique Identification of 50,000+ Virtual Reality Users
from Head & Hand Motion Data" (arXiv:2302.08927), which we do. Only the head-mounted display track was
extracted from any corpus.

---

## 4 Method

### 4.1 Input and the two encodings

A window is head pose sampled at 20 Hz: the orientation quaternion and the position, seven channels.
Windows are 10 s long with a new one every 5 s, except on ball-throwing, where each throw is one 2 s
window and the model is a 2 s variant of the same arm.

**`dyn` (behaviour only).** Every frame is expressed relative to the window's own mean pose: position
centred on the window mean, orientation relative to the window's mean heading, gravity kept. This removes
placement, height and posture, and is invariant to any rigid transformation of the capture frame.
Anything a `dyn` model identifies is behaviour. The reference study's encoding is static-free by a
different route, so its figures and our `dyn` figures are comparable in the sense that matters.

**`raw` (pose as recorded).** Identical pipeline, absolute head pose kept. The difference `raw − dyn`, on
the same backbone, objective, windows and users, is our static-cue audit.

### 4.2 Model, enrolment and matching

The backbone is a bidirectional LSTM producing a 128-dimensional embedding, trained with an additive
angular-margin softmax over training identities (margin 0.35, scale 30). The classifier is discarded and
embeddings are compared by cosine. Training uses per-dataset normalisation fitted on training users
only, cross-session positives where a corpus has them, a 25% validation-user draw for epoch selection,
up to 120 epochs and a patience of 15.

For identification, each enrolled user's **template** is the renormalised mean of their gallery window
embeddings; a probe window is assigned to the nearest template by cosine, ties rank-averaged. We report
rank-1 accuracy and always quote N, because chance is 1/N.

### 4.3 The training-free baseline

For every evaluation we also compute a **three-number lookup**: each window is reduced to its mean head
position, standardised per corpus, and windows are compared by Euclidean distance. It needs no training.
Restricted to the vertical axis it reads height and posture; restricted to the two horizontal axes it
reads placement. For `dyn` models, whose input has no mean position, the corresponding baseline is
movement amplitude alone. These baselines are reported beside model figures, because a model figure that
a lookup matches is not evidence of learning.

### 4.4 Uncertainty, registration and gating

All intervals are cluster bootstraps over **users** (10,000 resamples), seeds averaged within a user
first, and every contrast is paired on the same users. With seventeen test users the binomial standard
deviation of a rank-1 near 0.18 is about 0.09, so only paired contrasts can resolve the differences we
report.

Every quantity in Section 5 was registered before its run, with a band, a falsifier and a stated
meaning for the region between them. A band is settled by where the confidence interval falls, not by a
significance test. Every checkpoint is **gated** before it is scored: rescored through the pipeline's own
loader, it must reproduce its own recorded score (tolerance 10⁻⁴ on GPU). Numbers come from three seeds
unless stated.

### 4.5 Comparing against a released model

We compare against the reference study's **released model**, not a reimplementation. Running their own
accuracy calculator on their released embeddings reproduces every per-user value in all thirty-five
(gallery, probe) application cells at a maximum absolute difference of 0.0, including their published
0.180 cross-application mean. Their decision rule (nearest reference window, 15 s windows at 30 fps)
differs from ours (mean template, 10 s at 20 Hz), so we score in both directions: our embeddings through
their calculator, and theirs through our harness. The two directions agree on every verdict.

**Which of their figures we compare against.** In their within-application evaluation the reference
windows are a subset of the query windows from the same unbroken recording, sampled every 25 s from 15 s
windows, so every within-application query shares frames with some reference window. This is a
consequence of the corpus holding one recording per person and application, not an error, and it does
not affect their cross-application figure, whose reference and query sets come from different
applications. We therefore compare only against 0.180, and never pair our within-application figure with
their 83.1%. One consequence for the field is that the reported drop from 83.1% to 18.0% overstates the
cross-application collapse by whatever the overlap is worth.

---

## 5 Results

Table 2 is the organising object of this section: each row is a boundary between gallery (or training)
and probe (or test), each column a component. Sections 5.1–5.4 take it row by row.

| boundary | corpus | metric, N | learned (`dyn`) | static (`raw` / lookup) |
| --- | --- | --- | --- | --- |
| unseen people, same sitting, seated video | VR_User_Behavior, Head_and_Gaze | lookup AUC | — | placement carries the lookup (xz 0.70, 0.87) |
| unseen people, other activity, AR glasses | Nymeria | rank-1, N=17; k ≈ 1 h other-script, single 10 s probe | **0.555**; control 0.172 | lookup reads the shared map, not the person |
| unseen people **and unseen tasks**, AR glasses | Nymeria | AUC, 25 users | 0.615; −0.043 against seen tasks | as above |
| unseen people, **unseen application**, one sitting | Across-XR | rank-1, N=17 | zero-shot 0.234; exposed **0.375** (0.299 on the reference metric vs 0.180) | `raw` **+0.117** at epoch 1 |
| application held out of training | Across-XR | rank-1, N=17 | −0.016 [−0.043, +0.014] vs exposure to all five | — |
| **another day**, same headset | who-is-alyx | rank-1, N=12–17 | **0.483** vs 0.743 same day (−35%) | lateral lookup at chance (0.539 AUC); height holds (0.661 AUC) |
| **another day**, same headset | ball-throwing | rank-1, N=41 | **0.693** vs 0.824 same session (−16%) | `raw` −0.149 vs `dyn` |
| **another headset**, another day | ball-throwing | rank-1, N=41 | **0.458** (−0.235 vs same headset) | `raw` collapses to 0.133 |

*Table 2. What travels across which boundary. Rank-1 chance is 1/N. AUC rows are verification
(chance 0.50) and are not comparable in level to rank-1 rows. All rows three seeds. Every rank-1 row
uses one decision rule (A1): a template is the renormalised mean of a person's gallery window
embeddings, a probe window goes to the nearest template by cosine, and a tie counts as a miss.
Enrolment amount differs by row and is stated beside it: one application on Across-XR, one session on
who-is-alyx, five throws on ball-throwing, and all of a person's other-script windows (about an hour) on
Nymeria. The Nymeria rank-1 is therefore not set against the Across-XR rows without that caveat.*

`[FIGURE 1: Table 2 drawn as a two-column chart, one row per boundary ordered from "same sitting" to
"another headset", learned and static components side by side, each as a fraction of its own
same-session value where one exists. Data: this table and docs/PAPER_THESES.md.]`

### 5.1 What three numbers already identify

On the pooled seated and room-scale corpora, the three-number lookup scores 0.726 verification AUC on
held-out users against 0.723 for a trained model on the same folds and pairs (five leave-users-out
folds). Most of what a pooled head-pose model does needs no model.

Restricting the lookup to one axis shows what it reads (Table 3).

| corpus | sittings | xyz | height (y) | placement (xz) |
| --- | --- | --- | --- | --- |
| Head_and_Gaze | 1 | 0.870 | 0.690 | **0.872** |
| VR_User_Behavior | 1 | 0.719 | 0.640 | **0.700** |
| BOXRR-23, held-out users | many | 0.763 | **0.810** | 0.680 |
| who-is-alyx | **2, different days** | 0.593 | **0.661** | 0.539 |

*Table 3. Training-free lookup, verification AUC on held-out users, by axis. Three manifest seeds.*

On the single-sitting corpora the horizontal coordinates carry the whole lookup, and adding height adds
nothing. Participants occupy distinguishable spots in a shared rig: their own session means sit 0.20 m
apart against 0.40 m between participants, and every participant's sessions are one sitting, so where
the seat and tracking origin were placed that day is a per-person constant. On who-is-alyx, the only
corpus in the table whose sessions fall on different days, placement collapses to near chance (0.539)
while height holds (0.661). Measured as a distance ratio, a person's own head position on another day is
0.95 of the distance to a stranger's horizontally and 0.49 vertically. **Placement belongs to the
sitting; height survives a day.**

**Height survives a change of application only when posture is shared.** Across-XR's five applications
are all played standing, and there a participant's per-application mean heights are closer to each other
than to other participants' with probability 0.754, while horizontal placement is at chance (0.527): the
games move people differently and scramble placement themselves. Questset separates the two because its
groups differ in exactly that property (Table 4).

| Questset group | applications | placement P | height P | participants whose height changes > 0.20 m |
| --- | --- | --- | --- | --- |
| 1 | Beat Saber / Cooking Simulator (both standing) | 0.526 | **0.718** | 0 / 30 |
| 2 | Medal of Honor / Forklift Simulator (standing vs **seated**) | 0.549 | **0.493** | **30 / 30** |

*Table 4. P = probability that a participant's two per-application mean positions are closer than two
different participants'. 0.5 is chance.*

A training-free height lookup agrees: it identifies group 2 at exactly chance (0.033 at N=30) and group 1
above chance at 0.092. The registered level for group 1 was 0.15 and was not reached; the cue is present
but weaker than predicted. Wherever this paper says height survives an application change, the posture
qualifier is part of the claim.

<!-- sources: CLAUDE.md "A three-number lookup matches the trained model", per-axis and co-location
tables, distance ratio; questset_geometry.json, questset_static_lookup.json; Across-XR geometry in
CLAUDE.md Across-XR section. -->

### 5.2 Across applications: the anchor against the released model

A model trained on 3,072 identities from BOXRR-23 and who-is-alyx, never on Across-XR, identifies the
seventeen test users across applications at **0.234 [0.182, 0.292]** rank-1 (chance 0.059). Adding the
reference study's own twenty-three training users (3.87% of training windows) raises it to **0.375
[0.321, 0.433]**. Paired per user against the released model (Table 5):

| contrast, paired over the 17 test users | measured | 95% CI | users better | outcome |
| --- | --- | --- | --- | --- |
| zero-shot − reference, reference metric | +0.025 (0.206 vs 0.180) | [−0.031, +0.080] | 10/17 | unresolved |
| **exposed − reference, reference metric** | **+0.119** (0.299 vs 0.180) | **[+0.050, +0.192]** | **15/17** | **better** |
| exposed − reference, our metric | +0.176 (0.375 vs 0.199) | [+0.092, +0.260] | 16/17 | better |
| exposed − reference, ten-minute sequence, reference metric | +0.355 | [+0.202, +0.499] | — | better (secondary) |

*Table 5. Cross-application rank-1 at N=17, single 10 s window unless stated. The reference model uses
the head and both controllers and 15 s windows.*

The exposed contrast was registered before it ran (band +0.08 to +0.20 on the reference metric) and
landed inside it. The zero-shot contrast is unresolved under both metrics; the supportable sentence is
that every zero-shot seed sits above the reference mean, head-only and without having seen the corpus,
and we make no ranking claim from it.

**The gain is pretraining combined with exposure, not either alone.** Our model trained on the twenty-three
training users only, the reference study's own protocol, reaches 0.131 [0.088, 0.177], below the
reference model.

**Exposure carries to an application held out of training.** At 3,072 identities, a model exposed to
four of the five applications and never to the fifth identifies on that fifth application within −0.016
[−0.043, +0.014] of one exposed to all five, pooled over the five choices; the registered falsifier, a
cost below −0.05, is excluded. The held-out application is the deployment-realistic cell, and the
reference design does not test it.

**The static-cue audit.** Replacing `dyn` with `raw` on the zero-shot arm raises cross-application rank-1
from 0.234 to **0.351**, a paired **+0.117 [+0.042, +0.192]**, and every `raw` seed selected **epoch 1 of
16**. A model one epoch from initialisation reaches within a seed spread of the trained, exposed `dyn`
model. Since all five applications are played standing and placement is scrambled across them
(Section 5.1), this static term is height and posture rather than placement, and it is available without
learning anything. The reference study's encoding discards it by design, so their figure assesses
behavioural risk only. Inside one sitting, a behaviour-only assessment understates the total risk by
about this much. Section 5.4 shows that this term does not persist to another day.

<!-- sources: across_xr_alignment_RESULTS.md (zero-shot, C2-lo, C1, P2, schach_paired.json);
exposure_breadth_REGISTERED.md Amendment 5 (hold-out cost three seeds). -->

### 5.3 Across tasks on AR glasses

Nymeria records people on Project Aria glasses doing twenty daily-life scripts in one sitting. We trained
the same recipe with 141 Nymeria identities among 3,072 (the treatment) and, as control, with every
non-held-out Nymeria identity swapped out for BOXRR-23 at the same count, and scored both on the same 48
Nymeria people neither saw.

The evaluation denies the model the activity cue. For each script, the gallery is only the people who
recorded it; every template, the true person's and every impostor's, is built from that person's
*other* scripts, and the probe is a single 10 s window of the script. Enrolment is therefore all of a
person's other-script windows, about an hour, which is far more than one Across-XR application, and the
Nymeria figure is not set against the Across-XR rows without that caveat. We average over
(person, script) cells, which reads exactly at chance for an embedding that encodes only the activity.

At N=17 (chance 0.059, 115 cells over 46 people), the treatment identifies unseen people at **0.555
[0.508, 0.599]** and the control at **0.172 [0.138, 0.207]**, a paired gain of **+0.383
[+0.335, +0.429]**. With all 48 people as candidates the figures are 0.553 and 0.197 (+0.356
[+0.311, +0.402]). A model that has seen other people on the device identifies unseen people by how they
move, across activities, at more than nine times chance from a single 10 s window.

**The control is not at chance, and that is a finding.** A model that never saw the device identifies
across activities at about three times chance, which agrees with its zero-shot 0.234 on Across-XR. Its
verification figure on the same people, 0.472, reads below chance only because that test's negatives are
same-script pairs: a model whose scores partly encode activity rates two strangers doing the same thing
as more alike than one person doing two things. So that verification figure measures the pairing, not
an absence of person signal.

**Against the registration.** The registered band for the treatment was 0.20–0.45 and for the control
0.03–0.08, both derived from the verification AUC of 0.67. The treatment's AUC on this score set is
0.88, because each template averages about an hour of enrolment, and the usual non-Gaussian offset of
about +0.09 accounts for the rest; the control shows the same offset. The treatment lands above its band
and is not credited as exceeding it, because the control's falsifier fired. The gain spans the band and
the region above it, so its size relative to +0.38 is not resolved.

**It is not tied to the tasks it was trained on.** Removing five scripts (a quarter of the sequences)
from every training identity and scoring 25 unseen people doing only those five scripts, the model reads
0.615 against the full treatment's 0.657 on the same people, a cost of **−0.043 [−0.056, −0.030]**, inside
the registered band of −0.06 to 0. The control reads 0.506 on this set.

Three limits travel with this row. Nymeria records one sitting per person, so this is a same-session
figure and the persistence costs of Section 5.4 apply to it. Within that sitting, the device and the
location are constant per person: the protocol removes the activity cue, not the sitting, and how much
of the 0.555 the sitting carries is unmeasured. And Nymeria training does not carry beyond the
device's own population measurably: on ball-throwing, a control without Nymeria matches the treatment
within +0.002 [−0.017, +0.020].

<!-- sources: nymeria_rank1_REGISTERED.md Result and nymeria_rank1_read.json (rank-1, six gates at 0.0);
nymeria_in_domain_REGISTERED.md (constrained verification protocol, 3 seeds);
nymeria_lso_REGISTERED.md; broad_2s_REGISTERED.md Q1. -->

### 5.4 Across days and headsets

**who-is-alyx.** The Nymeria treatment identifies held-out alyx players across a gap of days at **0.483
[0.415, 0.550]** rank-1 (N=12–17, chance about 0.07), against 0.743 within one session: a cost of
**−0.261 [−0.345, −0.182]**, or 35% of the same-session value. The registered band for a modest cost
(≥ −0.15) is excluded. Day and session are confounded on this corpus; for deployment that confound does
not matter.

**Ball-throwing.** Each window is one 2 s throw; the model is the same treatment retrained at 2 s.
Rank-1 at N=41 (chance 0.024), Table 6:

| condition | `dyn` | `raw` |
| --- | --- | --- |
| C0 same session | 0.824 [0.784, 0.857] | 0.916 |
| **C1 same headset, another day** | **0.693 [0.649, 0.735]** | 0.545 |
| **C2 another headset, another day** | **0.458 [0.409, 0.509]** | 0.133 |
| day cost C1 − C0 | **−0.131 [−0.172, −0.089]** | |
| headset cost C2 − C1 | **−0.235 [−0.282, −0.191]** | |
| `raw − dyn` | | C0 +0.092 [+0.062, +0.125]; C1 **−0.149 [−0.210, −0.090]** |

*Table 6. Ball-throwing, three seeds. C1 pairs are 1–7 days apart, C2 pairs 1–30 days.*

The learned signature persists across days on a second corpus: from a single 2 s throw, people are
identified among 41 at about 28 times chance on another day, losing 16% of the same-session value. The
day cost is half alyx's, and it is the activity rather than the window length that differs: a 2 s model
on alyx keeps 53% of its same-session value, against 65% at 10 s and 84% on ball-throwing. One
repetitive throw is more stable from day to day than free locomotion.

**A headset change costs more than a day.** Identity survives it well above chance (0.458, 19 times
chance), but loses −0.235, or −0.142 [−0.206, −0.082] when C2 pairs are restricted to sessions at most
three days apart. The mechanism is not established. One account is that headset fit shifts the wearer's
absolute tilt, which `dyn` keeps; a training-free tilt lookup does identify across days (0.152) and loses
a third of that across headsets, but an encoding that removes absolute tilt keeps the same proportional
headset cost (0.622 against 0.661). We report the cost and not a cause.

**The static cue does not persist.** A `raw` model is better within a session (+0.092), worse across days
on the same headset (−0.149), and collapses across headsets to 0.133. Inside one sitting on Across-XR the
same substitution added +0.117. On a cross-day corpus it subtracts. What absolute pose gives a model is a
property of the session. One qualification: every `raw` seed selected epoch 2, so part of its cross-day
deficit may be behaviour it never learned. Either reading gives the same conclusion for this paper: on a
cross-day corpus a behaviour-only model does not understate what a pose model identifies.

<!-- sources: alyx_cross_day_REGISTERED.md Result 2 (GPU); ballthrowing_cross_day_REGISTERED.md Result;
broad_2s_REGISTERED.md three-seed Result and Q3b. -->

### 5.5 The population mean conceals the individual

Every rank-1 above is a mean over people who differ a great deal. The argument is strongest on the
reference study's own released values. Their per-user cross-application rank-1 runs from **0.068 to
0.371** on a single window and from **0.043 to 0.818** over a ten-minute sequence, on the same seventeen
people: one participant is identified four-fifths of the time behind a population figure of 0.31. Our arms
show the same spread (per-user standard deviation 0.069 zero-shot and 0.110 exposed, against 0.090 for
the reference model).

"The model identifies users at 18%" and "a user has an 18% chance of being identified" are different
claims, and only the first is supported by a mean. A risk assessment concerns the most exposed person, so
it needs the distribution. On a large in-domain population we found the split stable: the users nearly
always identified and those nearly never identified remain the same people when training identities grow
five-fold, so the spread is a property of the people rather than a transitional state that more data
removes.

`[FIGURE 2: per-user cross-application rank-1 for the 17 test users, reference model and our two arms,
population means dashed, chance at 1/17. Exists as docs/acceptance/schach_per_user.png.]`

<!-- sources: PAPER_PLAN per-user section, schach_paired.json; CLAUDE.md "The offset repeats everywhere
measured" (BOXRR in-domain stability, two seeds per arm). -->

---

## 6 Discussion

**Three components, three distances.** Table 2 reads as a single pattern. Placement identifies people
well and only within a sitting: it carries the lookup on single-sitting corpora and is at chance across
days and across applications. Height and posture survive a day and an application change that keeps
posture, and are destroyed by one that does not. The learned behavioural signature is the only component
that transfers to people, applications and tasks the model never saw, and it is also the one that
persists, though not intact: it keeps 65–84% of its same-session value across a day and less across a
change of headset.

**What this means for a risk assessment.** Two errors are available, in opposite directions, and the
published literature commits one or the other. A behaviour-only figure, computed on an encoding that
discards absolute pose, understates what a recipient of the raw stream can learn within one sitting: on
Across-XR, by about +0.117 of rank-1, available without training. A same-sitting figure overstates what
persists: by a sixth to a third of its value across a day, more across a headset, and, for the static
part, almost entirely. Neither figure is wrong. Each answers a narrower question than the one it is
usually taken to answer, and an assessment should state which session boundary it crosses.

**What does and does not move transfer.** Exposure to other people in the target application family is
the only data-side lever we measured to cross an application boundary. More training identities of the
same activity raised in-domain identification sharply and moved cross-corpus transfer by 0.001, and
swapping identities for a genuinely different activity at a fixed count moved it by −0.0012
[−0.0045, +0.0020]. Nymeria training buys the AR-glasses population and not, measurably, a new corpus.

**On devices.** Every corpus except Nymeria is VR. The head-only scope means the method runs on AR
glasses, and Section 5.3 shows that it identifies unseen people across tasks on them, but no cross-day
AR-glasses corpus exists that we know of. Whether the AR-glasses figure persists to another day is the
most direct open question this paper leaves.

---

## 7 Limitations

- **Small test populations.** Seventeen Across-XR test users, 12–17 alyx players per checkpoint, 41
  ball-throwers. User-level uncertainty dominates every interval, and differences around ±0.05 are not
  resolvable on the smaller ones.
- **Persistence rests on two corpora**, neither on AR glasses. The ball-throwing figures come from a 2 s
  model that is weaker in domain (0.55 AUC against 0.708 at 10 s), at an assumed 45 Hz, with a fixed
  headset order.
- **One fully crossed cross-application corpus.** Across-XR is the only one we know of; Questset is two
  two-application corpora. Both are one sitting.
- **One external reference.** Every claim of improvement is made against one released system. Our
  decision rule is not theirs; we score both directions and they agree, but they are not the same rule.
- **The headline arm was chosen after the first registration.** The registered prediction targeted the
  zero-shot arm; the exposed arm's contrast was registered before it ran, but the decision to lead with
  it was taken later.
- **The AR-glasses contrast is not perfectly symmetric.** The Nymeria treatment selects its epoch
  partly on Nymeria validation users and the control cannot, and the 48 held-out people are one draw.
  The protocol removes the activity cue but not the sitting: device and location are constant per
  person, and their share of the rank-1 is unmeasured. Its registered bands were missed in both arms
  (Section 5.3).
- **The headset cost has no measured mechanism**, and the `raw` arms selected their epoch at 1 or 2, so
  the static audit's cross-day reading carries a convergence confound.
- **The architecture is not novel.** This is a measurement paper.

---

## 8 Conclusion

A head-motion identification figure mixes where the headset sat, how tall the wearer is, and how they
move, and the three travel different distances. Placement ends with the sitting. Height survives a day
but not a change of posture. Behaviour transfers to unseen people, applications and tasks, including on
AR glasses, and persists across days at a cost of a sixth to a third, more across a headset. Anchored
against the released state of the art on its own corpus, a head-only behavioural model with exposure to
the corpus's other participants identifies unseen users across applications better than a model that also
uses both hand controllers. The practical consequence is a reporting rule: every identification figure
should state which session boundary it crosses, carry a training-free baseline beside it, and report
the distribution over people rather than the mean alone.

---

## References

1. L. Schach, C. Rack, R. P. McMahan, M. E. Latoschik. *Motion-Based User Identification across XR and
   Metaverse Applications by Deep Classification and Similarity Learning.* Frontiers in Virtual Reality,
   2026. doi:10.3389/frvir.2026.1743491. Dataset (Across-XR), CC BY-NC-SA 4.0; code, data and released
   models at gitlab.informatik.uni-wuerzburg.de.
2. V. Nair, W. Guo, J. Mattern, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song. *Unique Identification of
   50,000+ Virtual Reality Users from Head & Hand Motion Data.* USENIX Security, 2023. arXiv:2302.08927.
   **Required by clause 5 of the BOXRR-23 Data Use Agreement.**
3. V. Nair et al. *Berkeley Open Extended Reality Recordings 2023 (BOXRR-23).* IEEE TVCG, 2024.
   doi:10.1109/TVCG.2024.3372087.
4. C. Rack, T. Fernando, M. Yalcin, A. Hotho, M. E. Latoschik. *Who is Alyx? A new behavioral biometric
   dataset for user identification in XR.* Frontiers in Virtual Reality, 2023.
   doi:10.3389/frvir.2023.1272234.
5. C. Rack, K. Kobs, T. Fernando, A. Hotho, M. E. Latoschik. *Versatile User Identification in Extended
   Reality using Pretrained Similarity-Learning.* arXiv:2302.07517, 2024.
6. C. Rack, A. Hotho, M. E. Latoschik. *Comparison of Data Encodings and Machine Learning Architectures
   for User Identification on Arbitrary Motion Sequences.* IEEE AIVR, 2022.
7. L. Ma et al. *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild.* ECCV,
   2024. arXiv:2406.09905.
8. S. Baldoni, S. Benhamadi, F. Chiariotti, M. Zorzi, F. Battisti. *Questset: A VR Dataset for Network
   and QoE Studies.* ACM MMSys, 2024. doi:10.1145/3625468.3652187.
9. S. Baldoni et al. *Movement- and Traffic-based User Identification in Commercial Virtual Reality
   Applications: Threats and Opportunities.* arXiv:2501.16326, 2025.
10. Terascale All-sensing Research Studio, Wright State University. *Multimodal cross-system VR ball
    throwing dataset for VR biometrics.* Data in Brief, 2025.
    **[AUTHOR LIST AND VOLUME TO BE COMPLETED.]**
11. F. Wang, J. Cheng, W. Liu, H. Liu. *Additive Margin Softmax for Face Verification.* IEEE Signal
    Processing Letters, 2018. **[VERIFY.]**
12. M. R. Miller et al. *Personal identifiability of user tracking data during observation of 360-degree
    VR video.* Scientific Reports, 2020. **[VERIFY.]**
13. K. Rogers et al. Identification from head and hand motion, 2015. **[TO BE COMPLETED.]**
14. C. Wu, Z. Tan, Z. Wang, S. Yang. *A Dataset for Exploring User Behaviors in VR Spherical Video
    Streaming.* ACM MMSys, 2017. doi:10.1145/3083187.3083210. *(VR_User_Behavior.)*
15. Y. Jin, J. Liu, F. Wang, S. Cui. *Where Are You Looking? A Large-Scale Dataset of Head and Gaze
    Behavior for 360-Degree Videos and a Pilot Study.* ACM Multimedia, 2022, pp. 1025–1034.
    *(Head_and_Gaze.)*
