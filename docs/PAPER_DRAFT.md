# Paper draft

**Status: complete first draft.** Every number is taken from `docs/PROGRESS_REPORT.md` and the
certificates it cites; where two internal sources disagreed, the later corrected value was used.
Figure slots are marked `[FIGURE n: ...]` and reference the chart specifications in
`docs/PROGRESS_REPORT.md` Part C. One figure already exists as a file
(`docs/acceptance/schach_per_user.png`).

---

## Title options

1. **Head Motion Alone Identifies XR Users Across Applications: A Re-Assessment of Cross-Application
   Biometric Risk**
2. **Behaviour Travels With Exposure: Head-Only Identification Across XR Applications, and What a
   Behaviour-Only Risk Assessment Misses**
3. **Who Are You in the Next Application? A Static-Cue Audit of Cross-Application Identification in
   Extended Reality**

---

## Abstract

Motion-based identification in extended reality is reported as close to solved within a single
application and as collapsing across applications. The reference study for this problem reports 83.1% rank-1
identification within an application and 18.0% across applications among seventeen unseen users
(chance 5.88%), using head motion together with both hand controllers. We re-assess that
cross-application risk on the same corpus, the same user split and the same metric, using **head
motion only** — a scope chosen so that the method applies to AR glasses, which track the head and
have no hand controllers. We compare against the reference study's *released model*, whose published
per-user values our harness reproduces exactly (35 cells, maximum absolute difference 0.0), and we
pair every contrast user by user with a cluster bootstrap over the seventeen test users. A head-only
model pretrained on 3,072 identities from other corpora and then trained on the reference study's own
twenty-three training users reaches 0.299 cross-application rank-1 at N=17 under their metric against
their 0.180, a paired difference of **+0.119 [+0.050, +0.192]**, better on fifteen of seventeen users;
under our own template metric the same contrast is **+0.176 [+0.092, +0.260]** (0.375 against 0.199),
better on sixteen of seventeen. A model that never sees any data from the corpus is **unresolved**
against the reference (+0.025 [−0.031, +0.080]) and is not reported as a beat. Three further results
qualify the risk picture rather than the ranking. First, a static-cue audit of our own headline: a
model that keeps absolute head pose reaches 0.351 cross-application **after a single training epoch**,
+0.117 [+0.042, +0.192] above our behaviour-only figure, so a behaviour-only assessment understates
the risk. Second, the orthogonal embedding alignment the reference study proposes as future work does
not carry when fitted honestly on training users (+0.011 [−0.020, +0.041] against a registered band of
+0.05 to +0.20), and we identify why: the corpus offers at most thirty-two people recorded in two or
more applications to fit on. Third, cross-application rank-1 varies across individuals far more than
its population mean suggests — in the reference study's own released per-user values it runs from
0.068 to 0.371 on a single window and from 0.043 to 0.818 over ten minutes, on the same seventeen
people. Predictions were registered with falsifiers before the runs and negative outcomes are reported
alongside positive ones.

---

## 1 Introduction

Extended reality (XR) systems cannot function without streaming motion. A head-mounted display must
report its pose many times per second for rendering to track the wearer's head, and that stream leaves
the device by necessity rather than by choice. The same signal that makes the display work is a
biometric: a decade of results shows that how a person moves in XR identifies them. This is
simultaneously an opportunity — continuous, frictionless authentication without a password or a
fingerprint sensor — and a privacy exposure, because the identifying signal is emitted to every
application the user runs and to whatever infrastructure carries it.

The state of the art within a single application is strong enough that the interesting question has
moved elsewhere. Schach, Rack, McMahan and Latoschik (2026) report 83.1% rank-1 identification among
seventeen unseen users within an application, and **18.0% across applications** — gallery drawn from
one application, probe from another, chance 5.88% at a gallery of seventeen. Their own abstract is
explicit that the ability to identify users "across different XR applications remains limited". That
reported collapse, from near-solved to roughly three times chance, is the gap this paper addresses, and
18.0% is the number against which everything here is measured. We show in Section 5.4 that the drop
overstates the collapse — their within-application figure contains self-matches and their
cross-application figure does not — so the two are not on the same footing, and we never pair our own
within-application number against 83.1%.

**Scope: head motion only.** We use head orientation and head position and nothing else. This is a
design decision about which device class the method must cover, not a concession. Every published
cross-application result we are aware of, including the reference study's, uses the head together with
both hand controllers; the reference study's eighteen per-frame features are four head-rotation
features and fourteen from the two controllers. AR glasses track the head and have no hand
controllers, so a model that needs controller channels cannot run on them at all. Choosing head-only
therefore buys coverage of the whole device class, and it has a direct consequence for how the results
below should be read: where we win, we win against a system with more sensors and longer windows —
10 s against 15 s. We state that asymmetry where we win and never borrow it as an excuse where we do
not.

**Two properties of the reference design shape what is worth measuring.** The first is that its
"cross-application" cell holds unseen *users* of applications that were seen during training: the
model is trained on all five applications and evaluated on held-out people, with gallery and probe
drawn from different applications. The cell in which both the user *and* the application are unseen is
never tested, and it is the deployment-realistic one — a new application appears that was not in the
training distribution. The second is that the reference encoding removes head position by
construction: body-relative referencing fixes the head-mounted display at its own origin and the
velocity encoding then differentiates, so absolute head pose never reaches their model. Their figures
are consequently *not* inflated by placement or anthropometry. They are, by the same token, a
measurement of *behavioural* risk specifically, and the total risk a wearer faces is the behavioural
part plus whatever static anthropometry and posture contribute. Nothing in their method can measure
the second term.

**Contributions.**

1. **A head-only beat of the reference study's released model on its own corpus, people and metric,
   under a sensor handicap.** With pretraining on other corpora plus the reference study's own
   twenty-three training users, +0.119 [+0.050, +0.192] under their metric and +0.176 [+0.092, +0.260]
   under ours, at N=17 on their test users 32–48 (Section 6.2). The zero-shot arm, which never sees the
   corpus, is **unresolved** against them and is reported as such (Section 6.1).
2. **A static-cue audit of our own headline.** Keeping absolute head pose adds +0.117 [+0.042, +0.192]
   cross-application, and every seed reaches that after one training epoch. A behaviour-only risk
   assessment therefore understates the risk (Section 6.6).
3. **The unseen-application cell the reference design never tests, and a measured carry across it.**
   Leave-one-application-out training still carries +0.049 [+0.021, +0.078] pooled over five held-out
   applications, including applications absent from every pretraining corpus (Section 6.4). The
   stricter registered threshold for this claim was not met, and we say so.
4. **A negative answer, with a mechanism, to the reference study's own proposed fix.** Train-user-only
   orthogonal alignment of application embedding spaces does not carry on a static-free encoding. The
   fitted map is **person-specific**: it helps, when it helps at all, only the people it was fitted on.
   The corpus offers at most thirty-two participants outside the evaluation split to fit on, a number no
   amount of pretraining raises; whether *more* would make the fit transfer is not settled here
   (Section 6.7).
5. **The per-user distribution as a metric argument**, made first on the reference study's own released
   per-user arrays and then on ours: the population mean conceals individuals identified far more often
   than it reports (Section 6.9).
6. **Methodological.** Predictions registered with falsifiers before the runs and read against
   confidence intervals rather than p-values; every checkpoint gated against its own recorded score
   before any new number is taken from it; the comparison made against a released model rather than a
   reimplementation; and negative outcomes reported as prominently as positive ones.

`[FIGURE 1: headline paired comparison — cross-application rank-1 at N=17 on a single window, two
panels (their nearest-reference metric; our template metric), three systems each (their released
model, our zero-shot arm, our exposed arm), 95% CI whiskers, chance line at 0.0588. Chart spec C1.]`

---

## 2 Related work

**Motion identification in XR.** The observation that XR motion identifies its source predates the
current generation of hardware; Rogers et al. (2015) and Miller et al. (2020) established that head
and hand trajectories carry identity, and Miller et al. (2021) extended the setting to matching users
across systems with a Siamese network. Rack et al. (2023) contributed both a dataset and a protocol:
"Who is Alyx?" records seventy-six players of *Half-Life: Alyx* over mostly two sessions on different
days, and reports cross-session identification of users *seen* during training in the high seventies.
Nair et al. (2023) demonstrated the scale of the exposure, identifying users from among more than
50,000 in the Berkeley Open Extended Reality Recordings corpus from head and hand motion alone.

**Pretrained similarity learning.** The move from per-corpus classifiers to embeddings trained with
metric-learning objectives is what makes generalisation to unseen users measurable at all. Rack et al.
(2024) train a similarity model on a large motion corpus and transfer it to new populations, which is
the family our own model belongs to, and which the reference study also adopts.

**Encodings.** Rack et al. (2022) introduced the body-relative family — SR, BR, BRV, BRA — and reported
the acceleration variant best in their setting. Their ordering was obtained with a full body frame
derived from the head *and* both controllers. A head-only rig cannot construct that frame, and we
measure the ordering to invert in our setting: on pooled head-only corpora the raw encoding beats the
best body-relative alternative by a wide margin, because the body-relative encodings remove the
absolute pose that carries most of a head-only signal and return nothing constructed from the hands in
its place. This is one reason our behaviour-only encoding is defined differently (Section 4.2) rather
than adopted from that family.

**Cross-application identification.** Baldoni et al. (2025) report roughly 30% identification across
two applications with a classification model. The reference study for this paper, Schach et al. (2026),
is the first to evaluate identification across five applications for the same population, with a
disjoint user split, and it is the work we compare against throughout. A further study on the Questset
corpus (Baldoni et al., 2025b) reports above 95% within-game identification and no model above 0.30
cross-game at a gallery of thirty. That figure is not a competitor to ours and is not treated as one:
its test users are *seen* during training (first eight minutes of a recording for training, the next
two for testing, within each participant) and it uses the head plus both controllers, so it measures
recognition of a known enrolled user rather than generalisation to people never seen. Its usefulness
here is corroborative — a protocol more favourable than ours in two ways at once, from a group with no
stake in our framing, and the cross-application collapse still happens.

**Embedding alignment.** The reference study's section 6.2.5 is the point of departure for one of our
results. They find that the embedding spaces learned for different applications appear to differ by an
orthogonal transformation — rotation and reflection, without scaling or translation — and that aligning
them lifts cross-application accuracy from 18.0% to 52.3% on a single window and from 30.8% to 94.3%
over ten minutes. They then disqualify their own result, because the transformations were fitted on
the *test* users: they describe it as a post hoc diagnostic upper bound rather than a deployable,
generalising solution, and they name the honest version — fit the transformations on training and
validation users, then apply them to unseen test users and applications — as future work. We run that
honest version in Section 6.7. Note that this material appears only in the journal version of their
paper; the preprint does not contain it. Their released evaluation code contains an
`orthogonal_procrustes`-based multi-domain alignment, which corroborates the method independently of
the text.

---

## 3 Datasets

### 3.1 Across-XR: the evaluation instrument

Across-XR (Schach et al., 2026; CC BY-NC-SA 4.0) records **49 participants in five applications**:
Superhot VR, Half-Life: Alyx, Beat Saber, Synth Riders, and a Social VR scenario. All 245
(participant, application) cells are present — the corpus is **fully crossed**, which is what allows a
cross-application contrast in which the person, the headset and the room are fixed and only the
application changes. Each session is ten to fifteen minutes, recorded at 90.9 Hz on one headset model,
with positions in centimetres and a y-up convention that matches our pipeline, so no axis remapping is
required.

**The user split is read from the corpus, not approximated.** Our converted copy carries the split
column, and it reproduces the reference study's partition digit-exactly.

| split | user IDs | count | role in this paper |
| --- | --- | --- | --- |
| train | 0–22 | 23 | training users for the exposed arms — **exactly the reference study's training users** |
| validation | 23–31 | 9 | epoch selection for the corpus-only arms; alignment fitting population |
| **test** | **32–48** | **17** | **every number reported at N=17 is on these users, and no arm ever trains on them** |

This table is placed early deliberately. The obvious objection to any result obtained with exposure to
Across-XR is that we trained on the corpus we are evaluated on. The exposure is to participants
**0–22** and the evaluation is on participants **32–48**; the reference study's own model trained on
0–22 as well. The arms are matched on that axis, not advantaged, and the disjointness is a property of
the split file rather than of our bookkeeping.

**Structural facts that bound what the corpus can support.** Each (participant, application) cell is a
single unbroken recording: the corpus carries a take identifier, but it is a redundant relabelling of
the application identifier and is constant across every row of every file, so there is **no second take
anywhere**. The five applications were played in sequence in one sitting, ten to fifteen minutes each
plus breaks, so a cross-application pair carries real temporal separation of up to roughly an hour, in
an order that is the same for every participant — but never a different day. Two consequences follow.
The corpus can say nothing about persistence across days, which we list as a limitation in Section 8.
And a within-application evaluation on this corpus has no way to separate gallery from probe across
recordings, which is the structural point we return to in Section 5.4.

**Static geometry of the corpus.** Because our second contribution is an audit of static cues, we
measured them on the corpus itself before running any model. Taking each participant's mean head
position per application, the probability that a participant's own two per-application means are closer
together than two different participants' is 0.527 for the lateral coordinates — chance — and **0.754
for height**. Within a single application the lateral figure is 0.7525. The reading is that the games
move people differently: a fifteen-minute mean position records where the game makes a player stand
rather than where the rig sits, so lateral placement is scrambled by the application change itself,
while head height survives. All five applications are played standing by the population — per-game mean
head heights run 1.533 m to 1.606 m, per-person span across the five has a median of 0.068 m, and two
of forty-nine participants shift by more than 0.20 m. The qualifier this makes necessary is developed
in Section 6.6.1.

### 3.2 Pretraining corpora

**BOXRR-23** (Nair et al., 2024) supplies the bulk of our pretraining identities: 4,020 users converted
from the release, **head track only**, all Beat Saber. We verified on the full release index that no
user in BOXRR-23 is recorded in two applications, which is relevant beyond our own pipeline — it means
the largest XR motion corpus in existence cannot support a cross-application study, and corpora with
the structure Across-XR has are genuinely scarce.

**who-is-alyx** (Rack et al., 2023) supplies 76 players of Half-Life: Alyx across 146 sessions, mostly
two sessions on different days. It is the only corpus in our pretraining set with cross-day structure.

Two of the five Across-XR applications — Beat Saber and Half-Life: Alyx — are therefore pretraining
*activities*, with different people, different rigs and different capture pipelines. Superhot VR, Synth
Riders and the Social VR scenario appear in no pretraining corpus of ours. This is an asymmetry we
control for explicitly in Section 6.4 rather than hope away.

### 3.3 Questset: a second corpus and a second gallery size

Questset (Baldoni et al., 2024; CC BY 4.0) records **60 complete participants** (70 recruited, 10
withdrew to cybersickness) in four commercial titles, **two per participant by group**: group 1 plays
Beat Saber and Cooking Simulator, group 2 plays Medal of Honor: Above and Beyond and Forklift
Simulator. It is **not fully crossed** — the groups are disjoint, so it is two two-application corpora
of about thirty participants each, and Across-XR remains the only fully crossed cross-application
corpus we know of. Each participant is recorded in one sitting, one session per game, so Questset adds
no temporal persistence either.

Its value here is twofold. It supplies a **second gallery size**: every interval in this paper is
limited by seventeen test users, and Questset allows the same static-cue measurements at N=30 as well
as at a matched N=17. And, because its two groups differ in a property Across-XR holds constant, it
scopes one of our claims — see Section 6.6.1.

### 3.4 Data statement

Across-XR is distributed under CC BY-NC-SA 4.0. Questset is distributed under CC BY 4.0 and the
attribution travels with our converted copy. BOXRR-23 is used under a signed Data Use Agreement with
the University of California, Berkeley, with institutional ethics approval in place in advance of use
as that agreement requires; **clause 5 of the agreement requires that any public disclosure cite Nair
et al., "Unique Identification of 50,000+ Virtual Reality Users from Head & Hand Motion Data"
(arXiv:2302.08927)**, which we do, in addition to the BOXRR-23 dataset paper. Only the head-mounted
display track was ever extracted from BOXRR-23.

`[TABLE 1: dataset summary — corpus, participants, applications, sessions per participant, sampling
rate, licence/terms, role. Data: PROGRESS_REPORT D.1.]`

---

## 4 Method

### 4.1 Input

A window is ten seconds of head pose sampled at 20 Hz, with a new window every five seconds. Each frame
contributes seven channels: the head orientation quaternion (x, y, z, w) and the head position
(x, y, z). There are no controller channels, no hand channels and no gaze channels, by the scope
decision stated in Section 1.

### 4.2 The `dyn` encoding

Our headline encoding, `dyn`, expresses every frame of a window relative to the window's **own mean
pose**, keeping gravity. Position is centred on the window mean and orientation is taken relative to
the window's mean heading. This removes height, seating position and placement in the tracking space,
and it is invariant to any rigid transformation of the capture frame — so a figure produced under `dyn`
cannot be a measurement of where the headset sat that day, in which room, or on how tall a person.
Anything that survives it is behaviour.

The contrast we audit against is `raw`: pose as recorded, absolute head position included. `dyn` and
`raw` are the same pipeline with the same backbone, objective, windows and protocol, differing only in
this transform, which is what makes the audit in Section 6.6 a clean subtraction.

The reference study's BRV encoding is static-free by a different route — the head is fixed at its own
origin and the result is differentiated — so their figures and our `dyn` figures are comparable in the
sense that matters: neither can be read from a static cue. This is the reason the headline of this
paper is a `dyn` figure, and that choice was fixed before any `raw` number existed.

### 4.3 Model and objective

The backbone is a bidirectional LSTM producing a 128-dimensional embedding, trained with an additive
angular-margin softmax objective (AM-Softmax, margin 0.35, scale 30) over training identities. The
classifier head is discarded after training and embeddings are compared by cosine similarity. Training
uses per-dataset normalisation with statistics fitted on training users only, within-dataset negatives,
cross-session positives, a 25% validation-user draw for epoch selection, 120 epochs and a patience of
fifteen.

**The architecture is not novel and we do not present it as one.** A BiLSTM trained with an
angular-margin identity loss is a standard recipe from speaker and face verification. The contribution
of this paper is the protocol, the measurements and the negative results; a reader looking for a
modelling advance will not find one here, and framing it as one would invite exactly the wrong review.

### 4.4 Enrolment and matching

For each enrolled user we form a **template**: the renormalised mean of all of that user's embedding
vectors from the gallery application. A probe is a single ten-second window from a different
application. The prediction is the nearest template by cosine similarity among the N=17 test users, and
the reported figure is rank-1 accuracy.

### 4.5 Training arms

The arms differ only in what identities are in the training set. Identity counts are the number of
identities the loader actually held after the validation draw, never the pool before it.

| arm | training data | Across-XR exposure | trained identities | dose (share of training windows) | seeds |
| --- | --- | --- | --- | --- | --- |
| **zero-shot** | BOXRR-23 + who-is-alyx | none | 3,072 | 0 | 3 |
| Z-676 | BOXRR-23 (600 users) + who-is-alyx | none | 495 | 0 | 1 |
| C2-hi | Z-676 minus 23 BOXRR users, plus Across-XR 0–22 | yes | 495 | 14.1% | 1 |
| **C2-lo** | zero-shot corpus + Across-XR 0–22 | yes | 3,095 | 3.87% | 3 |
| C2-lo-half | as C2-lo, first half of each Across-XR session | yes | 3,095 | 1.96% | 1 |
| C1 / C1-full | Across-XR 0–22 only (the reference study's own protocol) | yes | 23 | 100% | 1 each |
| P3 × 5 | C2-hi composition with one application removed | four of five | 495 | ~11.6% | 1 (two applications: 2) |
| `raw` variants | as zero-shot / C2-lo, `encoding=raw` | as base | as base | as base | 3 / 1 |

**The composition is provable rather than asserted.** The loaders' own window counts close exactly:
C2-lo held 540,107 windows, of which 20,896 are Across-XR, and 540,107 − 20,896 = **519,211**, which is
the zero-shot arm's training set to the window. The treatment arm is therefore the control's corpus
plus Across-XR users 0–22 and nothing else.

`[FIGURE 2: method pipeline — head pose (7 channels) → 10 s windows at 20 Hz, stride 5 s → dyn encoding
→ BiLSTM + AM-Softmax → 128-d embedding → template matching at N=17, with a side panel giving the
reference study's corresponding choices (head + both controllers, BRV, 15 s at 30 fps, Transformer+GRU,
480-d, nearest reference window). Chart spec C12.]`

`[TABLE 2: protocol comparison — sensors, encoding, window length, sample rate, architecture, embedding
size, decision rule, training data, for the reference study and for us. Data: PROGRESS_REPORT slide 8,
C12.]`

---

## 5 Evaluation protocol

### 5.1 Primary and secondary metrics

**The primary metric is rank-1 identification accuracy on a single window at N=17**, on test users
32–48, with the gallery drawn from one application and the probe from another, averaged over the twenty
ordered off-diagonal application pairs. Chance is 1/17 = 0.0588 and is quoted with every figure, as is
N: rank-1 at a gallery of seventeen and rank-1 at a gallery of thirty are different questions.

**The secondary metric is accuracy over a ten-minute probe sequence**, a majority vote over the probe
windows of the first 600 seconds, giving one decision per user per cell. It is reported *beside* the
primary metric and never in place of it. This ordering was fixed before any result existed, and it is
worth stating why we did not revisit it: the ten-minute metric separates our exposed arm from the
reference study far more cleanly than the single-window metric does, and promoting the metric that
happened to give the better result, after seeing which one did, would be moving the goalposts by
another route. It is also the metric that saturates within an application — the reference study's
within-application ten-minute figure is 1.0000 for all seventeen users — so it can only separate methods
across applications.

### 5.2 Comparing against a released model, in both directions

We compare against the reference study's **released model**, not a reimplementation. Their published
evaluation artefacts contain per-user `precision_at_1` arrays for all seventeen test users in each of
thirty-five cells. Running their own accuracy calculator verbatim on their own released embeddings
reproduces **every per-user value in all thirty-five cells at a maximum absolute difference of 0.0**,
recovering their published 0.8314 within-application and 0.1804 cross-application means exactly. This
is the referent for every comparison below.

The correspondence between array index and user identity was **reconstructed rather than assumed**. Each
user's five-application window-count vector is unique across the seventeen test users, which pins index
*k* to user 32+*k* independently of any assumption about ordering; the label remapping and the
per-class averaging in their metric library both sort ascending, which agrees; and we verified
exhaustively that no class was dropped for insufficient samples, since a dropped class would shorten an
array and break the correspondence invisibly. In total, 463,996 embeddings were accounted for across
eighty-five (user, application) cells.

Matching the metric is not the same as matching the decision rule. Theirs is nearest-reference-window
matching on 15 s windows at 30 fps; ours is a mean template on 10 s windows at 20 Hz. We therefore
score **both directions with a single harness each**: D1 puts our embeddings through their calculator,
and D2 puts their embeddings through our template harness. Both directions are reported for every
headline contrast, and they agree on every verdict.

### 5.3 Uncertainty and pairing

All uncertainty is a cluster bootstrap over the **seventeen test users** with 10,000 resamples, seeds
averaged inside each user before resampling, and every contrast paired on the same users. The effective
sample size for this problem is users, not windows: at a rank-1 near 0.18 with seventeen users the
binomial standard deviation is √(0.18 × 0.82 / 17) = **0.093**, so a single unpaired split cannot
resolve a difference below about 0.09. That arithmetic was done before the runs, and it is why a claim
of beating the reference study had to be established paired across users and seeds rather than by
comparing two point estimates. The twenty application pairs share the same seventeen users and are
therefore *not* twenty independent samples; we read them as quasi-replications inside one corpus.

Registered bands are read against the confidence interval, not against a significance test. An interval
can fail to exclude zero while excluding the entire hypothesis it was built to test, and reporting only
a test in that case turns a decisive negative into an inconclusive one.

### 5.4 A structural note on the reference study's within-application figure

The reference study's within-application figure of 83.1% and its cross-application figure of 18.0% are
not on the same footing as each other, and we state this once, structurally, because it bears on which
of their numbers we compare against.

In their within-application evaluation, the reference set for a cell is every 150th embedding of the
query application and the query set is every embedding of that same application, drawn from the same
unbroken recording, with the nearest-neighbour search not told that the reference is a subset of the
queries. Their reference embeddings are computed at a stride of five frames from 450-frame windows, so
references sit 750 frames apart while a window spans 450: **every within-application query window
shares frames with some reference window** except at a recording's edges. Graded, 0.67% of queries are
the identical vector at distance zero and 59% share at least half their frames.

**This is a property of the corpus, not an error.** Across-XR holds exactly one recording per
(user, application) cell, so a within-application evaluation has no second take from which to draw a
gallery; their design cannot avoid the overlap. **Their cross-application figure of 18.0% is entirely
unaffected** — the reference and query sets come from different applications by construction — and it is
the figure every comparison in this paper is made against. The one consequence for us is that we never
pair our own within-application number against their 83.1%, and that the drop from 83.1 to 18.0
overstates the cross-application collapse by whatever the overlap is worth.

### 5.5 Registration and gating

Predictions P1–P3 were registered with bands and falsifiers, dated, before any cross-application run;
arm-level bands and seven subsequent amendments are recorded in the same way. Amendments are kept in
place rather than edited away, and the original registration text is preserved beside them, so that the
history of every band is auditable rather than asserted. One amendment rule was applied throughout: a
registration may be amended for a fact about the instrument that could have been known without running
the experiment, and never for a measurement.

Every checkpoint is **gated** before any new number is taken from it: rescored through the pipeline's
own loader, it must reproduce its own recorded verification score. Twenty-three gates passed, with gaps
from 5.3 × 10⁻⁸ to 2.9 × 10⁻⁴. Verification scores appear only as gate referents and never beside a
rank-1 figure, since verification (two classes, chance 0.50) and identification (chance 1/N) are
different quantities. The alignment code of Section 6.7 additionally passes a fixture gate on synthetic
rotated embeddings whose correct answer is known by construction.

`[TABLE 3: registered predictions and verdicts — prediction, registered band and falsifier, measured
value with interval, verdict. Chart spec C14.]`

`[POPULATE TABLE 3 FROM docs/PAPER_OUTLINE.md (R5) AND across_xr_alignment_RESULTS.md — NOT from
PAPER_PLAN's summary table. P2 in particular: it has a registered band (+0.00..+0.06, falsifier
< -0.03, Amendment 6) and the measurement +0.117 [+0.042, +0.192] sits ABOVE it, so the verdict is
"falsifier excluded, size unresolved", not "held". Every row of this table must carry its interval
beside its band — a band is settled by where the interval falls, not by whether a point estimate is
inside it, and this table is where a reader checks that we followed our own rule.]`

---

## 6 Results

### 6.1 Head-only, zero-shot: a placement, not a beat

A model trained on 3,072 identities from BOXRR-23 and who-is-alyx, and never on any Across-XR data,
reaches **0.234 [0.182, 0.292]** cross-application rank-1 at N=17 over three gated seeds
(0.231 / 0.230 / 0.240) under our template metric, against chance 0.0588. Its within-application figure
is 0.500 [0.462, 0.538] and its ten-minute cross-application figure is 0.357 [0.257, 0.474]. This was
registered in advance with a band of 0.18 to 0.35 and a falsifier below 0.12; it **held**, inside the
band. Scored through the reference study's own calculator (direction D1) the same arm reads
**0.206 [0.175, 0.240]** against their 0.180 [0.140, 0.225].

**Paired against their released model on the same seventeen users, this contrast is unresolved:**
+0.025 [−0.031, +0.080] under their metric, better on ten of seventeen users, and +0.035
[−0.050, +0.122] under ours, again ten of seventeen. Over ten minutes it is +0.106 [−0.039, +0.256] and
+0.069 [−0.091, +0.240], unresolved under both. Their own per-user interval reaches 0.225, and our
zero-shot interval overlaps it heavily; with seventeen users a difference of +0.05 is not resolvable,
and no amount of framing changes that.

**The one sentence this supports is that every zero-shot seed sits above their reported mean, head-only
and never having seen the corpus.** We do not call it a beat and we do not present it as one. A model
that has seen neither these people nor these applications placing above a controller-based published
mean is a result about how much of cross-application identification is available from the head alone;
it is not a ranking claim, and treating it as one would be the same error, with the sign reversed, as
arguing away a near-miss that went against us.

### 6.2 Exposure on the reference study's own training users: the headline

Adding Across-XR training users 0–22 to the same pretraining corpus — 3,095 trained identities, a dose
of 3.87% of training windows — raises cross-application rank-1 to **0.375 [0.321, 0.433]** over three
seeds (range 0.010) under our metric and **0.299 [0.249, 0.354]** under theirs. Paired per user against
their released model:

| contrast (paired per user; cluster bootstrap over the 17 test users) | measured | 95% CI | users better | outcome |
| --- | --- | --- | --- | --- |
| zero-shot − theirs, **their** metric | +0.025 (0.206 vs 0.180) | [−0.031, +0.080] | 10/17 | UNRESOLVED |
| **C2-lo − theirs, their metric** | **+0.119** (0.299 vs 0.180) | **[+0.050, +0.192]** | **15/17** | **BEAT** |
| zero-shot − theirs, **our** metric | +0.035 (0.234 vs 0.199) | [−0.050, +0.122] | 10/17 | UNRESOLVED |
| **C2-lo − theirs, our metric** | **+0.176** (0.375 vs 0.199) | **[+0.092, +0.260]** | **16/17** | **BEAT** |

The result holds in both directions of the harness, on a single window, against a system that uses the
head **plus both hand controllers** and 15 s windows against our 10 s. The contrast was registered
before it ran, at +0.08 to +0.20 under their metric and +0.05 to +0.17 under ours; the first landed
inside its band and the second landed 0.006 past its upper edge, which we declare rather than argue
either way.

**The secondary metric agrees, and more widely.** Over a ten-minute probe sequence the exposed arm
reads 0.663 [0.571, 0.753] under their sequence metric against their 0.308 [0.208, 0.420], a paired
+0.355 [+0.202, +0.499]; under our vote it reads 0.711 [0.635, 0.785] against their 0.288
[0.191, 0.394], a paired +0.423 [+0.293, +0.546]. Their interval tops out at 0.420 and ours begins at
0.571. The agreement between the two metrics is what matters here rather than the size of the second:
they could have disagreed, and they did not.

**The control that keeps this honest.** Our model trained on the reference study's twenty-three
training users *alone* — their protocol, our model, no pretraining — reaches 0.131 [0.088, 0.177], and
in its budget-matched form 0.164 [0.128, 0.205]. Both sit **below** their 0.180. The gain is therefore
not our model applied to their data; it is pretraining on other corpora **combined with** exposure to
the corpus's other participants, and neither ingredient produces it alone.

`[FIGURE 3: ten-minute secondary comparison — three systems under each of the two metrics with CI
whiskers. Chart spec C3.]`

### 6.3 Per application pair

Among the twenty ordered application pairs, the exposed arm beats the reference study **resolvably in
eleven and loses none**; the zero-shot arm beats it resolvably in three and loses none — which is
precisely why its overall contrast remains unresolved, and the three should not be quoted as zero-shot
beating the reference study. The two largest gains in both of our arms are the rhythm-game pair in both
directions: Synth Riders → Beat Saber reads 0.475 [0.401, 0.547] zero-shot and 0.591 [0.536, 0.654]
exposed; Beat Saber → Synth Riders reads 0.432 [0.344, 0.522] and 0.546 [0.472, 0.620]. That affinity
is already present in the arm that never saw Across-XR, so it belongs to the activity pair rather than
being created by exposure. Because the twenty cells share the same seventeen users, they are
quasi-replications rather than independent samples.

`[FIGURE 4: 5×5 per-cell heatmaps under their metric — their released model, our zero-shot arm, our
exposed arm, shared colour scale, diagonal hatched as within-application. Chart spec C4a.]`

`[FIGURE 5: forest plot of the twenty per-cell paired differences, exposed arm minus theirs, with 95%
CIs and a zero line. Chart spec C4b.]`

### 6.4 Exposure carries to an application held out of training

The reference design never tests the cell in which both the user and the application are unseen. We do.
Holding one application out of training entirely and evaluating the eight cells that involve it, exposure
to the remaining four applications still carries **+0.049 [+0.021, +0.078]** over the matched unexposed
control, pooled over the five choices of held-out application.

| held-out application | in our pretraining? | seeds | gain over unexposed control | 95% CI |
| --- | --- | --- | --- | --- |
| Superhot VR | no | 1 | +0.034 | [−0.004, +0.070] |
| Half-Life: Alyx | yes | 1 | +0.026 | [−0.004, +0.059] |
| Beat Saber | yes | 1 | +0.084 | [+0.044, +0.122] |
| Synth Riders | **no** | 2 | +0.065 | [+0.034, +0.095] |
| Social VR | no | 2 | +0.038 | [+0.001, +0.073] |
| **pooled** | — | — | **+0.049** | **[+0.021, +0.078]** |
| uncovered subset | no | — | +0.046 | [+0.017, +0.074] |
| covered subset | yes | — | +0.055 | [+0.024, +0.089] |

**The coverage control is what makes this stand.** If the carry were pretraining leaking through the
hold-out, applications absent from every pretraining corpus would show none of it. They read +0.046
against +0.055 for the two that are covered, and Synth Riders — which appears in no pretraining corpus
of ours at all — carries at +0.065 [+0.034, +0.095] over two seeds.

**The stricter registered threshold was not met, and we report that.** The falsifier for this
prediction was a pooled gain at or below zero, and it is excluded. But a second, stricter condition was
registered for the phrase "crosses an activity boundary" — that the lower edge of the interval exceed
+0.030 — and the lower edge is +0.021. The carry is real; the stronger wording is not earned.

A related registered prediction also held: the unseen-application cell sits **below** the
seen-application cell, −0.036 [−0.054, −0.018]. Training on all five applications was buying something,
which is worth establishing rather than assuming. This contrast is **dose-confounded and was registered
as such**: the held-out-application arm also trains on 20% less Across-XR data, so two explanations are
sufficient for the gap. The registered control separates them — on the *seen* cells the same pair reads
−0.009 [−0.025, +0.008], so the 20% cut costs nothing measurable and the deficit is attributable to the
held-out application rather than to dose. We report the confound and its control together because the
contrast is not clean on its own.

`[FIGURE 6: forest plot of leave-one-application-out gains, per application, pooled, and
covered/uncovered subsets, with the +0.030 registered threshold drawn and labelled NOT MET. Chart spec
C6.]`

### 6.5 Identity count interacts with exposure, and dose does not explain it

Pretraining identity count is the one data-side lever that has repeatedly been measured to help within a
domain. Across this domain boundary it is flat **without** exposure: reducing the pretraining pool from
3,072 to 495 trained identities changes cross-application rank-1 by −0.013 [−0.039, +0.013]. **With**
exposure it is not flat: the 495-identity exposed arm loses to the 3,095-identity exposed arm by −0.061
[−0.099, −0.026], and the 495-identity exposed arm gains +0.089 [+0.048, +0.131] over its own unexposed
control. Identity count and exposure are therefore not two independent additive levers; the second is
what makes the first pay across an application boundary.

**Dose cannot account for the pair, and correcting for it widens the effect rather than narrowing it.**
Halving the in-domain windows at fixed identities costs −0.028 [−0.062, +0.009] and a 20% cut costs
−0.009 [−0.025, +0.008], so dose is a small and roughly linear term — and the arm that *loses* by 0.061
is the one with the **higher** dose (14.1% against 3.87%).

`[FIGURE 7: arm chart — cross-application rank-1 by training composition with CI whiskers, the reference
study's 0.180 drawn as a solid line and chance as a dashed one, dose annotated per arm. Chart spec C5.]`

`[TABLE 4: all arms — composition, exposure, trained identities, dose, seeds, within-application,
cross-application with CI, ten-minute. Chart spec C5.]`

### 6.6 A static-cue audit of our own headline

This project's methodological habit has been to audit static cues out of other people's numbers. We ran
the same audit on our own, and it is the second contribution of this paper.

Replacing `dyn` with `raw` — keeping absolute head pose, changing nothing else — raises zero-shot
cross-application rank-1 from 0.234 to **0.351** (seeds 0.364 / 0.353 / 0.335), a paired **+0.117
[+0.042, +0.192]**. Within an application the same substitution is worth +0.223 [+0.184, +0.263]
(0.723 against 0.500), which is larger, and which is what carries the within-application placement cue
this corpus has (lateral P(within<between) 0.7525 within an application against 0.527 across
applications). The within-application `raw` cell is therefore not quoted as a biometric figure.

**The finding is not the size of the gain; it is the epoch at which it appears.** Every `raw` seed
selected **epoch 1 of 16** — patience fired immediately. A model one epoch from initialisation reaches
0.351 cross-application from the head alone. The cue is not something a model must learn; it sits on the
surface of the input. That is worse news than a trained model reaching the same place. The `raw` arm's
verification score against a training-free recorded-position lookup (0.704–0.733 against 0.585–0.598)
says it is reading more than mean position alone: posture as well as height.

**The privacy reading, and it lands on the reference study's own framing rather than contradicting it.**
Their paper is explicitly a risk assessment, and their encoding discards head position by construction,
so what it assesses is **behavioural** risk. On this corpus, static anthropometry and posture —
available after one epoch, with no behaviour required — add +0.117 head-only across applications and
reach 0.351, within a seed spread of the 0.375 that a trained, exposed behavioural model reaches. This
does not say their numbers are inflated; they are not, and we state that explicitly. It says that a
behaviour-only assessment **understates the total risk**, which is a finding their framing asks for and
their method cannot produce.

**With exposure the static advantage is unresolved, and over time it reverses.** The exposed `raw` arm
reads 0.404 against the exposed `dyn` arm's 0.375, a difference of +0.029 [−0.068, +0.134] on one seed.
Over ten minutes the behavioural model leads outright: 0.711 against 0.497. Averaging more evidence
lifts a learned per-window signal, whose error is variance, and cannot lift a static cue, whose error is
a per-recording bias.

**The headline stays on `dyn`, and that was decided before any `raw` number existed.** A `raw`
comparison against a BRV-based model would win partly on a cue their encoding discards on purpose, which
would not be a like-for-like comparison. The audit sits beside the headline rather than in a footnote,
because running it and burying it would be indefensible in a paper whose contribution is partly the
auditing of static cues.

`[FIGURE 8: static-cue audit — paired dyn vs raw bars for cross-application single window, within
application, and cross-application with exposure, with per-seed dots and the "epoch 1 of 16"
annotation. Chart spec C7.]`

#### 6.6.1 The surviving static cue is head height — across applications that share a posture

Across-XR's five applications scramble lateral placement (P = 0.527, chance) and preserve head height
(P = 0.754), which is why the `raw` gain above should be read as anthropometry rather than as a rig
artefact. That reading needs a qualifier, and a second corpus supplies it. Questset's two groups differ
in exactly the property Across-XR holds constant: group 1's two titles are both played standing, while
group 2 pairs a standing title with a **seated** driving simulator.

| group | applications | lateral P | height P | median height change | participants moving > 0.20 m |
| --- | --- | --- | --- | --- | --- |
| 1 | Beat Saber / Cooking Simulator (both standing) | 0.526 | **0.718** | 0.051 m | 0 / 30 |
| 2 | Medal of Honor / Forklift Simulator (standing vs seated) | 0.549 | **0.493** (chance) | 0.438 m | **30 / 30** |

Head height survives a change of application only **across applications that share a posture**: all
thirty group-2 participants drop about 0.44 m, and the cue is destroyed, while it is nearly perfectly
preserved across the group-1 pair. A training-free height-only lookup agrees, reading exactly chance for
group 2 at both gallery sizes (0.059 at N=17, 0.033 at N=30) against 0.153 and 0.092 for group 1; the
registered prediction for this comparison held for group 2 as predicted and, for group 1, landed above
chance but **below** its registered level, which we report as such. Lateral placement is at chance in
both groups, replicating the Across-XR figure on a second corpus. Wherever this paper says head height
is the static cue that survives an application change, the posture qualifier is part of the claim rather
than a caveat on it.

### 6.7 The reference study's proposed fix does not carry, and the reason is the corpus

The reference study reports that application embedding spaces differ by an orthogonal transformation and
that aligning them lifts cross-application accuracy from 18.0% to 52.3%, then disqualifies that result
because the transformations were fitted on the test users, and names the honest version as future work.
We ran the honest version on our own embeddings: fit the orthogonal map on users 0–31 — the training and
validation users their paper names — and apply it to the unseen test users.

| variant | arm | gain over the unaligned model | 95% CI | registered expectation |
| --- | --- | --- | --- | --- |
| fitted on the **test** users (diagnostic ceiling) | zero-shot, 3 seeds | +0.026 | [+0.000, +0.051] | ≥ +0.15 |
| **fitted on users 0–31 (honest)** | zero-shot, 3 seeds | **+0.011** | **[−0.020, +0.041]** | band +0.05 to +0.20 |
| permuted correspondences (null) | zero-shot | −0.074 | [−0.126, −0.029] | — |
| unrestricted 128-d fit | zero-shot | −0.055 (vs the honest fit) | [−0.086, −0.029] | — |
| fitted on users 0–31 (honest) | exposed, 3 seeds | −0.003 | [−0.011, +0.005] | — |
| reference study, fitted on test users | published | +0.34 (0.180 → 0.523) | — | — |

**The honest fit does not resolvably carry on `dyn`, the headline encoding.** The registered band of
+0.05 to +0.20 is excluded. The test-fitted ceiling on our own embedding — their illegitimate route,
reproduced — is +0.026 [+0.000, +0.051], **thirteen times smaller** than the +0.34 they measure, and it
is itself run-dependent: at identical configuration on the exposed arm, three seeds read +0.148, −0.004
and +0.001. A single-run diagnostic bound of that kind is not evidence that application embeddings
differ by a rotation, which raises the evidential bar for every claim of this shape — including the one
this experiment set out to reproduce. We do **not** claim their +0.34 is wrong; we report that our
embedding does not have that structure to recover.

**The scope of the negative matters.** On the `raw` encoding, one seed reads +0.032 [+0.007, +0.057],
whole interval above zero. That is consistent with the encoding rather than a counter-example — a `raw`
embedding retains a static frame that an orthogonal map can genuinely rotate — but it means the negative
is a `dyn` result and must be stated as one.

**The mechanism is the corpus, and it is the actionable half.** The permuted-correspondence null *hurts*
by −0.074, so the orthogonal component our fit recovers is real and person-specific; it is simply tiny.
The honest fit fails not because the fit is rank-deficient but because of the **test-fitted advantage**:
the ceiling variant fits on the seventeen people it is then scored on and reaches +0.148, while the
honest variant fits on thirty-two *other* people and reaches −0.008 on the same arm and the same seed.
That +0.148 is **one run of three at identical configuration** (the other two read −0.004 and +0.016,
as reported above), so it is a single-run bound and not a rate; the contrast below rests on the *sign*
of the gap between the two variants, which holds on every seed, and not on the magnitude of the
ceiling. More correspondences did not help,
because the correspondences are not the same people. (The rank argument is real and belongs to a
different variant: the unrestricted 128-d fit, which must invent an arbitrary 96-dimensional complement
from at most thirty-two correspondences, loses −0.055 [−0.086, −0.029] in six of six checkpoints. It is
not the reason the honest route fails, and conflating the two attaches a wrong mechanism to a right
conclusion.)

**So the recommendation to the field follows from the honest fit's own reason.** The correspondences
available for fitting an alignment are bounded by the number of people recorded in two or more
applications — thirty-two here, and the reference study had the same thirty-two — and no amount of
pretraining raises that number. The honest answer to their future-work proposal is not "we failed to
make it work" but **"the map we fitted on thirty-two people did not transfer to seventeen others"**.
Whether a larger fitting population would cure that is **not established by this experiment**, and we
decline to present it as a design target: the failure we measured is one of transfer *across people*,
not of having too few of them, so more participants might or might not help.

`[FIGURE 9: alignment variants — dot-and-whisker of each variant's gain over the unaligned model, with
zero and the reference study's published +0.34 drawn as reference lines. Chart spec C13.]`

### 6.8 Two further registered outcomes that went against us

**A registered mechanism failed.** We predicted, from an enrolment-averaging argument, that forming mean
templates over the reference study's embeddings would lift their model into 0.20–0.32 under our metric.
It lifted them by **+0.019 only** (0.180 → 0.199), missing the band's lower edge by 0.001. We do not
argue the edge, in the same way we do not argue a 0.001 edge that falls in our favour. The useful half
is the mechanism: template averaging buys our embeddings +0.028 (zero-shot) and +0.076 (exposed) and
buys their nearest-reference embedding almost nothing, so **the averaging gain is model-specific** —
which was found by registering a prediction that then failed.

**A hyperparameter gain reversed at scale.** An angular-margin setting measured to be worth +0.016 at
419 training identities costs **−0.028 [−0.045, −0.012]** in cross-application rank-1 at the
4,096-identity scale, with the within-application and ten-minute figures agreeing in sign. Only the sign
reversal is claimed, because the two measurements are different metrics at different scales. The
mechanism was named in the registration in advance: margins tuned for corpora with tens of thousands of
identities push too hard at four hundred, and that explanation predicts the reversal at four thousand.

For completeness, an earlier experiment in this programme tested whether **activity diversity** in the
pretraining corpus improves cross-domain transfer, by swapping fifty identities of one activity for
fifty of a genuinely different one at a fixed identity count. It measured −0.0012 [−0.0045, +0.0020]
against a registered band of +0.005 to +0.03 — the entire band above the entire interval. Together with
the flat identity-count result of Section 6.5, this is the context in which exposure stands out: it is
the only data-side lever we have measured to cross an application boundary at all.

### 6.9 The per-user distribution

Every rank-1 figure in this paper, and in the reference study, is a population mean over seventeen
people who differ a great deal. **The distribution argument can be made on the reference study's own
released values**, which is stronger than making it on ours alone: their per-user cross-application
rank-1 runs from **0.068 to 0.371** on a single window and from **0.043 to 0.818** over ten minutes, on
the same seventeen people. One participant is identified four-fifths of the time behind a population
figure of 0.31. A risk assessment is a claim about the most exposed person, not about the average one,
and "the model identifies users at 18%" and "a user has an 18% chance of being identified" are different
claims of which only the first is supported by a mean.

The same holds of our own arms. Per-user spread (standard deviation across the seventeen) is 0.090 for
the reference model, 0.069 for our zero-shot arm and 0.110 for our exposed arm, over per-user ranges of
0.068–0.371, 0.105–0.363 and 0.148–0.528 respectively.

**An exploratory observation, clearly labelled as one.** The two systems' per-user *orderings* are not
strongly correlated: Spearman ρ = −0.010 between the reference model and our exposed arm under their
metric, 95% CI [−0.49, +0.47], permutation p = 0.97. This is not an artefact of a flat distribution on
either side — neither is flat — and it is not noise in the per-user values themselves, because the
internal controls are high: our own two arms agree at ρ = +0.767 [+0.45, +0.91], and each of our arms
scored under the two different metrics agrees with itself at +0.939 and +0.926. As an example, user 32 is
the reference model's worst case (0.068, barely above chance) and among our best (0.295 zero-shot, 0.487
exposed).

Three qualifications travel with this and none of them is optional. It was **not registered in advance**.
At seventeen users the Fisher interval on a Spearman correlation is about ±0.5 wide, so the supportable
phrase is **"not strongly correlated"** — it excludes the 0.77–0.94 that our internal controls show, and
it cannot exclude a moderate correlation. And the **cause cannot be attributed**: the two systems differ
in sensor set, architecture and encoding simultaneously, and this corpus cannot separate them. The
agreement of +0.767 between our own two arms does exclude *training exposure* as the scrambling factor,
since those arms share architecture, encoding and sensors, but the remaining three move together.

What survives is two ordering claims, and they are the ones worth stating: **a per-person risk audit
conducted under one system does not necessarily transfer to another**, so measuring oneself safe under a
particular model is weak evidence of safety; and **a defence that protects the top-k most identifiable
users has no stable target**, because the list of who those are changes with the system. We tested a
coverage corollary — that running both systems would expose materially more people than either alone —
and it **failed**: our exposed arm already catches all but two of the seventeen users better than the
reference model does, so the union buys at most one additional person at any threshold and none at the
high ones. The disagreement is in the ordering, not in the coverage, and we withdraw the coverage
reading.

`[FIGURE 10: per-user cross-application rank-1, seventeen test users, two panels (their metric; our
metric), three dots per user (their released model, our zero-shot arm, our exposed arm), population
means dashed, chance at 1/17. This figure already exists as `docs/acceptance/schach_per_user.png`.]`

---

## 7 Discussion

**What transfers across an application boundary, and what does not.** The clearest pattern in these
results is a negative one with a positive exception. More pretraining identities of the same activity do
not help across an application boundary — the difference between 495 and 3,072 pretraining identities is
−0.013 [−0.039, +0.013] without exposure — and greater *activity* diversity in the pretraining corpus did
not help either, when tested at a fixed identity count. What does carry is **exposure**: having seen
other people perform the applications in question, even at a dose of under 4% of training windows, and
even when the specific application at evaluation was held out of training entirely. Identity count then
becomes a lever again, which is why the two must be reported as interacting rather than as additive. We
state this as the first data-side lever *we* have measured to cross an application boundary, not as a
field-wide claim: the mechanism by which exposure helps is not established here, and a plausible reading
— that it adapts the representation to a family of activity structures rather than to the specific
people or the specific application — is a hypothesis this corpus cannot test.

**Task structure matters more than task count.** The two rhythm games transfer to each other at roughly
twice the mean cross-application rate, in both directions, in the arm that never saw Across-XR at all.
This says the affinity is a property of the activity pair rather than something exposure creates, and it
suggests that "cross-application" is not one quantity: a gallery collected in a rhythm game generalises
to another rhythm game unusually well. We stop there deliberately. The wider claim — that
cross-application transfer is *ordered* by task structure — was registered and **failed at the bottom**:
the application that carries least is Half-Life: Alyx at +0.026, an activity our pretraining covers,
while the Social VR scenario we predicted would carry least sits mid-table (Section 6.4). So the
rhythm-pair affinity is established and a general task-structure ordering is not, and "poorly elsewhere"
is not a thing we measured. A risk assessment that averages over application pairs still
reports a number that no particular pair of applications has.

**What this means for risk assessment.** Three of our results bear on how cross-application risk should
be measured rather than on how well it can be exploited.

The first is the static-cue audit. A risk assessment built on a behaviour-only encoding measures
behavioural risk, which is the right quantity for some questions and an *underestimate* of what a wearer
faces. On this corpus a head-only model one epoch from initialisation reaches 0.351 cross-application
from static anthropometry and posture alone. Any deployment in which head position reaches an
application carries that term whether
or not the behavioural term is present, and it requires no model training worth the name. The
qualification that head height survives an application change only **across applications that share a
posture** narrows where this bites but does not remove it; the applications in these corpora that share a
posture are the majority.

The second concerns the sensor set. Non-trivial cross-application identification is available from
head tracking alone — in the exposed arm, better than a published head-plus-controllers figure on the
same people. So a missing hand or controller channel does not by itself put a device outside this risk.
We stop short of a claim about AR glasses specifically: **every corpus scored here is VR**, and head-only
is our scope rather than a device we tested on.

The third is the distribution. The field reports rank-1 means, and the means conceal a wide spread over
individuals that is visible in the reference study's own released arrays. A defence, a disclosure or a
consent notice calibrated on a population mean is calibrated on nobody in particular.

**Alignment as a corpus specification.** The most useful output of our negative alignment result is not
the null itself but the number thirty-two. Fitting a transformation between two applications' embedding
spaces requires people recorded in both, and the only fully crossed cross-application corpus we are
aware of offers thirty-two of them among its non-test users. Whether a corpus with more would support
the method is **not established**: the map fitted on those thirty-two did not transfer to seventeen
others, which is a failure across people rather than a shortage of them, and a larger fitting
population might or might not cure it. What is available now is the negative itself, and it is
available because the route was run honestly rather than assumed to fail.

**The within-application gap remains open and confounded.** Our within-application figures (0.500
zero-shot, 0.616 exposed) sit well below the reference study's 0.831, but the two are not comparable for
the structural reason of Section 5.4, and even if they were, the gap would confound sensor set,
architecture and encoding. No sensor-set claim is available from this work in either direction, and we do
not make one.

---

## 8 Limitations

**Seventeen test users.** User-level uncertainty dominates every interval in this paper, and there is no
fix inside the corpus: the split is the reference study's and changing it would break the comparison. We
mitigate it with intervals everywhere rather than point estimates, with the per-user figure, with per-cell
results read as twenty quasi-replications inside one corpus (eleven of twenty resolved, none lost), and
with a second gallery size of thirty from a second corpus for the static-cue measurements. It remains the
binding constraint, and a difference of +0.05 is simply not resolvable here.

**One fully crossed corpus.** Across-XR is the only fully crossed cross-application corpus we are aware
of. Questset gives each participant two of four titles in disjoint groups, so it is two two-application
corpora rather than a crossed design, and BOXRR-23 — by far the largest XR motion corpus — has zero users
recorded in two applications. That scarcity is a finding about the field's data rather than an excuse,
but it bounds external validity all the same: one corpus, forty-nine people, one headset model, one
sitting, one laboratory.

**No temporal persistence anywhere.** Every (participant, application) cell in Across-XR is a single
unbroken recording, and the five applications were played in one sitting, so cross-application pairs
carry up to about an hour of separation and **never a different day**. Questset is also one sitting per
participant with one session per game. Nothing in this paper therefore speaks to whether a
cross-application identification succeeds a week later, which is the deployment question, and we expect a
figure obtained across days to be lower than the figures reported here.

**The architecture is not novel.** A BiLSTM trained with an additive angular-margin softmax is a standard
recipe. This is not a modelling paper and should not be read as one; the contribution is the protocol,
the measurements and the negative results.

**The self-match property of the reference study's within-application figure** means it is never paired
with ours, so the within-application comparison remains unresolved, and confounded between sensor set and
architecture even in principle.

**The per-user rank-disagreement observation is exploratory.** It was not registered in advance, it rests
on seventeen users with a correlation interval of about ±0.5, its attribution is confounded across three
simultaneous differences, and its coverage corollary was tested and failed. It is reported as a
hypothesis with a figure, and it needs a registered replication on a second corpus.

**Several arms are single-seed** — the smaller-identity arms, the dose arms, three of the five
leave-one-application-out arms, the exposed `raw` arm and the margin screen — and seed variance is
arm-specific rather than a pipeline constant: trained-out arms show a seed range of 0.010 on the primary
metric while the epoch-1 `raw` arm shows 0.029.

**The decision rule is not matched to the reference study** even where the metric and the users are:
nearest-reference-window matching on 15 s windows at 30 fps against a mean template on 10 s windows at
20 Hz. We address this by scoring both directions with a single harness each and reporting both, and the
two directions agree on every verdict, but they are not the same decision rule.

**The headline arm choice is post hoc relative to the first registration.** The registered prediction P1
targeted the zero-shot arm; the exposed arm's contrast against the reference study was registered before
it ran, but the decision to lead with it was made afterwards, and we say so rather than presenting the
registration as covering more than it does.

**The posture scoping result is single-corpus**, and its registration was written the same day it ran; it
needs a replication registered before a further corpus is acquired.

**We compare against a single published system.** Every claim of improvement in this paper is made
against one reference study. That study is the appropriate one — it is, to our knowledge, the only
published cross-application identification result that releases its trained model, its precomputed
test embeddings and its per-user accuracy arrays, which is what makes a paired per-user test possible
at all rather than a comparison of two point estimates. But a single reference means our comparison
inherits whatever is idiosyncratic to it: its architecture, its body-relative velocity encoding, its
fifteen-second window and its enrolment rule. A second published baseline evaluated under the same
harness would separate "our approach generalises better" from "our approach differs usefully from
this particular system", and we do not have one. The per-user rank disagreement reported in §6.9 is
a direct symptom of this: two systems can agree closely on a population mean while ordering the same
people differently, so a single reference constrains the population figure far more tightly than it
constrains any claim about mechanism.

---

## 9 Conclusion

Cross-application identification in extended reality is usually reported as a single number that
collapses from near-solved within an application to roughly three times chance across applications. This
paper re-assesses that number three ways. Measured against the reference study's released model, on its
own corpus, its own test users and its own metric, a head-only system that has been exposed to the
corpus's *other* participants identifies unseen users across applications better than a system using the
head and both hand controllers — +0.119 [+0.050, +0.192] on a single window at N=17, better on fifteen of
seventeen users, and +0.355 [+0.202, +0.499] over a ten-minute sequence. Without any exposure to the
corpus the comparison is unresolved, and we report it as unresolved. Measured against itself, the same
head-only pipeline shows that a model one epoch from initialisation reaches 0.351 cross-application from
static anthropometry and posture alone, so a behaviour-only assessment understates the total risk — a
result that extends the reference study's framing rather than contesting it. And measured against the
distribution rather than the mean, the reference study's own released per-user values run from 0.068 to
0.371 on a single window, so the population figure the field quotes conceals both the most exposed
individual and the least.

The route the reference study proposes for closing the cross-application gap — fitting orthogonal
alignments on training users — does not carry on a static-free encoding. The map it fits is
person-specific: it helps, when it helps at all, only the people it was fitted on, and the corpus offers
at most thirty-two participants outside the evaluation split to fit it on. Whether a far larger fitting
population would recover a transformation that does transfer is **untested here**, and it is the most
actionable *question* this work leaves: the corpus that could answer it is one in which many people are
recorded in many applications.

---

## References

1. L. Schach, C. Rack, R. P. McMahan, M. E. Latoschik. *Motion-Based User Identification across XR and
   Metaverse Applications by Deep Classification and Similarity Learning.* Frontiers in Virtual Reality,
   2026. doi:10.3389/frvir.2026.1743491. Preprint arXiv:2509.08539. Dataset (Across-XR), CC BY-NC-SA
   4.0: https://go.uniwue.de/identification-across-xr-applications. Code, data and released models:
   gitlab.informatik.uni-wuerzburg.de/hci/software/research-prototypes/2025-frontiers-identification-across-xr-applications/
   *(The orthogonal-alignment analysis of section 6.2.5 appears only in the Frontiers version; the
   unaligned headline figures are identical in both versions.)*
2. V. Nair, W. Guo, J. Mattern, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song. *Unique Identification of
   50,000+ Virtual Reality Users from Head & Hand Motion Data.* USENIX Security Symposium, 2023.
   arXiv:2302.08927. **Citation required by clause 5 of the BOXRR-23 Data Use Agreement.**
3. V. Nair, W. Guo, R. Wang, J. F. O'Brien, L. Rosenberg, D. Song. *Berkeley Open Extended Reality
   Recordings 2023 (BOXRR-23): 4.7 Million Motion Capture Recordings from 105,852 Extended Reality Device
   Users.* IEEE Transactions on Visualization and Computer Graphics, 2024. doi:10.1109/TVCG.2024.3372087.
   arXiv:2310.00430.
4. C. Rack, T. Fernando, M. Yalcin, A. Hotho, M. E. Latoschik. *Who is Alyx? A new behavioral biometric
   dataset for user identification in XR.* Frontiers in Virtual Reality, 2023.
   doi:10.3389/frvir.2023.1272234. Dataset: Zenodo doi:10.5281/zenodo.8379914.
5. C. Rack, K. Kobs, T. Fernando, A. Hotho, M. E. Latoschik. *Versatile User Identification in Extended
   Reality using Pretrained Similarity-Learning.* arXiv:2302.07517, 2024.
6. C. Rack, A. Hotho, M. E. Latoschik. *Comparison of Data Encodings and Machine Learning Architectures
   for User Identification on Arbitrary Motion Sequences.* IEEE AIVR, 2022. *(The SR/BR/BRV/BRA
   encodings.)* See also C. Rack et al., *Motion Learning Toolbox*, IEEE VR Workshops, 2024.
7. S. Baldoni, S. Benhamadi, F. Chiariotti, M. Zorzi, F. Battisti. *Questset: A VR Dataset for Network and
   QoE Studies.* ACM MMSys, 2024. doi:10.1145/3625468.3652187. Data (CC BY 4.0):
   researchdata.cab.unipd.it/1239/ (doi:10.25430/researchdata.cab.unipd.it.00001179).
8. S. Baldoni, S. Benhamadi, F. Chiariotti, M. Zorzi, F. Battisti. *Movement- and Traffic-based User
   Identification in Commercial Virtual Reality Applications: Threats and Opportunities.*
   arXiv:2501.16326, 2025. *(On Questset; test users are seen during training, and it uses the head plus
   both controllers, so it measures a different quantity from this work.)*
9. S. Baldoni et al. Cross-application identification across two applications by classification, IEEE VR,
   2025. **[REFERENCE TO BE COMPLETED — full bibliographic entry to be taken from Schach et al.'s
   reference list.]**
10. M. R. Miller, F. Herrera, H. Jun, J. A. Landay, J. N. Bailenson. *Personal identifiability of user
    tracking data during observation of 360-degree VR video.* Scientific Reports, 2020. **[REFERENCE TO BE
    VERIFIED against Schach et al.'s reference list.]**
11. M. R. Miller et al. Cross-system identification of VR users with a Siamese network, 2021.
    **[REFERENCE TO BE COMPLETED — full bibliographic entry to be taken from Schach et al.'s reference
    list.]**
12. K. Rogers et al. Identification from head and hand motion in VR, 2015. **[REFERENCE TO BE COMPLETED —
    full bibliographic entry to be taken from Schach et al.'s reference list.]**
13. F. Wang, J. Cheng, W. Liu, H. Liu. *Additive Margin Softmax for Face Verification.* IEEE Signal
    Processing Letters, 25(7):926–930, 2018. **[REFERENCE ADDED BY HAND — not present in any project
    source; verify before submission.]**
14. L. Ma, Y. Ye, F. Hong, et al. *Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in
    the Wild.* ECCV, 2024. arXiv:2406.09905. *(The activity-diversity experiment of Section 6.8.)*
