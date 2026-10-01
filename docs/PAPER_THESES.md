# Candidate theses for the paper

Written 2026-10-01 by session 726be628, after consulting the Coordinator session and a full read of
`docs/PAPER_PLAN.md`, `PAPER_DRAFT.md`, `PROGRESS_REPORT.md`, `GENERALISATION_PROPOSAL.md`,
`LITERATURE_BRIEFING.md` and all thirteen `docs/acceptance/*_REGISTERED.md` / `*_RESULTS.md`
files. Nothing here is a new measurement; every number is quoted from a registered, gated result
and carries the scope caveat recorded beside it. The existing `PAPER_DRAFT.md` is one of the
candidates, not the premise.

## The situation in three sentences

The draft (2026-09-17) leads with a paired comparison against Schach et al.'s released model on
Across-XR - the project's tightest external anchor, but one sitting, seventeen people. Since then
the project has measured what the draft lists as an unsizeable limitation: cross-day persistence
(alyx -0.261, ball-throwing -0.131), a cross-headset cost (-0.235), a learned signature on real AR
glasses that survives activity matching (+0.182) and leave-script-out (-0.043), and a reversal of
the static-cue audit on a cross-day corpus (`raw` +0.09 within a session, -0.149 across days).
The two bodies of evidence answer each other's strongest objection, and that is the thesis
opportunity.

## The evidence, grouped by the boundary it crosses

Every result below is three seeds and registered unless marked. "Boundary" is what differs between
gallery/training and probe/test. This table is the organising object for thesis C below.

| boundary | corpus | metric, N | learned (`dyn`) component | static (`raw`/lookup) component | source |
| --- | --- | --- | --- | --- | --- |
| unseen users, same activity, same sitting | BOXRR (94 clean users) | rank-1 @N=17 | rank-1 0.862 @N=17 k=16; 0.948 at 2096 ids (2 seeds) | height alone 0.38 | CLAUDE.md step 6, 9.14 |
| unseen users, seated video corpora | 7 held-out corpora | verification AUC, pooled | 0.60-0.62 AUC, saturates in identity count | placement lookup 0.73 beats the model | GENERALISATION_PROPOSAL 9.1-9.14 |
| unseen users, **AR glasses**, activity matched | Nymeria (48 users) | verification AUC, 48 users | constrained AUC **0.669**, delta vs control CI [+0.164, +0.231] | lookup = shared SLAM map (0.73), not a person | `nymeria_in_domain_REGISTERED.md` |
| unseen users **and unseen tasks**, AR glasses | Nymeria LSO (25 users, 5 scripts) | verification AUC, 25 users | 0.6145; LSO - treatment **-0.043 [-0.056, -0.030]** | - | `nymeria_lso_REGISTERED.md` |
| unseen users, **unseen application** (same sitting) | Across-XR, users 32-48 | rank-1 @N=17 | zero-shot 0.234; exposed (C2-lo) 0.375; **+0.119 [+0.050, +0.192]** vs Schach's 0.180 on their metric | `raw` adds **+0.117** at epoch 1 | `across_xr_alignment_RESULTS.md` |
| application held out of training, at scale | Across-XR breadth | rank-1 @N=17 | hold-out cost **-0.016 [-0.043, +0.014]**; breadth - treatment +0.108 | - | `exposure_breadth_REGISTERED.md` Am. 5 |
| unseen corpus, second cross-application set | Questset | rank-1 @N=30, per user | +0.056 [+0.037, +0.076] over zero-shot from breadth | height lookup at chance where posture changes | `exposure_breadth_REGISTERED.md` (Am. 4-5), `questset_geometry.py` / `questset_static_lookup.py` |
| **different day**, same headset | alyx (N 12-17) | rank-1, N 12-17 per seed | 0.483 vs 0.743 same day: **-0.261 [-0.345, -0.182]** | lateral lookup collapses (0.539), height holds (0.661) - lookup verification AUC, not rank-1 | `alyx_cross_day_REGISTERED.md` |
| **different day**, same headset | ball-throwing (N=41, 2 s model) | rank-1 @N=41 | 0.693 vs 0.824: **-0.131 [-0.172, -0.089]** | `raw` -0.149 across days | `ballthrowing_cross_day_REGISTERED.md`, `broad_2s_REGISTERED.md` |
| **different headset**, different day | ball-throwing | rank-1 @N=41 | 0.458: **-0.235 [-0.282, -0.191]**; tilt removal does not reduce it | `raw` collapses to 0.133 | same |

Three structural nulls sit beside the table, all registered: identity count does not cross an
activity boundary (0.672/0.672/0.671 raw; dyn saturates after 1000); activity diversity does not
either (-0.0012 [-0.0045, +0.0020], 5 seeds); Nymeria training buys nothing on ball-throwing
(+0.002 [-0.017, +0.020], three seeds) or the seated corpora (+0.003, one seed at 240 epochs, `e240_transfer_REGISTERED.md`). Exposure to other people in the target
application family is the only data-side lever measured to cross an application boundary.

## Candidate theses

### A. The draft: cross-application risk re-assessed against the released SOTA, head-only

**Sentence.** Head motion alone, with exposure to other users of the target corpus, identifies
unseen users across XR applications at least as well as the published head-plus-controllers model
on its own people and metric; a behaviour-only assessment understates the risk; the SOTA's own
proposed alignment fix does not transfer; the population mean conceals the individual.

**Carried by.** Schach paired +0.119 [+0.050, +0.192] (their metric) / +0.176 (ours), 15/17 users;
zero-shot unresolved and said so; P2 static audit +0.117 [+0.042, +0.192] at epoch 1; P3/breadth
unseen-application carry; alignment A2-A1 +0.011 [-0.020, +0.041] with band excluded; Schach's own
per-user range 0.068-0.371.

**Strongest objection.** Everything is one sitting and seventeen people, and the project's own
newer data say a sixth to a third of same-session identification does not survive a day and that the
+0.117 static term is *session-bound* on a cross-day corpus. A reviewer who knows the field asks
what 0.375 is worth next week, and the draft currently answers "unknown". Also: the beat requires
training on their 23 users (the draft handles this honestly, but it is the attack that matters).

**Needs.** No GPU. Fold the cross-day / cross-headset costs into section 6.6 and 8 as a sized
qualifier rather than a bare limitation.

**Verdict.** Best-supported single comparison in the repo and should remain the headline
*measurement* whatever the thesis. As a *thesis* it is now narrower than the evidence.

### B. The Coordinator's: behaviour persists, pose belongs to the session

**Sentence.** Head motion carries a learned behavioural identity that generalises to unseen
people, unseen tasks and unseen days, on AR glasses as on VR headsets, while the static pose cues
that dominate same-sitting figures belong to the recording session, not the person, and do not
survive a change of day.

**Carried by.** Nymeria constrained 0.669 and LSO -0.043; alyx and ball-throwing cross-day figures;
the `raw` reversal (+0.09 within session, -0.149 across days, 0.133 across headsets); the lookup
decomposition (placement within a sitting, height across days, Nymeria a map); Questset posture
scoping (height at chance when posture changes).

**Strongest objection.** Several small corpora with no external comparator for most rows: Nymeria
is verification AUC on 48 users from one draw, one sitting on the target device; alyx is 12-17
users per seed with day and session confounded; ball-throwing uses a weak 2 s model (0.55 AUC in
domain) at an assumed 45 Hz with fixed headset order. "Across unseen days" rests on two corpora,
neither on AR glasses. And "it persists" and "a third of it does not persist" are the same number
read two ways - the framing has to commit to one.

**Needs.** rank-1/CMC on the Nymeria 48 (CPU, cheap - puts the AR-glasses result on the field's
axis); a per-user distribution of the constrained figure; ideally a 10 s-capable cross-day corpus
so the day cost is not only measured at 2 s and on alyx. No cross-day AR-glasses corpus exists in
the catalogue, so that caveat is permanent for now.

**Verdict.** The most novel and the most privacy-relevant story in the repo; alone it is
under-anchored.

### C. Unified: what a head-only biometric identifies, and how far each part travels (RECOMMENDED)

**Sentence.** A head-motion biometric is the sum of three components - placement in the tracking
space, head height and posture, and a learned movement signature - and each travels a different
distance: placement dies at the end of the sitting, height survives a day but not a change of
posture, and the learned signature crosses people, applications and tasks, pays a sixth to a third
of its value across a day depending on the activity, and more across a headset. Published
same-sitting figures cannot say how much of what they report persists, and where position reaches
the model they also mix placement in without saying so.

**Carried by.** The boundary table above, in full: A supplies the external anchor (one row of the
table, measured against a released model, paired), B supplies the rows the field has never
measured (days, headsets, tasks on AR glasses), and the static decomposition supplies the
column split. The three nulls say which levers do not move which rows.

**Why it is the strongest.** Each of A and B has exactly one strongest objection, and the other
answers it: A's "what is it worth next week" is answered by B's cross-day rows; B's "no anchor,
small corpora" is answered by A's paired comparison and 23 gates. The project's most distinctive
methodological claim - a mandatory training-free baseline beside every model figure - is what makes
the column split possible, so it stops being a side note and becomes the method. And the privacy
reading sharpens: a behaviour-only assessment (Schach's) understates *same-sitting* risk by the
static term, and a same-sitting assessment (everyone's, ours included) overstates *persistent*
risk by a sixth to a third. Both halves are measured.

**Strongest objection.** Breadth over depth: a reviewer can say each row is a different corpus,
window, metric and model, so the "table" is an assembly rather than one experiment. The defence is
that every row is registered, gated and multi-seed, and the paper says what each row is; but the
headline must still be *one* paired number (A's), with the table as the frame, or the paper reads
as a survey of the repo. Also: the ball-throwing row is a 2 s model and the Nymeria row is
verification, so the table mixes metrics; say so in the table itself.

**Needs.** Nothing on GPU to write it. Cheap CPU additions that make the table honest: Nymeria
rank-1 on the 48; the cross-day cost on alyx at 10 s is already there (ratio 0.650) and should sit
beside the 2 s one. Optional and expensive: a Rack 2023 re-run (36 h/seed) for a second baseline -
*not* required; the draft's limitation 8 already prices the single-reference issue.

### D. Second paper: the AR-glasses result on its own

**Sentence.** On real AR glasses, a head-only model identifies people it never saw, doing daily
tasks no training identity performed, from how they move rather than what they did.

**Carried by.** Nymeria in-domain and LSO, with the three controls and the script-pair protocol.
**Objection.** One sitting per participant and no cross-day AR-glasses corpus exists; in-domain
only (the same model buys nothing on unseen corpora). **Needs.** rank-1 and per-user; a second
held-out user draw; a cross-day AR corpus if one ever appears. Keep as paper two, or as the
AR-glasses row of C.

### E. Methods/critique paper: the static-cue audit of the field

**Sentence.** Most head-pose identification in the published corpora is placement and height;
four public corpora store a direction vector where a position should be; a three-number
training-free lookup matches a trained model and should be reported beside every figure.

**Carried by.** Lookup 0.726 vs model 0.723; per-axis decomposition; the tier-2 direction-vector
finding; Nymeria as a map; Questset posture; Across-XR epoch-1 `raw`; Schach's self-match
structure (diagonal cells only). **Objection.** Audits our corpora and our models, not other
groups' models (Schach's is static-free by construction); the earlier 5-fold programme is less
tightly seeded than the later arms; "position identifies" may read as known. **Needs.** No GPU;
careful naming of tier-1 vs tier-2 corpora. Has all its numbers today; venue-dependent.

## Recommendation

Write **C** as the paper, with **A's paired comparison as the headline measurement** and the
boundary table as the figure the paper is organised around; **B's sentence is the discussion's
conclusion, not the title's claim**. Concretely, relative to the current draft:

1. Title/abstract: move from "re-assessment of cross-application risk" to "what a head-only
   biometric identifies and how far it travels", keeping the Schach comparison as the first
   number in the abstract.
2. Add one results section (cross-day, cross-headset, AR-glasses rows) and one figure (the
   boundary table or a heat-map of it). All inputs exist in `docs/acceptance/`.
3. Promote the training-free baseline from a methodological bullet to the method that makes the
   column split possible.
4. Rewrite section 8's "no temporal persistence anywhere" as a sized cost, and rewrite 6.6's
   privacy reading with the cross-day qualifier (the static term is session-bound on
   ball-throwing).
5. Keep D and E as follow-on papers; neither needs the GPU now.

## Sentences that must not appear in any version

From CLAUDE.md and the acceptance files, verbatim constraints: no "beat" for zero-shot (0.181 vs
0.180 is a placement); nothing paired against Schach's 0.831 or 0.785; no "exposure carries in both
directions" (Questset reverse +0.014 [-0.006, +0.035]); no "Nymeria training generalises to new
activities" (null on ball-throwing); no mechanism for the headset cost (tilt tested, not
supported; device confounded with elapsed time); no `raw` figure on Nymeria or the seated corpora
without the placement caveat; no "more correspondences would fix alignment" (unearned); Nymeria
row figures 0.717/0.726 are never quoted - the constrained 0.669 is; margin 0.1/15 reverses at
scale; every one-sitting figure carries the cross-day cost (a sixth to a third: ball-throwing
0.131/0.824, alyx 0.261/0.743) when read as persistence; the `raw` cross-day deficit is not a mechanism (epoch 2 selected).

## Coordinator review (2026-10-01), applied at merge

Five corrections. Every number was checked against its source; the rest of the document stands.

1. **"About a third" overgeneralised one corpus, and the phrase was mine.** I gave it in my consultation
   answer. The measured cross-day costs are 0.261 of 0.743 on alyx (35%) and 0.131 of 0.824 on
   ball-throwing (16%). Every instance now reads "a sixth to a third, depending on the activity".
2. **"Published same-sitting figures, including the SOTA's, measure a mixture" was false of Schach et al.**
   Their BRV encoding discards head position by construction, which thesis E already says. What their
   figure cannot say is how much of it persists. The sentence now says that, and confines the placement
   mixture to models whose input contains position.
3. **The Questset row's source is now right.** +0.056 [+0.037, +0.076] is breadth minus zero-shot from
   `exposure_breadth_REGISTERED.md` (Amendments 4-5), not from the reverse-direction arm.
4. **"Nymeria training buys nothing on the seated corpora (+0.003)" is one seed at 240 epochs.** It is now
   marked as such; the ball-throwing null is the three-seed one.
5. **Order the "placement dies at the end of the sitting" evidence.** Lead with the training-free legs:
   alyx lateral 0.539 against height 0.661, and the co-location geometry. The `raw` model's cross-day
   deficit carries the epoch-2 confound and is corroboration, not the premise.

On metric mixing in the boundary table: give every row its own metric and N column (done in 2c25588; at merge, the Questset row was corrected to N=30, which is what the +0.056 reading uses, and the alyx static cell was marked as AUC). The cheapest fix is
Nymeria rank-1 on the 48 held-out users, which needs CPU only. The choice of thesis remains the user's.
