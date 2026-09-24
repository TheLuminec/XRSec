# Experiment 2 — exposure breadth across applications, at scale — REGISTERED (2026-09-23, Coordinator; setup only, not run)

**Premise.** The only data-side lever ever measured to cross an activity boundary is *exposure*: P3 trained
on four Across-XR applications for 23 people and carried +0.049 rank-1 to the fifth, unseen application
(495 identities, no Nymeria). Breadth of people did nothing (identity count), breadth of corpora did
nothing (arm B, Nymeria transfer). This asks whether breadth of **tasks** does, at the scale the project
now trains at: every application we hold at once, with one held out.

**Arms, one run per held-out Across-XR application X ∈ {superhot_vr, half_life_alyx, beat_saber,
synth_riders, social_vr}, seed 1** (`configs/exposure_breadth_<X>_s1.yaml` from
`exposure_breadth_lists.py`):

| | training identities | tasks in training |
|---|---|---|
| **breadth-X** (new) | BOXRR + alyx (the treatment's post-draw lists minus 23 BOXRR; seed 1: 2,848 + 60) + Nymeria 141 + Across-XR 0–22 = **3,072** | Beat Saber, Alyx, 20 daily-life scripts, 4 Across-XR applications |
| P3-X (exists, Miami) | BOXRR 415 + alyx 57 + Across-XR 0–22 = 495 | Beat Saber, Alyx, 4 applications |
| in-domain treatment (exists) | 3,072 | Beat Saber, Alyx, 20 scripts — no Across-XR |
| zero-shot / Z-676 (exist) | 3,072 / 495 | Beat Saber, Alyx |

Identity matching by the post-draw rule: the treatment's lists (48 Nymeria held out, 1,071 pinned
validation, 141 BOXRR dropped) plus **23 more BOXRR training users dropped** (the last 23 in sorted
order after the 141) to make room for Across-XR 0–22; Across-XR 23–31 are the explicit validation users
for that corpus (Schach's own validation split, and it keeps the fractional draw off the corpus); Across-XR
32–48 are excluded and evaluated (`test_on_excluded=true`, beside the 48 Nymeria held-out, per-dataset
AUC recorded). Corpus for training: `CrossApplicationXR_LOAO_<X>` (X's sessions absent from every user,
`build_loao_corpus.py`). **The unit is the X-cells** — rank-1 at N=17 on users 32–48 with X as gallery or
probe — scored by the programme's own harness (`across_xr_alignment.py`, `--normalizer-dataset`), pooled
over the five X, paired on cells and users; Questset cross-game rank-1 (A1 protocol, N=17 and N=30) as
the second, fully unseen instrument.

**Dose, stated:** Across-XR at ≈ 16.7k of ≈ 780k training windows (≈ 2 %), against P3's 11.6 %. So a
null against P3 has two explanations (breadth does not help; the objective barely saw the applications)
and only the first licenses the conclusion — **a `balance_identities=cap` companion run on one X is the
registered follow-up if the null lands**, not an amendment after the fact.

**Registered outcomes** (rank-1, X-cells pooled over five X, user bootstrap):

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| breadth-X − P3-X | **+0.00 to +0.06** — scale and daily-life tasks add a little to exposure | **< −0.03** — more and broader data *hurts* the held-out application (dose or interference) | −0.03 to 0.00: not resolved; run the cap companion before reading |
| breadth-X − treatment on the same X-cells | **≥ +0.03** — exposure to four applications carries, as P3 showed, on top of Nymeria breadth | **≤ 0** — with Nymeria in training, Across-XR exposure adds nothing: breadth of daily-life tasks already covered it | 0 to +0.03: unresolved at one seed |
| non-X cells, breadth-X − C2-lo-like full exposure | within ±0.03 | outside | — (control: removing one application leaves seen ones where they were) |
| Questset cross-game rank-1, breadth-X vs zero-shot | **+0.00 to +0.05** | **< −0.03** | +0.05 to +0.10: breadth transfers to a fully unseen corpus — seed it |

Which outcome is strong: the second row's falsifier — it would say Nymeria's 20 scripts already supply
what application exposure supplies, which collapses "task breadth" into one lever with two sources; the
first row's falsifier would say scale interferes. Five runs, one seed each (the unit is the application,
as in P3); ~90 min each on Miami. Not launched: this registration and the generator are the setup.

## Amendment 1 — 2026-09-24, launched on Miami; a leak check that was wrong on its own key

Five runs chained, synth_riders first; composed config asserted before each launch; lists match on all
five (training identities 3,072 = BOXRR 2,848 + alyx 60 + Nymeria 141 + Across-XR 23; dropB+23
`42100430b1a7`, excl `a6db3a689da2`); LOAO corpora rebuilt with the builder's own assertions. **A leak
check compared excluded and validation users by basename and fired on every config: the three "matches"
were who_is_alyx users numbered 32, 33, 34 — two corpora with numeric user directories compared on the
wrong key.** At path level there is no overlap anywhere. The same basename-collision trap that bit the
checkpoint copy, now inside the check written to catch a leak; recorded so the next check compares
paths. Miami's runner also refuses if any *other* LOAO tree appears in a composed config — five corpora
differing only in which application is absent is exactly the shape where the wrong one completes with
a plausible number for the wrong cell.

## Amendment 2 — 2026-09-24, five rows landed; dose measured; the harness made to score this arm's shape

Instrument facts only; no X-cell number exists yet and none has been read.

**The five training rows** (Miami, `154136a`, shard rows on `origin/miami-server`, identity `af7cf72022`, all
rc=0, 3,072 training identities, 65 excluded, one seed each):

| held out X | run_id | pooled AUC | LOAO_X AUC | Nymeria AUC | best_epoch / run |
|---|---|---|---|---|---|
| synth_riders | 96e53d1f5b62 | 0.7068 | 0.7005 | 0.7105 | 120 / 120 |
| social_vr | a1d8379d8ac0 | 0.7026 | 0.6889 | 0.7119 | 116 / 120 |
| superhot_vr | 2e22f1de5aee | 0.7051 | 0.6952 | 0.7137 | 120 / 120 |
| half_life_alyx | 530917952021 | 0.7001 | 0.7034 | 0.7048 | 113 / 120 |
| beat_saber | a82a59857062 | 0.7030 | 0.6915 | 0.7118 | 120 / 120 |

These are verification AUCs on the arm's own evaluation users, not the registered unit.

**Dose, measured rather than assumed:** Across-XR supplies **16,798 of 655,247 training windows, 2.56 %**
(Miami, through `build_sample_index` at the arm's settings). The registration said ≈ 16.7k of ≈ 780k, ≈ 2 %.
The numerator was right and the denominator was not. Quote 2.6 %, against P3's 11.6 %.

**Budget:** three of five selected epoch 120 of 120 and none stopped on patience. The P3 comparators are
censored the same way (their rows: best_epoch 113-120 of 120, none stopped early), so the categorical
convergence check reads *matched*: neither arm stopped early. Any breadth−P3 delta still carries a
shared censoring term, and the registered cap companion stays the follow-up for the −0.03..0.00 region.

**The alignment harness could not score this arm, and three edits make it do so** (`across_xr_alignment.py`;
docs/acceptance only, so `code_identity` does not move):

1. The gate built its evaluation set from `test_dirs` alone. This arm has none: it evaluates through
   `test_on_excluded` over `data_dirs`. The empty list loaded 0 users and died with the bare
   ZeroDivisionError. It now builds from `data_dirs` with the swap flipped, as the pipeline does.
2. That path seeds the pair draw with part 2, not part 4. With part 4 the gate failed at 1.9e-3; with
   part 2 it reproduced **0.706772 at 0.0e+00, 65 users** (Miami, both found there).
3. The population assertion `eval_users == 17` was P3's. It now expects the excluded users that lie
   under the evaluation directories: 17 for P3, 65 here. The 17-user guard on the *scoring*
   population (Schach's users 32-48) is untouched.

**And a fourth, found by the regression, which matters more than the three.** Omitting
`--normalizer-dataset` on a LOAO-trained checkpoint does not fail. The gate still passes, because it
loads the checkpoint's own corpus name, but scoring falls back to a target fit. On the P3 superhot
checkpoint that moved **A1 0.283 → 0.250 and A2′−A1 from +0.003 to +0.058**, a false positive that
would have read as alignment working. The harness now **refuses** a checkpoint holding Across-XR
statistics under a name other than the one being scored, and refuses a `--normalizer-dataset` the
checkpoint does not hold. Verified on AVALON, CPU, both directions:

| case | result |
|---|---|
| P3 superhot, no flag | refuses, rc=1 |
| P3 superhot, wrong LOAO name | refuses, rc=1 |
| P3 superhot, right flag | gate 2.8e-5; output **identical** to the pre-edit harness on this machine |
| zero-shot seed 1, no flag (stats under the scored name) | gate 2.0e-5; A0/A1/A2/A2′/A2null **digit-identical** to the committed file |

Against the committed P3 file (written on Miami), every arm used here reproduces to the digit. The
unrestricted 128-d fit (A2full) and the dimension curve at m=24/32 differ by up to 0.049 and 0.003. The
pre-edit harness gives the same values on AVALON as the edited one, so this is a machine difference in a
full SVD over at most 32 correspondences, where most of the basis is arbitrary. It predates this
amendment, and no registered quantity here uses A2full.

## Amendment 3 — 2026-09-24, the reading: exposure carries at scale, and holding an application out costs nothing measurable

Read by `exposure_breadth_read.py`, committed at `35e2468` before any breadth number existed. Inputs: six
Across-XR files (`2a5dc12`) and eight Questset files (`4e843c6`), every gate 0.0e+00 on the training GPU.
Result: `exposure_breadth_read.json` (`7c713ca`). Rank-1, user bootstrap, scored by where the interval falls.
One seed per X.

**Regions named before reading** (the registration left them open): row 1 above +0.06 means "exceeds,
report as such"; Questset −0.03..0 means "not resolved", and above +0.10 means "transfers, seed it".

| row | measured | reading |
|---|---|---|
| 1. breadth-X − P3-X, X-cells, N=17 | **+0.097 [+0.069, +0.126]** | **above the +0.00..+0.06 band, whole interval**: exceeds as registered |
| 2. breadth-X − Nymeria treatment, X-cells | **+0.112 [+0.083, +0.144]** | **in band** (≥ +0.03); the falsifier (≤ 0) is far away |
| 3. control: breadth − C2-lo (3-seed mean), non-X cells | +0.020 [+0.003, +0.041] | mean within ±0.03, **interval straddles the +0.03 edge**: the control holds at the mean and is not fully contained |
| 4. Questset: breadth (5 X) − zero-shot (3 seeds), 60 users, N=30 | **+0.062 [+0.042, +0.083]** | **straddles +0.05**: the mean sits in the "transfers, seed it" region, and the lower edge sits in the band. The falsifier (< −0.03) is excluded |

X-cell levels per held-out X, breadth / P3 / treatment: superhot 0.336 / 0.223 / 0.237, alyx 0.337 / 0.231 /
0.230, beat_saber 0.443 / 0.351 / 0.302, synth_riders 0.435 / 0.324 / 0.293, social_vr 0.286 / 0.224 / 0.215.
Every X moves the same way. The smallest gains are social_vr's (+0.062 / +0.071).

**Row 1 varies two things, as registered.** Breadth against P3 is 495 → 3,072 identities *and* Nymeria's 141
together, so "exceeds" cannot be attributed to either alone. The unregistered diagnostics below are what
bound it.

**Diagnostics, unregistered and labelled so.** They were computed after the rows, from the same committed
files, with the same bootstrap:

| contrast, X-cells | measured | what it says |
|---|---|---|
| breadth (X never seen) − C2-lo (X seen, 3 seeds, 3,095 identities) | **−0.007 [−0.038, +0.028]** | at scale, holding the fifth application out costs nothing measurable |
| breadth − zero-shot seed 1 (no Across-XR, no Nymeria) | +0.137 [+0.102, +0.173] | exposure to four applications carries to the fifth, at scale |
| Nymeria treatment − zero-shot seed 1 | +0.025 [−0.007, +0.057] | Nymeria alone moves Across-XR little, consistent with arm B's null on the seated corpora |

P3 at 495 identities paid **−0.036 [−0.054, −0.018]** against C2-hi for the same hold-out, pooled over the
five X (`across_xr_alignment_p3.json`). So the
sentence this supports is: **at 3,072 identities, exposure to four applications substitutes for the fifth
within the precision of one seed**, and at 495 it did not. The P3 note "identity count is not flat with
exposure" is the same mechanism, seen from the hold-out side.

**A no-cost hold-out reads exactly like a leak, so a leak was excluded before this was written.** Each
checkpoint's stored normaliser statistics were recomputed under six hypotheses: each application removed,
or none. All five match only their own hold-out at 0.0, and every other hypothesis, "none removed"
included, sits at 4.4e-3 or more. The P3 checkpoints were the positive control and read identically
(`loao_leak_check.py`, `loao_leak_check_{p3,exposure_breadth}.json`). Miami's filename, session-set and
inode checks on its own corpora agree. This rests on the checkpoints, not on either node's account.

**Qualifications that travel with the rows.**
- One seed per X, and five X.
- 3 of 5 runs selected epoch 120 of 120, as did the P3 comparators.
- Row 2's treatment has no Across-XR statistics and is normalised by a target fit, while breadth uses its own training statistics. This is the same convention every zero-shot comparator in the programme uses.
- Row 4 mixes Nymeria and Across-XR exposure. The treatment's Questset score separates them; see the decomposition below.
- Questset is one sitting per user.

**The Questset A1 arm (`questset_arms_REGISTERED_miami.md`) is settled by the same files.** Zero-shot,
three seeds, GPU, both directions:

| group | N=17 (chance 0.059) | N=30 (chance 0.033) |
|---|---|---|
| 1, Beat Saber / Cooking | 0.179 (0.199 / 0.168 / 0.170) | 0.126 |
| 2, Medal of Honor / Forklift | 0.217 (0.215 / 0.223 / 0.213) | 0.147 |

At N=17 both groups are **inside the registered 0.15–0.40 band**, so the Across-XR zero-shot result
survives a change of corpus. The falsifier (< 0.10) does not fire at either N. At N=30 both sit in the
0.10–0.15 region the registration calls "weakened". That registration gave one band for both gallery sizes
and never scaled it for the lower chance level at N=30, so the N=30 reading is reported as it falls and
not reinterpreted. **Group 2, where every static cue is at chance** (height lookup 0.033 at N=30), reads
3.7× chance at N=17 zero-shot. That figure is behavioural by measurement.

CPU against GPU on zero-shot seeds 1 and 3: every Questset cell within 5e-4.

**Row 4 decomposed: Across-XR exposure carries to a fully unseen corpus, and Nymeria does not.** This is an
unregistered diagnostic. Nymeria treatment seed 1 was scored on Questset on Miami's GPU (gate 0.0e+00,
`exposure_breadth_questset_treatment_s1_gpu.json`, `6f171d2`). Per user at N=30, 60 users:

| contrast | measured |
|---|---|
| treatment − zero-shot (Nymeria's 141 identities for 141 BOXRR) | +0.006 [−0.014, +0.026] |
| breadth − treatment (Across-XR 0–22 for 23 BOXRR, four applications, 2.6 % of windows) | **+0.056 [+0.039, +0.073]** |

The second contrast swaps exactly 23 identities, so it isolates exposure to four other VR applications, and
it moves Questset titles that appear in no training corpus. Qualifications: one treatment seed, and four of
the five breadth checkpoints include Beat Saber in their exposure, which Questset group 1 also contains. The
A3 covered/uncovered split is the check for that and has not been run.
