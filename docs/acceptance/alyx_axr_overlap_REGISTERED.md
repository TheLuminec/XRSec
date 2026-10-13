# Who-is-Alyx and Across-XR may share participants (registered 2026-10-12, before any run)

## Why

Dr. Feng (email, 2026-10-12) asked that held-out users never appear in training, pretraining included, and
asked specifically whether Who-is-Alyx and Across-XR, both from Würzburg, share participants. Every
Across-XR result in this project pretrains on Who-is-Alyx (BOXRR + alyx). If any of Schach et al.'s test
users 32-48 also took part in Who-is-Alyx, then "never seen in training" is false for them.

## What the documentation says (checked 2026-10-12)

| fact | Who-is-Alyx | Across-XR |
| --- | --- | --- |
| institution | HCI group, University of Würzburg | HCI group, University of Würzburg (same authors: Rack, Latoschik) |
| recruitment | "the participant recruitment system of our faculty" | "our university's participant recruitment system" |
| collection period | 2021-12-22 to 2022-07 (session notes in its Readme) | not stated; arXiv 2509.08539, September 2025 |
| participants | 76 (40 male, 36 female, age 19-33) | 49 (27 male, 22 female, age 19-54, height 161-191 cm) |
| per-person metadata | `summary.csv`: age, height, weight, gender, VR experience | **none published**; aggregates only |
| overlap stated? | no | no |

**What this settles and what it does not.**
- Overlap is **plausible**: the same group recruited from the same university pool about two years apart.
- It is **not documented either way**, and the published files cannot settle it, because Across-XR
  ships no per-person metadata.
- One bound exists. Across-XR reports that **38 of 49 participants had never played any VR game**, and
  7 + 1 + 3 had played one, two or all of the study's games. A Who-is-Alyx participant played Half-Life:
  Alyx in that study, so if they reported it, **at most 11 of 49** Across-XR participants could be shared.
  That bound rests on self-report and is not a measurement.
- Matching people across the two datasets with our own model is **not** used to settle it. That would
  be re-identifying study participants, which this work argues against.
- **The settling route is the authors.** A draft question is in COORDINATION (2026-10-12) for the user to
  send.

## The test that does not need the answer: retrain without Who-is-Alyx

If the Across-XR results do not depend on Who-is-Alyx in pretraining, then participant overlap cannot be
inflating them, whatever the overlap is. Two arms carry the paper's Across-XR claims:
- **zero-shot**: BOXRR + alyx, never trained on Across-XR; A1 0.234;
- **C2-lo**: zero-shot plus Across-XR users 0-22; A1 0.375; the +0.119 paired comparison against Schach.

Each is retrained three times (seeds 1-3) with **`who_is_alyx` removed from `data_dirs` and nothing else
changed**. Miami takes each original run's own `.hydra/config.yaml` and edits only `data_dirs` and
`experiment_name`:

| arm | original run dirs (seeds 1 / 2 / 3) |
| --- | --- |
| zero-shot | 2026-09-10/17-24-33, 18-41-34, 20-05-56 |
| C2-lo | 2026-09-10/22-01-22, 23-44-03, 2026-09-11/02-21-10 |

**Identity count.** The validation draw is 25 % of the pool, so dropping 76 alyx users trains on about
57 fewer identities: about 3,015 against 3,072 for zero-shot, and about 3,038 against 3,095 for C2-lo.
At this scale identity count is saturated (CLAUDE.md: 1,535 to 3,072 moved transfer by 0.001), so this is
recorded rather than corrected.

**Scoring.** `across_xr_alignment.py` on cuda, gated. **The six original checkpoints are re-scored with
the same harness version on the same device**, so both sides of every pair come from one harness. The
originals' committed files predate the per-cell, per-user output. Reading uses `alyx_axr_overlap_read.py`,
fixture-tested and committed with this file.

## Registered quantities

One quantity per arm: **A1 without alyx minus A1 with alyx**, per user (users 32-48, N=17), each side
averaged over its three seeds, user bootstrap.

| region | meaning |
| --- | --- |
| below -0.03 | Who-is-Alyx in pretraining contributes. Participant overlap is **not excluded** as the reason, and neither is content: Half-Life: Alyx is one of the five Across-XR applications. Read the per-cell split before saying which |
| **-0.03 to +0.03** | **BAND (predicted)**: the result does not depend on Who-is-Alyx, so any overlap cannot be inflating it by more than 0.03. The concern is closed for the claim whatever the authors answer |
| above +0.03 | removing alyx helps. Unexpected; report as such |

**Resolution.** These arms differ in training composition, and for that kind of contrast paired intervals
here have run at about ±0.035 (CLAUDE.md). So an interval spanning the band and a neighbour is likely,
and it reads as **not resolved**. The whole interval inside the band is the strong, decision-relevant
outcome.

**Descriptive, not registered:** the same difference split into cells that involve `half_life_alyx` and
cells that do not.
- A drop concentrated in the Half-Life cells points to content (the activity).
- A uniform drop does not separate people from content.
- The registered verdict is not moved by this split.

**What else produces each outcome.**
- A drop is produced by overlap, by Half-Life content exposure, and by the 57 fewer identities. That is
  three explanations, so a drop is the weak outcome.
- The band is produced only if alyx adds nothing to these figures, so the band is the strong outcome.

## Run acceptance (Miami)

- Every launch under `avm run "<name>" -- ... gated_launch.sh run ...`.
- No pull while live.
- Each config must differ from its original only in `data_dirs` and `experiment_name`. Check this with a
  diff, as for the GNN arm.
- Push each result file before anything reads it.
- Output names:
  - `across_xr_alignment_{zeroshot,c2lo}_noalyx_seed{n}.json` for the new arms;
  - `across_xr_alignment_{zeroshot,c2lo}_rescored_seed{n}.json` for the six original re-scores.

## Amendment 1, 2026-10-12: code identity, an instrument fact raised by Miami before any number exists

The originals were trained at `517cdaa57b`; the no-alyx runs train at `af7cf72022`. The two steps between
them are both recorded in CLAUDE.md:
- `03ea8e2376`: the `dyn` index is built in blocks, with output and every metadata tensor hash-identical.
- `af7cf72022`: logging only, accepted by a control seed reproducing digit-identical.

So the arms differ in identity but not in training numerics, by recorded acceptance rather than by
argument. Both sides are scored by the same current harness, as registered. The split guard
(`alyx_axr_overlap_split_guard.json`) reproduces the originals' 3,072 / 3,095 training identities and gives
3,015 / 3,038 for the no-alyx arms, with no held-out or validation Across-XR user in any training list.

## Amendment 2, 2026-10-13: the reader's file glob, an instrument fact found before any number was read

The first run of `alyx_axr_overlap_read.py` on the propagated files (dcac5ac) crashed before printing
anything. Its glob `across_xr_alignment_{arm}_{noalyx,rescored}_seed*.json` also matches the gate
certificates `*_seed{n}_gate.json` that Miami writes beside each result. Those certificates are lists, and the
loader expects a result dict. The fixture had no gate files, so it never reproduced that condition.
- **Fix:** the reader excludes `*_gate.json` and nothing else. The arithmetic, regions and bootstrap are
  unchanged.
- **Fixture:** it now writes gate files beside every result and asserts 3 + 3 seeds are read per arm. It
  fails against the old reader (rc 1) and passes against the fixed one.
- **Acceptance evidenced before reading:**
  - all 12 checkpoint gates are cuda, gap 0.0, 17 users each;
  - all six outcomes show `row_checks_pass`, with 3,015 / 3,038 identities at `af7cf72022`.
- **Pending from Miami:** the config diff, which must show only `data_dirs` and `experiment_name` differ.
  The Result is recorded as conditional on it.

## Result, 2026-10-13: the BAND prediction FAILED. Who-is-Alyx in pretraining contributes, in both arms

Read with `alyx_axr_overlap_read.py` after Amendment 2; output `alyx_axr_overlap_read.json` (2950cdd),
committed before this text. **Conditional on the config diff from Miami** (Amendment 2), which is pending.

| arm | A1 with alyx (re-scored) | A1 without alyx | no-alyx minus with-alyx, per user (N=17) | region |
| --- | --- | --- | --- | --- |
| zero-shot | 0.234 | **0.168** | **-0.066 [-0.101, -0.034]** | below -0.03: alyx contributes |
| C2-lo | 0.375 | **0.323** | **-0.052 [-0.064, -0.040]** | below -0.03: alyx contributes |

**Both intervals lie wholly below the band. The registered prediction (BAND) failed in both arms.**
- **Seeds.** Every no-alyx seed is below every with-alyx seed in both arms:
  - zero-shot: 0.182 / 0.161 / 0.160 against 0.231 / 0.230 / 0.240;
  - C2-lo: 0.329 / 0.339 / 0.301 against 0.368 / 0.379 / 0.378.

  The interval is per-user with seeds averaged. The seed spread does not reach across the gap.
- **Instrument.** The re-scored originals reproduce their committed figures within 4.1e-4 (cuda against the
  committed CPU scoring, inside the recorded 7e-4). All 12 gates were 0.0.

**What this means, as registered.** Who-is-Alyx in pretraining contributes to both Across-XR figures.
**Participant overlap is not excluded as the reason, and neither is content.** The registration named
three explanations for a drop and this result does not choose between them:
1. shared participants;
2. Half-Life: Alyx content;
3. the 57 fewer training identities.

On (3), the project's own record is that identity count is saturated at this scale (CLAUDE.md:
1,535 → 3,072 moved transfer by 0.001). That is recorded here, not argued.

**Descriptive per-cell split (not registered; it does not move the verdict).**

| arm | cells involving Half-Life | cells not involving Half-Life |
| --- | --- | --- |
| zero-shot | -0.080 [-0.127, -0.035] | -0.057 [-0.089, -0.028] |
| C2-lo | -0.066 [-0.079, -0.053] | -0.043 [-0.059, -0.027] |

- The drop is larger where Half-Life is involved, but it is clearly present where it is not.
- So Half-Life content alone does not account for it.
- A broad drop does not separate people from content, as registered.

**An explanation the registration did not list.** Who-is-Alyx is the only multi-session corpus in this
pretraining pool. Its cross-session positives (`cross_session_positives`) may teach session invariance that
helps every cross-application pair. This is untested and is recorded as an omission in the registration's
"what else" list, not as an account of the result.

**Consequences for what may be quoted.**
- **The overlap-free Across-XR figures are the no-alyx ones:** zero-shot **0.168** and C2-lo **0.323**
  (N=17, rank-1, single window). No Würzburg corpus is in their training.
- Zero-shot without alyx is **below** Schach et al.'s published 0.180 (head + controllers). The
  with-alyx 0.234 may be quoted only with this result beside it, until the authors answer the overlap question.
- The paired comparison against Schach (+0.119 for C2-lo) has **not** been recomputed for the no-alyx arm.
  Until it is, it carries the same condition.
- **The author question (COORDINATION) is now decision-relevant.**
  - If no Across-XR test user (32-48) took part in Who-is-Alyx, the with-alyx figures are users-unseen.
    The drop is then a content or data effect, and they stand.
  - Otherwise the no-alyx figures replace them.
