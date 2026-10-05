# Nymeria in-domain arm on the rank-1 axis — DRAFT FOR COORDINATOR REVIEW (2026-10-05)

**Status: DRAFT. Not registered until the Coordinator accepts it. Written before any rank-1 number for
these checkpoints exists.** The harness and fixture were built and run on synthetic data only
(`nymeria_rank1_fixture.py`: 57 checks pass). No checkpoint has been loaded and nothing has been scored.

## Purpose

The paper reports every identification row as rank-1 at N=17. The AR-glasses row (Nymeria in-domain,
`nymeria_in_domain_REGISTERED.md`) exists only as verification AUC: constrained 0.669 at 120 epochs (every
positive one person across two scripts, every negative two people in one script). This puts that arm on
the rank-1 axis, on the same 48 held-out people and the same six checkpoints, with the activity cue
denied the way the constrained verification protocol denies it.

## Population and checkpoints

- **People:** the 48 held-out Nymeria users (`nymeria_in_domain_heldout48.txt`), never trained on or
  validated on by either arm. One sitting per person, 2–8 sequences (median 5), 2–7 distinct scripts.
- **Windows:** `dyn`, 10 s, stride 5. 47,796 windows (asserted against `nymeria_script_pair.json`).
- **Checkpoints:** the six 120-epoch checkpoints. The stem is
  `nymeria-in-domain_multi3_bilstm_10s_20hz_emb128_train.pth` under `runs/2026-09-21/<dir>/checkpoints/`.

| seed | arm | run dir | run_id | recorded `selected_test_auc` (gate target) | constrained AUC, GPU (Amendment 17) |
|---|---|---|---|---|---|
| 1 | control | 07-55-55_train | 6579aad03df0 | 0.541536678870519 | 0.4731 |
| 1 | treatment | 09-16-23_train | 240b8138b9fa | 0.7082129749986861 | 0.6609 |
| 2 | control | 10-49-06_train | 36862c64b4e6 | 0.5269215934806399 | 0.4660 |
| 2 | treatment | 12-08-08_train | 0a76c71c60ea | 0.7263105014959971 | 0.6782 |
| 3 | control | 13-38-42_train | 319b7f32b347 | 0.5385744935936398 | 0.4764 |
| 3 | treatment | 14-57-30_train | e2ff80b76f95 | 0.7176762554380629 | 0.6649 |

The rows are in `results/runs/feng-ms-7b51.jsonl`. The e240 pair is out of scope (user-approved scope:
three seeds × two arms).

## Gate

Each checkpoint goes through `across_xr_alignment.gate`, the same gate `alyx_cross_day.py` used on these
treatment checkpoints on cuda, where the gap was 0.0. It must reproduce its own `selected_test_auc` on its
own 48 evaluation users **on cuda, within 1e-4**.

| gate outcome | means |
|---|---|
| gap ≤ 1e-4 on cuda | read |
| gap > 1e-4, or no row found, or a path does not resolve | **refused**: a JSON with `refused` is written and nothing is read from it. It is not re-scored on another device and the tolerance is not widened (Amendment 16) |

The harness also asserts four things before scoring:
- the checkpoint's excluded Nymeria users are exactly the 48;
- `test_on_excluded` is set;
- no held-out user is a validation user;
- the scoring index holds exactly the 48 users and 47,796 windows, and every window has a script label.

## Protocol (`docs/acceptance/nymeria_rank1.py`)

The A1 rule is imported, so there is one implementation:
- L2-normalised window embeddings;
- a template is the renormalised centroid of a user's windows;
- cosine similarity;
- ties are rank-averaged, so a tie is never rank 1. A constant scorer therefore gives rank-1 = 0 and mean
  rank (N+1)/2, not 1/N. Note that `step6_seated_dyn.rank1` uses a different convention, 1/(better+tied).

The full-N computation is asserted equal to `rank1_per_user` on every gallery at run time. **The probe is
one 10 s window.**

**Primary: `constrained`, a script-matched gallery with the probe's script excluded from every template.**
- For each script s, the candidates are the held-out users who recorded s.
- Each candidate's template is the centroid of their windows from **scripts other than s**.
- The probes are the candidates' s-windows.
- So the true person and every impostor are enrolled on activities other than the probe's, and every
  candidate did the probe's activity. This is the structure of Across-XR A1, where every candidate recorded
  the probe's application.
- **N=17:** for every probe, 200 galleries of 16 impostors are drawn from the other candidates for that
  script. The draw seed is 67, and the draws are identical for every checkpoint, so the arms pair.
- Only scripts with ≥ 17 candidates enter N=17. On the 48 there are five: S7-Cooking 29, S2-Where_is_X 23,
  S16-Simon_says 22, S12-Game_night 21, S10-Housekeeping 20. That gives **115 (user, script) cells over
  46 users**; eric_martin and thomas_nixon have none of these scripts.
- **N=all:** 18 scripts with ≥ 2 candidates, 200 cells, 48 users. Each cell is scored at its own N, and the
  chance level is reported beside it.

**Why not the per-pair design, or the suggested fallback.** Both were decided from the scripts table alone,
before any number existed.
- **Per-pair design** (gallery script g, probe script p, all templates from g): only 4 ordered pairs reach
  17 users and 20 reach 10, covering 72 of 700 user-pair slots at N ≥ 17. It is kept as the secondary
  `pair` protocol (N ≥ 10, per cell).
- **"All 48 candidates, every template minus s" fallback:** this does **not** deny the activity cue on this
  corpus. Nymeria scripts come in bundles: over the 188 non-held-out participants the co-occurrence lift
  reaches 6.3 and several pairs never co-occur. The true person recorded s, so their other scripts share
  s's bundle more often than a random impostor's do. Fixture 5 measures this leak under a bundle-aligned
  activity-only embedding: **+0.05 to +0.08 at N=17** above chance, against ≤ 0.01 for the constrained
  protocol. The fallback is run as `fallback_all48` (a diagnostic only, never quoted).

**The registered statistic is CELL-BALANCED.**
- A cell is one (user, probe script) pair. Its value is that user's hit rate over their probes of that
  script. The figure is the mean over cells, with a cluster bootstrap over users (10,000 draws, the
  imported `N_BOOT`).
- Why: under an activity-only embedding a probe's embedding does not depend on who recorded it. So within
  one gallery the candidates' cell rates sum to 1 at N=all, and to n/17 at N=17 (Σₖ C(n−k,16)/C(n−1,16) =
  n/17). The cell mean is therefore chance whatever the activity geometry or window counts.
- The per-user mean, which the brief asked for, does **not** have this property. It reweights cells by how
  many eligible scripts a user has. On the real 48-user table, one activity-only embedding moved it **0.057
  above 1/17** (fixture 5). That is as large as the control band. Under the same embeddings the cell-balanced
  mean stayed within 0.01 of chance in every realisation.
- The per-user mean (with the imported `ci`) and the probe-pooled mean are reported beside the cell-balanced
  mean and are not registered. **This departs from the brief, and the Coordinator should accept or reject
  it explicitly.**

**Secondary rows, reported and not registered:**
- `constrained` at N=all.
- `unconstrained`: all 48 candidates; the true person's template excludes the probe's own sequence, any
  script.
- `pair` cells.
- `fallback_all48`.
- For each checkpoint, the verification AUC on the **same score set** as its N=17 rank-1 (genuine = probe
  against own template, impostor = probe against every other candidate). From that AUC the harness
  computes the Gaussian-implied rank-1 and the offset between the probe-pooled measurement and the
  implication. The implication formula was gated against the published alyx triple:
  0.1028 / 0.1496 / 0.0749 against 0.103 / 0.149 / 0.075.

**Do not read `unconstrained − constrained` as an activity share.** On this script table an activity-only
embedding reads **at or below** chance under `unconstrained`: 0.017 and 0.013 against a chance of 0.021 at
N=all (fixture 5). Leaving one sequence out removes the probe's script from the true person's template,
while impostors who did that script keep it. On the verification axis the unconstrained-minus-constrained
gap was read as "about 0.05 was activity". On the rank-1 axis the same subtraction does not mean that.

## Predictions, computed before any rank-1 number exists

All implications use the equal-variance Gaussian, d′ = √2·Φ⁻¹(AUC) and rank-1 = P(genuine beats N−1
impostors). This is the formula gated in the fixture.

| anchor | AUC | implied rank-1, N=17 |
|---|---|---|
| treatment, constrained verification (GPU, 3-seed mean) | 0.668 | **0.155** (seeds 0.150 / 0.164 / 0.153) |
| treatment, unconstrained on the same embeddings (CPU) | 0.715 | 0.199 |
| treatment rows | 0.717 | 0.202 |
| control, constrained verification | 0.472 | 0.049 |
| control, unconstrained on the same embeddings | 0.533 | 0.072 |
| control rows | 0.536 | 0.073 |

These are **k=1 against k=1** implications. Here the gallery side averages every non-s window of a person:
about 996 windows per user at stride 5, roughly 1 hour of recording with one script removed.
- **Gallery averaging.** If the gallery half of the window noise averages out (d′·√2), the implied values
  become 0.216 and 0.292. If it averages out more than that (d′·2), they become 0.320 and 0.448.
- **What the verification AUC brackets.** Verification constrained negatives are same-script (adversarial).
  Here the impostor templates exclude the probe's script, so the relevant AUC lies between the constrained
  and unconstrained figures.
- **Non-Gaussian offset.** The project's observed offset for learned `dyn` is +0.05 to +0.11 (BOXRR,
  CLAUDE.md); the static alyx case sits at about 0.
- **Nearest measured neighbour, same treatment model.** alyx cross-day 0.483 (N 12–17, gallery one whole
  session) and same-session 0.743. That is a different corpus and not a prediction for this one.
- **Control.** The cell-balanced statistic cannot be moved by activity alone. The control's below-chance
  verification figure came from same-script negatives, and this protocol has no same-script impostor
  template. Prediction: chance, 0.059.

## Registered quantities

Each quantity is scored by **where its 95 % interval falls**, never by p < 0.05. Every line is partitioned
from −∞ to +∞ with no unnamed region; `REGIONS` in the harness asserts this at import. Unit: the
constrained protocol, N=17, cell-balanced, each cell averaged over the three seeds, user bootstrap.

**1. Treatment rank-1** (46 users, 115 cells)

| region | means |
|---|---|
| **band: 0.20–0.45** | identifies unseen people across activities on glasses, with gallery averaging paying off as it does elsewhere |
| **falsifier: < 0.12** | the cross-activity person cue does not survive identification: below even the k=1 implication of the adversarial AUC (0.155) by more than 0.03 |
| **landing between them means: 0.12–0.20** | at the k=1 implication: averaging an hour of other-activity enrolment buys nothing over a single window. Report as such, not as a pass |
| above 0.45 | exceeding. Credit only after the control lands in its band and fixture 5 has been re-run on the node that scored it |

**2. Treatment − control, paired per cell**

| region | means |
|---|---|
| **band: +0.12 to +0.38** | the verification gain carries to identification |
| **falsifier: < +0.05** | in-domain training buys no identification across activities |
| **landing between them means: +0.05 to +0.12** | a real but smaller gain than the verification delta implies. *Not resolved* unless the interval clears +0.05 |
| above +0.38 | exceeding. Check the control first: a control below its band produces this for the wrong reason |

**3. Control rank-1**

| region | means |
|---|---|
| **band: 0.03–0.08** | at chance (1/17 = 0.059) within sampling: no zero-shot person cue across activities |
| **falsifier: > 0.12** | the zero-shot model identifies across activities. The control's sub-chance verification figure (0.47) came from the pairing, not from an absent person cue. Say so before quoting the delta |
| **landing between them means: 0.08–0.12** | a small zero-shot person cue that the adversarial pairing hid. Report it beside the delta |
| below 0.03 | anti-identifying when activity cannot move the statistic. Check the harness before reading anything |

**Partition check:**

| quantity | cut points | contiguous |
|---|---|---|
| 1 | −∞ \| 0.12 \| 0.20 \| 0.45 \| +∞ | yes |
| 2 | −∞ \| 0.05 \| 0.12 \| 0.38 \| +∞ | yes |
| 3 | −∞ \| 0.03 \| 0.08 \| 0.12 \| +∞ | yes |
| gate | ≤ 1e-4 \| > 1e-4 | yes |

**Which outcome is strong.**
- Quantity 1: the falsifier is the informative outcome. It would say the verification figure overstates
  what the model does on the identification axis across activities. The band holding confirms an
  implication. It is weaker, and it carries the enrolment-amount caveat below.
- Quantity 3: the strong outcome is the falsifier. It would change how the control's 0.47 has been read
  since Amendment 12.
- Before running, consider what else could produce each outcome.
  - A treatment in or above its band is also produced by any **person-constant cue that `dyn` and the
    script exclusion do not remove**. Nymeria sessions are one sitting, so the device (8 serials across the
    48 people, shared widely) and the location (25 locations, 11 of them used by one person only) are
    constant within a person.
  - The protocol denies activity. It does not deny the sitting. That is the caveat this arm has carried
    from the start (one sitting per participant), and it travels with the rank-1 row.

**Resolution.** In the fixture's stand-in reading, a no-person-signal control's cell-balanced interval was
about ±0.011 on 115 cells and 46 users. A real control with heterogeneous users should be wider. The bands
are 0.05 to 0.26 wide.

## Caveats that travel with the number

- **One sitting per person:** no cross-day cost is paid. On alyx the same treatment model lost 0.26 rank-1
  across days.
- **Enrolment amount differs from other rows.** The template here is all of a person's other-script
  windows, roughly an hour. Across-XR A1 enrols one application. **k must be stated with the row.**
- **N=17 covers 46 of 48 people.**
- **Probes are single 10 s windows.**

## Commands (Miami; run by the Coordinator session)

1. No `git pull` while any pipeline process is alive on the node (`systemctl --user list-units
   'xrsec-*.scope'` must be empty). Then pull, and confirm that the harness, the fixture and this file are
   present.
2. Run the fixture on the node first (CPU, about a minute):
   `.venv313/bin/python docs/acceptance/nymeria_rank1_fixture.py` — it must end `ALL FIXTURES PASS`.
3. Score on cuda under the cap, seed 1 before seed 2, one launch per seed. Each JSON is written the moment
   its checkpoint is scored. **Push each JSON to `origin/miami-server` as it lands, before reading it.**

```bash
S=nymeria-in-domain_multi3_bilstm_10s_20hz_emb128_train.pth
XRSEC_MARKER_DIR=/home/feng/xrsec_markers bash gated_launch.sh run nymeria_rank1_s1 -- \
  .venv313/bin/python docs/acceptance/nymeria_rank1.py score --device cuda --out-dir docs/acceptance --checkpoints \
  1=control=runs/2026-09-21/07-55-55_train/checkpoints/$S \
  1=treatment=runs/2026-09-21/09-16-23_train/checkpoints/$S
XRSEC_MARKER_DIR=/home/feng/xrsec_markers bash gated_launch.sh run nymeria_rank1_s2 -- \
  .venv313/bin/python docs/acceptance/nymeria_rank1.py score --device cuda --out-dir docs/acceptance --checkpoints \
  2=control=runs/2026-09-21/10-49-06_train/checkpoints/$S \
  2=treatment=runs/2026-09-21/12-08-08_train/checkpoints/$S
XRSEC_MARKER_DIR=/home/feng/xrsec_markers bash gated_launch.sh run nymeria_rank1_s3 -- \
  .venv313/bin/python docs/acceptance/nymeria_rank1.py score --device cuda --out-dir docs/acceptance --checkpoints \
  3=control=runs/2026-09-21/13-38-42_train/checkpoints/$S \
  3=treatment=runs/2026-09-21/14-57-30_train/checkpoints/$S
```

The outputs are `docs/acceptance/nymeria_rank1_{control,treatment}_s{1,2,3}_cuda.json`. Each holds the gate
result and, per protocol and setting, per-user and per-cell rank-1, chance, probe counts, the same-score AUC
and the implied rank-1. Scoring itself takes about 8 s per checkpoint after the index build (timed at full
scale on synthetic embeddings). Index-build peak should match `nymeria_script_pair` (1.4 GB).

4. Read the six committed files (anywhere):
   `python docs/acceptance/nymeria_rank1.py read docs/acceptance/nymeria_rank1_{treatment,control}_s{1,2,3}_cuda.json`
   This prints every protocol and setting, the three registered verdicts and implied-vs-measured, and writes
   `docs/acceptance/nymeria_rank1_read.json`. A refused checkpoint is reported and excluded from pairing,
   and the read exits 1.

Amendments go below this line, dated. The text above is not edited once the Coordinator has accepted it.

---
