# Cross-day identification on who_is_alyx: does what the models learn persist across days? — REGISTERED 2026-09-28, Coordinator; not run

**Question.** Every corpus behind the recent claims is a single sitting: Nymeria, Across-XR and
Questset. who_is_alyx is the one corpus whose two sessions sit on different days; its files are named
by date, typically 1-3 weeks apart. So it is the only place we can ask whether what the current models
identify survives a change of day, and how much of the same-day figure does not. User decision,
2026-09-28.

**Checkpoints.**
- Nymeria treatment, seeds 1-3.
- Exposure-breadth models: the five seed-1 checkpoints, and seeds 2-3 when they land.
- Every checkpoint is gated first on its own evaluation users by the alignment harness's gate. Tolerance: 1e-3 on CPU, 1e-4 on GPU. A CPU gap above 1e-3 goes to Miami's GPU, never to a wider tolerance.

**Users.** Each checkpoint's own alyx validation users that have two sessions on two different dates:
14, 17 and 12 users at seeds 1, 2 and 3.
- Never trained on. They chose the epoch on pooled verification AUC, which does not select on rank-1.
- The harness asserts the two dates differ.

**Metric.** Rank-1 at N = every eligible user of that seed (chance 1/N, about 0.07). It uses the A1 rule:
- Gallery: the renormalised mean embedding of a segment.
- Probe: each window of the other segment.
- Scoring: cosine, ties rank-averaged. Both directions are averaged.

Two conditions per user, same windows, same model:
- **Cross-day:** gallery on one day's session, probe on the other day's.
- **Same-day:** gallery on the first half of a session, probe on its second half, split at the median window start. Done for both sessions and averaged.

The **cost** is cross-day minus same-day, per user. It is paired on the same people and the same model.

**Pooling:**
- Treatment: (seed, user) units over three seeds, 43 units.
- Breadth: five checkpoints averaged per user within a seed.
- User bootstrap.

**Registered outcomes,** scored by where the interval falls:

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| cross-day rank-1 (treatment, pooled) | ≥ 0.20: the learned component identifies people across days | whole interval < 0.10: no cross-day identification beyond about 1.5× chance | 0.10–0.20: weak, report as such |
| cost, cross-day − same-day (treatment, pooled) | ≥ −0.15: one-sitting results carry across days at a modest cost | whole interval < −0.30: a large share of same-day identification is session-specific, so one-sitting figures overstate what persists | −0.30..−0.15: a substantial, partial cost |
| breadth − treatment, cross-day rank-1, seed-paired | reported with its interval, no band | — | — |

The third row is descriptive. It asks whether multi-application exposure changes cross-day persistence,
which this design was not built to power.

**Prior on record.** The 4096-identity `dyn` checkpoint read 0.586 at N=14 on unseen alyx users. That
figure averaged 16 windows on each side, where this protocol uses single-window probes, so a lower level
here is expected and is not a regression.

**Which outcome is strong.** The cost falsifier. It would qualify every Nymeria, Across-XR and Questset
claim as a same-sitting figure. Two limits:
- A modest cost cannot separate "a day" from "a session", because alyx confounds them. The deployment question is whether identification persists across both, so the confound does not weaken the decision.
- About 14 users per seed makes every interval wide. Pooling three seeds of the treatment is what makes the cost row resolvable at all. Expected half-width about ±0.08, which resolves the −0.30 line and may not resolve −0.15.

## Result 1 — 2026-09-28, the treatment rows (CPU, AVALON). The breadth row waits for Miami's GPU

Harness `alyx_cross_day.py`, reading `alyx_cross_day_read.py` (verdict fixtures tested both ways),
results `alyx_cross_day_s1.json` and `alyx_cross_day_read_s1.json`, committed before interpretation.
The treatment gates at seeds 1-3 read 8.8e-4, 4.0e-5 and 9.5e-4 on CPU, all within the registered 1e-3.
43 (seed, user) units, rank-1 at N = 12-17:

| quantity | measured | reading |
|---|---|---|
| cross-day rank-1 | **0.483 [0.415, 0.551]** | **band**: the learned component identifies people across days, at about 7× chance |
| same-day rank-1 (the control) | 0.743 [0.701, 0.783] | — |
| cost, cross-day − same-day | **−0.260 [−0.345, −0.181]** | the mean sits in the **partial** region. The interval excludes the band (≥ −0.15) and reaches into the falsifier region (< −0.30) |

**What it says.** A model trained on Nymeria's single sittings still identifies unseen alyx players
across a gap of days, well above chance. **But about a third of what it identifies within a session
does not persist to another day** (0.743 → 0.483), and "modest cost" is excluded. So every one-sitting
figure in this project (Nymeria, Across-XR, Questset) is a same-session figure. A cross-day figure would
be materially lower, by an amount this corpus puts at 0.18–0.35 of rank-1 at N≈14. The cost row
confounds "a different day" with "a different session", as registered. For deployment that confound
does not matter.

**Breadth (descriptive, seed 1, 3 of 5 checkpoints):** cross-day 0.408, cost −0.354 [−0.522, −0.195];
cross-day against the treatment −0.018 [−0.046, +0.008]. Multi-application exposure does not change
cross-day persistence measurably.

**Instrument fact.** synth_riders and social_vr refused the CPU gate at 1.7e-3 and 1.6e-3, and three
others sat at 8.5e-4–9.5e-4. The breadth checkpoints diverge CPU-to-GPU by more than the 7e-4 this
project documented for earlier checkpoints. As registered, all 18 checkpoints are re-scored on Miami's
GPU at the end of its chain (1e-4). That run supersedes this CPU reading for the breadth row and
cross-checks the treatment rows.

## Result 2 — 2026-09-29, GPU re-score of all 18 checkpoints (Miami, `alyx_cross_day_gpu.json`, 18/18 gates 0.0e+00)

The treatment rows reproduce the CPU reading within 1e-3: cross-day **0.483 [0.415, 0.550]**, same-day
0.743, cost **−0.261 [−0.345, −0.182]**. So Result 1's verdicts stand: the cross-day level is in the band,
and the cost sits in the partial region, excluding "modest". The two breadth checkpoints that refused on
CPU gate exactly on GPU. Breadth, five checkpoints per seed, descriptive:

| seed | cross-day | same-day | cost | cross-day, breadth − treatment |
|---|---|---|---|---|
| 1 | 0.413 | 0.759 | −0.346 [−0.511, −0.189] | −0.013 [−0.035, +0.011] |
| 2 | 0.430 | 0.723 | −0.293 [−0.435, −0.153] | −0.039 [−0.078, −0.003] |
| 3 | 0.566 | 0.807 | −0.241 [−0.398, −0.107] | −0.002 [−0.014, +0.009] |

**Multi-application exposure does not improve cross-day persistence;** it reads equal or slightly lower in
every seed. Result 2 supersedes Result 1 for the breadth row (Result 1 used 3 of 5 checkpoints at seed 1).
