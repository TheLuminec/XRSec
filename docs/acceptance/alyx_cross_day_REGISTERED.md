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
