# Ball-throwing: identification across days and across headsets — REGISTERED 2026-09-30, Coordinator; not run

**Question.** The alyx check put the cost of changing day at −0.261 of rank-1 (`alyx_cross_day_REGISTERED.md`),
on about 14 people per seed and one corpus. The ball-throwing corpus (41 people, 3 headsets × 2 days, sessions
1-30 days apart; `docs/acceptance/ballthrowing_corpus_gate.json`) can test the same question independently.
It can also test one nothing else we hold can: whether identity survives a change of headset.

**Corpus facts, verified independently before registering** (Coordinator, AVALON):
- up-axis invariant 0.925-0.946 on all six session types;
- head height against stated height r = 0.822-0.944 per session type, all 41 people, confirming the index-to-id mapping;
- 41 users, 2,460 throws.
- The sample rate is **assumed** 45 Hz for all three headsets (the files ship 135 samples per throw; the paper's
  per-device rates do not match what ships). This is a stated instrument uncertainty: it affects the time scale
  the model sees, identically for every condition compared below.

**The model.** Every checkpoint we hold reads 10 s windows, and a throw is about 3 s, so a short-window arm is
trained. It is the Nymeria in-domain treatment, unchanged in composition, seeds 1-3:
- 3,072 identities, the treatment's own 1,071 validation users and 48 held-out Nymeria users (list digests equal
  to the treatment's);
- `dyn`, **`sample_time=2`, `window_stride=5`**, so the training window count, and the epoch cost, stay near the
  10 s arm's;
- 120 epochs, patience 15, code identity `af7cf72022`, on Miami under `gated_launch.sh`;
- generator `treatment_short_lists.py`, with the split resolved through the loader's own function.

Each throw gives exactly one 2 s window (2,460 in all). None of the 41 people is in any training set.
**Gate:** each checkpoint must reproduce its own recorded figure on its 48 Nymeria users (1e-4 on GPU).

**Metric.** Rank-1 at **N = 41** (chance 0.024), A1 rule: gallery = the renormalised mean of one
session's window embeddings, probe = each window of the other side, cosine, ties rank-averaged, both
directions averaged, per user. Averaged over seeds per user, then a user bootstrap. Three conditions:

| condition | gallery / probe | separation |
|---|---|---|
| C0 same session | throws 0-4 / throws 5-9 of one session, all 6 sessions averaged | minutes, same headset |
| C1 same headset, other day | Q1↔Q2, V1↔V2, C1↔C2 | 1-7 days |
| C2 other headset, other day | the 12 cross-headset session pairs | 1-30 days |

**Registered outcomes,** scored by where the interval falls:

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| C1 level | interval lower ≥ 0.10 (≥ 4× chance): identifies across days from 2 s throws | interval upper < 0.05 (≈ 2× chance): no cross-day identification at this window | weak |
| day cost, C1 − C0 | ≥ −0.15: modest | whole interval < −0.30: most same-session identification is session-specific | −0.30..−0.15: partial (alyx read −0.26 here) |
| headset cost, C2 − C1 | ≥ −0.10: a headset change costs little beyond the day | whole interval < −0.20: identity does not survive a headset change | −0.20..−0.10: substantial |

**Two confounds stated before the run.**
1. C2's day gaps are longer than C1's (mean about 10 days against 1-3). So the headset cost is also
   reported **gap-matched**: C2 pairs at ≤ 3 days apart against C1. That is descriptive, and the
   registered row stays the unmatched one.
2. Head height is a real biometric here (r ≈ 0.9), so a training-free **height-only lookup** is
   scored beside the model in C1 and C2: mean recorded head y, standardised per session type, which
   removes each headset's scene-origin offset. It is descriptive, and it is what a static cue alone
   buys across days and headsets. `dyn` removes it from the model.

**Resolution.** 41 users, paired within user; expected interval half-width on differences about ±0.05. That
resolves the −0.30 and −0.20 lines and may not resolve −0.15 or −0.10.

**Which outcome is strong.** The headset falsifier. It would say the learned signature is tied to a tracking
system, which matters directly for the "all of XR" scope. A band holding on the headset row is weaker: it is
also what "everything is at chance anyway" predicts if C1 is weak. So the C1 level row is read first.
