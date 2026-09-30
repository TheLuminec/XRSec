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

## Result — 2026-09-30, three seeds, GPU

The treatment_2s arm, 3 seeds, all gates 0.0e+00 on its 48 Nymeria users; each scoring asserted 41 users
and 2,460 windows. Read by the pre-committed `ballthrowing_cross_day_read.py` (`c7653c2`), result
`ballthrowing_cross_day_read.json` (`0d3d9b8`). Rank-1 at N=41, chance 0.024.

| quantity | measured | reading |
|---|---|---|
| C0, same session | 0.824 [0.784, 0.857] | — |
| **C1, same headset, other day** | **0.693 [0.649, 0.735]** | **band**: about 28× chance, from single 2 s throws |
| C2, other headset, other day | 0.458 [0.409, 0.509] | — |
| **day cost, C1 − C0** | **−0.131 [−0.172, −0.089]** | the interval spans "modest" and "partial"; the mean sits in "modest" |
| **headset cost, C2 − C1** | **−0.235 [−0.282, −0.191]** | the interval spans "substantial" and the falsifier region; the mean sits in the falsifier region. The band (≥ −0.10) is excluded |
| gap-matched headset cost, C2 pairs ≤ 3 days apart − C1 (38 users; descriptive) | −0.142 [−0.206, −0.082] | about 0.09 of the unmatched cost is the longer day gap |
| height-only lookup (training-free; descriptive) | C1 0.099, C2 0.095 | about 4× chance, and unaffected by the headset change once each session type is standardised |

Per session pair (seed means): the same-headset pairs read 0.665-0.722, Quest-Vive 0.478-0.587,
Vive-Cosmos 0.463-0.511, and Quest-Cosmos 0.341-0.379. The ordering also follows the day gap
(Quest-Cosmos pairs are the furthest apart, about 15 days on average), which is why the gap-matched row
exists.

**What it says.**
1. **People are identified across days from a single 2-second throw**, at 0.69 among 41. That is a
   second corpus, after alyx, where the learned `dyn` component persists across days. The day cost here
   (−0.13) is about half of alyx's (−0.26), on a far more stereotyped task.
2. **Changing headset costs substantially more than changing day.** About −0.24 unmatched and −0.14
   gap-matched, so identity survives a headset change well above chance (0.46) but not intact. The
   falsifier ("does not survive") is not met: its line sits inside the interval, and the level stays at
   19× chance.
3. The static height cue is weak here (0.10) but device-robust once standardised. The model's
   device-sensitivity is in the dynamics it reads, not in a static offset. **One mechanism hypothesis, not
   tested:** `dyn` keeps absolute pitch and roll (gravity), and how a headset sits on the head differs by
   model, so part of the headset cost may be fit rather than behaviour.

**Qualifications.**
- The model is a 2 s model that reads only 0.55 AUC in domain on Nymeria, against 0.708 at 10 s. These
  figures are from a weaker model and a highly stereotyped, phase-aligned task: each window is one whole
  throw.
- The 45 Hz rate is assumed.
- The headset order was fixed (Quest, then Vive, then Cosmos), so headset and elapsed time are not fully
  separable even gap-matched.
- 41 people.
