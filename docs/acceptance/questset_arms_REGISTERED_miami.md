# Questset arms A1/A2/A3 — transcribed on the node that will run them

**Registered by XRSec Coordinator before this corpus reached this disk; transcribed here so
the node executing the arms holds the bands it will be judged against.** The Coordinator's
copy is authoritative — if these disagree, theirs wins and this file is the error.

Corpus verified here first: 123/123 files by sha256, corpus gate PASS
(`questset_corpus_gate_miami.json`). Licence **CC BY 4.0** — no clause 4, no clause 15, no
cloud prohibition; and this corpus's freedom must not leak onto BOXRR-derived files.

## A1 — zero-shot cross-application rank-1

Head-only `dyn`, 10 s, the existing 3-seed zero-shot checkpoints, **never trained on any
Questset byte**. Gallery = game A, probe = game B, unseen users, both ordered directions,
each group separately.

| outcome | reading |
|---|---|
| **< 0.10** | **FALSIFIER** — the Across-XR zero-shot result does not survive a change of corpus. The outcome that most needs reporting |
| 0.10–0.15 | weakened; report as such |
| **0.15–0.40** | **registered band** |
| > 0.40 | above band; report as exceeding |

Report at **matched N=17 AND at N=30**. The N=30 figure is why this corpus was acquired —
every interval in the paper is N=17-limited.

## A2 — the clean behavioural arm (group 2) against group 1 as contrast

**Registered:** raw minus `dyn` is **smaller on group 2** than on group 1, because the games
have already removed the height cue there (group 2 height P = 0.493, chance; group 1 = 0.718).
**Falsifier:** group 2's raw-minus-`dyn` is equal to or larger than group 1's.

Within-corpus, within-pipeline, so far better powered than any cross-corpus contrast we hold —
and we did not own one before this corpus.

**Two confounds found during the corpus gate, to be carried into the write-up rather than met
in the residuals:**

1. **All five ~112–116 Hz sessions sit in group 1** — one entire side of this contrast. The
   other 115 sessions are 59.93 ± 0.28 Hz, so the five are a separate population, not a tail.
   **If group 1 comes out different for an unregistered reason, check this first.**
2. **`g1o2u07` holds `beat_saber` at 59.9 Hz and `cooking_simulator` at 114.0 Hz** — for that
   single identity a cross-application arm draws **gallery and probe at different native
   rates**. The only place resampling acts *within* a person rather than between them, and the
   only identity whose own pair is not internally comparable.

## A3 — covered/uncovered control

Beat Saber is in our BOXRR pretraining; Cooking Simulator, Medal of Honor and Forklift
Simulator are in nothing we hold. **Registered directional:** cells involving Beat Saber do not
read higher than uncovered cells by more than **+0.05** — what P3's Synth Riders control
established on Across-XR.

## Gates — held as gates, not as checks

- **Corpus gate** on the files received here, not the ones sent. **DONE, PASS.**
- **Checkpoint gate:** each checkpoint reproduces its own recorded figure to **< 1e-4**, and
  **the gap is written down**, before any Questset number is quoted. A number from an ungated
  checkpoint is not a result. *(This node has twice recorded a safeguard as present when it was
  not; the difference here is that the gap gets recorded rather than asserted.)*

## How the result is reported — user decision, 2026-09-16

> "I think we should include the cross task needing the same posture but we won't focus on it
> too much."

**The decision splits, and only one half is discretionary.**

1. **The qualifier is OBLIGATORY.** Wherever the paper says head height is the static cue that
   survives an application change, it carries **"across applications that share a posture"**.
   Without it the sentence is **false**: group 2 reads height P 0.493 and a height-only lookup
   of 0.033 at N=30, dead on chance. Correcting a claim we already make is a different act from
   promoting a new finding, and "don't focus on it too much" governs only the second.
2. **The finding is BUDGETED**, fixed now so it cannot creep: one subsection in the static-cue
   audit (one paragraph + the two-row table), one sentence in limitations noting it needs a
   registered replication. **No abstract, no contribution bullet, no figure of its own.**

**A2's design does not change — only where its result lands.** It is evidence inside the audit
subsection, not a section of its own.

**Why bounding it is right rather than timid:** one corpus, acquired for another purpose, with
its registration written the same day it ran. The registration was genuine and the falsifier
genuinely fired, so it is reportable — but a finding that could carry a paper needs a prediction
registered *before* the corpus exists, plus a replication. **Reporting it small is what keeps it
available to report large later; featuring it now on one corpus would spend it.**

## One caveat from the corpus gate that touches any orientation-derived feature

The device-local-+Y-into-world invariant is **title-dependent**: 0.984 beat_saber, 0.950
medal_of_honor, 0.938 forklift_simulator, **0.866 cooking_simulator** — a 0.12 spread that a
pooled figure hides. 0.866 is a game of looking down and reaching, not a defect. Expect any
orientation-derived feature to behave differently across the four titles, and quote per-title.
