# Proposal: strided windows and multi-window scoring

Status: **Part 2 approved and implemented** (`mode=curve`, `model/templates.py`).
Part 1 implemented, with the temporal-separation guard, ahead of the k-curve that
was going to gate it. The gate was "do more windows help?"; the reason for building it
now is different - `sample_time` cannot be swept honestly without a stride, because
lengthening a window shortens the example count at the same time. The k-curve still
decides whether the extra windows are worth using; it no longer decides whether the
mechanism exists.

The secondary fix in this section (interpolating rather than picking the nearest
frame) shipped separately as `resample: bin`, using bin-averaging rather than the
slerp described below - averaging is the anti-aliasing filter the decimation case
actually needs, and empty intervals are interpolated.

### Bound on what Part 2 can achieve — noticed during implementation

A template is drawn from **one session**, which is the correct choice: mixing sessions
inside a template would average away the very session variability that cross-session
evaluation exists to expose. But it follows that averaging k windows reduces only the
**within-session** component of the error and leaves the **between-session** shift
untouched — and the between-session shift is precisely what makes cross-session
verification hard.

So the gain from k is bounded by how much of the error is within-session noise rather
than between-session offset. If the curve flattens early, that is not a failure of
the implementation; it is a measurement saying the error is dominated by between-
session variation, which would itself be worth knowing and would point at
session-invariant representations rather than more evidence per decision.

A first run on a deliberately undertrained checkpoint (3 epochs, 5 held-out users,
640 pairs) came out flat: AUC 0.710 at k=1 against 0.712 at k=16. That is far too
noisy to conclude from and is recorded only so nobody reads the first real curve as a
surprise.

## What the pipeline does now

`Sampler` cuts each session into **consecutive non-overlapping** windows of
`sample_time` seconds, each holding `sample_time * sample_rate` frames chosen by
*nearest-neighbour lookup* against target times. Every window becomes an independent
example, and verification compares exactly **one window against one window**.

Five consequences, all measured or read directly off the code:

1. **Non-overlapping windows discard most of the available examples.** A 45-minute
   session at 5s windows yields 540 windows. At a 1s stride it would yield ~2700.
   Data has been the binding constraint on the behavioural component (headroom
   doubled from 48 to 343 identities while the anthropometric part stayed flat), so
   a 5x multiplier on windows-per-identity is not a small thing.
2. **Nearest-neighbour resampling fabricates motion above the native rate.** At
   `sample_rate=20`, ViewGauss (10.1Hz) is 50.5% exact duplicate consecutive frames
   and PanoSaliency (16.5Hz) is 25.9%. Derived velocity there is partly invented.
3. **Windows are aligned to session start**, so boundaries fall arbitrarily relative
   to behaviour. A movement pattern straddling a boundary is split across two
   examples and appears in neither cleanly.
4. **One window is one decision.** `sample_count = floor(duration / sample_time)`,
   and the tail is dropped. More importantly, no real verification system decides
   from a single 5-second window; it aggregates evidence. We are measuring the
   hardest possible operating point and reporting it as *the* number.
5. **The cache stores fixed windows**, so any jitter or re-striding invalidates it —
   which is fine, but it means the current design cannot be augmented in place.

## Two separable changes

They are independently valuable and independently testable. **Part 2 is cheaper,
higher-expected-value, and should go first.**

### Part 2 (do first): multi-window scoring

Aggregate `k` windows per side into one template before comparing, instead of
comparing single windows.

```
                       current (k = 1)
    window A ──► extractor ──► e_A ─┐
                                    ├──► cosine ──► logit ──► same / different
    window B ──► extractor ──► e_B ─┘

                       proposed (k = 4)
    A1 A2 A3 A4 ──► extractor ──► e1 e2 e3 e4 ──► L2-normalise each
                                                   ──► mean ──► L2 ──► template T_A ─┐
                                                                                     ├─► cosine
    B1 B2 B3 B4 ──► extractor ──► f1 f2 f3 f4 ──► L2-normalise each                  │
                                                   ──► mean ──► L2 ──► template T_B ─┘
```

Rules that make it honest:

- The `k` windows on each side come from **one session**, and the two sides come from
  **different sessions of the different-or-same user** exactly as
  `cross_session_positives` already requires. An enrolment template drawn across two
  sessions would leak session variability into the template and flatter the score.
- Normalise each embedding **before** averaging, then renormalise the mean. Averaging
  unnormalised embeddings lets a single large-magnitude window dominate the template.
- Report a **curve over k ∈ {1, 2, 4, 8, 16}**, with k=1 being the existing number.
  This is an added dimension, not a replacement metric. Nobody should be able to say
  we quietly changed what we were measuring.

**This needs no retraining.** Templates are built from embeddings of an already
trained model, so the whole k-curve costs one forward pass over the evaluation set
per checkpoint. Every existing checkpoint can be re-scored. That is the single
cheapest large experiment available to us.

**Why it should work.** This is standard practice in speaker verification, where
going from one utterance to ten typically halves EER. Averaging k independent noisy
observations of a stable per-person quantity reduces the noise by roughly √k while
leaving the signal intact.

**Predictions, falsifiable.** From EER 0.34 at k=1 I would expect **0.25–0.28 at
k=8**. If EER improves by less than ~0.02 at k=8, the per-window embeddings are
correlated rather than independently noisy, and the ceiling is the representation
rather than the evidence available — which would be a genuine and useful negative.

**It also discriminates the anthropometry question, from a new direction.** Absolute
head height is a *static* property, so averaging should clean it up substantially.
Behaviour varies genuinely between windows, so averaging helps it less. If the
uncentred arm gains much more from k than the centred arm does, that is independent
confirmation that the dominant signal is a body measurement. The **shape** of the two
curves is the measurement, and it needs no new training runs.

### Part 1: strided rolling windows

Add `window_stride` in seconds, defaulting to `sample_time` (exactly today's
behaviour). `window_stride < sample_time` produces overlap.

```
    session ├──────────────────────────────────────────────────┤

    now      [ w0 ][ w1 ][ w2 ][ w3 ]              stride = sample_time
    proposed [ w0 ]
                [ w1 ]
                   [ w2 ]                          stride = sample_time / 2
                      [ w3 ]  ...
```

**This is dangerous without one specific guard, and the danger is the exact class of
bug we spent today removing.** Two windows overlapping by 80% share most of their
frames. A positive pair drawn from two overlapping windows is close to a self-match:
trivially easy, and it would inflate the result invisibly, because held-out positives
would be inflated the same way and no train/test gap would appear. That is the same
shape as the same-session shortcut and the label-imbalance skew.

So Part 1 requires, non-optionally:

- **Window start times tracked per window**, alongside the session ids already
  carried in `SampleIndex.window_session_ids` (cache version bump).
- **Pair generation enforcing a minimum temporal separation** — at least
  `sample_time` between the two windows' starts, so no pair ever shares a frame.
- **A test asserting no paired windows overlap**, in the same family as
  `test_within_dataset_negatives_never_cross_datasets`.

**Secondary fix worth folding in:** resample by **linear interpolation** rather than
nearest-neighbour, and refuse (or warn loudly) when `sample_rate` exceeds a dataset's
native rate. This removes the 50.5% duplicate-frame problem at source instead of
leaving it as a caveat in the docs. Quaternions should be interpolated with slerp,
not componentwise linear interpolation followed by renormalisation, if we do this
properly.

**Prediction.** Modest, and possibly negative. Overlapping windows are correlated, so
5x the windows is nowhere near 5x the information, and correlated examples can make
overfitting faster rather than slower. Honest expectation: **−0.01 to +0.02**. Worth
doing mainly because Part 2 needs enough windows per session to build templates at
k=16 without exhausting short sessions — ViewGauss sessions are only ~4 per user and
420 windows total across 35 users.

## Cost

| | implementation | GPU |
| --- | --- | --- |
| Part 2 | ~150 lines + tests, in `eval.py` and a new template builder | one forward pass per checkpoint per k; **no retraining** |
| Part 1 | ~200 lines + tests, in `sampler.py`, `dataset.py`, cache v4 | one sweep to measure; retraining required |

## What I would ask for

1. Approve **Part 2** now. It is cheap, needs no retraining, gives a much more
   defensible operating point, and its k-curve independently tests the anthropometry
   conclusion.
2. Hold **Part 1** until Part 2 lands, then decide. If Part 2 shows the k-curve
   flattening early, more windows will not help and Part 1 is not worth the risk of
   introducing an overlap leak.
3. If Part 1 is approved later, treat the temporal-separation guard as part of the
   feature and not a follow-up. Shipping strided windows without it would manufacture
   a new inflation bug in a project that has just spent a day removing three.
