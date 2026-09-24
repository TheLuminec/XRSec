# Experiment 3 — test-time adaptation from unlabelled target motion — REGISTERED (2026-09-23, Coordinator; setup only, not run)

**Premise.** The deployment verifies one wearer on their own device, so the device can see the wearer's
unlabelled motion before it verifies anyone. The pipeline's `target_fit` normalisation is already a
label-free adaptation of the *input* statistics and is the best label-free option measured. This asks
whether label-free adaptation of the *embedding* buys anything more on a corpus the model never saw.
Alignment by an orthogonal map fitted on correspondences is closed (13× smaller than published, never
carries on `dyn`); this is a different object — a distribution match needing no correspondences.

**Variants, all computed on the same embeddings of the same pairs** (`docs/acceptance/tta_transfer.py`,
built on `e240_transfer.py`'s loaders; e240 treatment and control checkpoints; every unseen corpus plus
the pooled seated seven):

| variant | uses | what it removes |
|---|---|---|
| none | the checkpoint's own scores | — (the gate: must reproduce `evaluate()`'s AUC on the same manifest exactly) |
| **centre** | the target corpus's mean embedding over all its windows | a first-moment shift of the target's embedding cloud |
| **CORAL** | target covariance (whiten) and the source covariance (re-colour) from the checkpoint's own BOXRR training windows, first 100 users in sorted order | second-moment mismatch |

No labels, no identities, no correspondences; the transform is fitted on every window of the target
corpus, which is more than a device would see and is therefore the *upper bound* of this route.

**Registered, before running** (Δ AUC against `none`, same pairs; seated seven pooled is the headline,
each corpus reported):

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| centre − none, pooled seated | **+0.00 to +0.03** | **< −0.02** (centring destroys a cue the cosine head was using) or **> +0.06** (the transfer gap was mostly a first-moment shift — a finding, to be seeded) | +0.03 to +0.06: worthwhile, register a seeded arm |
| CORAL − none, pooled seated | **+0.00 to +0.04** | **< −0.02** or **> +0.08** | +0.04 to +0.08: worthwhile, seeded arm |
| Across-XR cross-application, Questset cross-game | same bands as pooled seated | same | same |
| treatment vs control | the adaptation gain is the same sign on both (it is a property of the route, not of Nymeria training) | opposite signs on the pooled figure | one arm inside ±0.01: not resolved for that arm |

Which outcome is strong: a gain above +0.06 on the pooled seated figure would be the first
model-side lever to move transfer here and would reframe the seated gap as distribution shift rather
than corpus ceiling; the band holding says the embedding cloud is already where it needs to be and the
route is closed at this scale. Single seed per checkpoint; the two checkpoints share one seed, so the
treatment/control agreement is a consistency check, not replication.

## Amendment 1 — 2026-09-24, RESULT: centring buys +0.01 to +0.03 per corpus, CORAL nothing; an instrument fact on the pooled line

Twenty gates on the embedding path 0.0 to 5.6e-8 (`tta_transfer.json`, rows pushed at c15f449 before this
reading). Δ AUC against `none`, same pairs, same embeddings:

| corpus | treatment: centre / CORAL | control: centre / CORAL |
|---|---|---|
| ViewGauss | +0.014 / −0.056 | +0.013 / −0.029 |
| Head_and_Gaze | +0.025 / +0.010 | +0.020 / +0.016 |
| VR_User_Behavior | +0.020 / +0.013 | +0.018 / +0.001 |
| NJIT | +0.009 / −0.022 | +0.009 / −0.015 |
| EyeNavGS | +0.029 / +0.009 | +0.021 / +0.004 |
| Panonut360 (tier 2) | +0.026 / +0.010 | +0.029 / +0.008 |
| PanoSaliency (tier 2) | −0.007 / −0.119 | −0.008 / −0.119 |
| Across-XR, cross-application | +0.016 / +0.007 | +0.017 / −0.001 |
| Questset, cross-game | +0.024 / +0.020 | +0.020 / +0.004 |
| **seated seven, pooled — per-corpus centring** | **+0.018** / −0.016 | **+0.009** / −0.017 |
| seated seven, pooled — one mean over all seven (as first computed) | −0.022 | −0.014 |

**Instrument fact, and it is why the last row exists.** The registered pooled line was first computed
by subtracting *one* mean over the windows of all seven corpora — not the operation a device performs,
which is centring on the one corpus it sees — and that line read −0.022 (treatment), which is the
registered falsifier as literally written. It applied the wrong constant to every corpus. The harness
now also computes per-corpus centring on the pooled loader (each window minus its own corpus's mean,
identical to `centre` on a single corpus), and that line is the registered quantity: **+0.018 and
+0.009, inside the +0.00..+0.03 band on both arms, same sign.** Recorded as an amendment for a fact about
the instrument, with the as-computed figure kept beside it rather than deleted.

**Reading.** Label-free centring of the embedding on the target corpus is worth **+0.01 to +0.03 on every
tier-1 corpus and both cross-application corpora, on both checkpoints** — small, consistent, and the
upper bound of this route (fitted on every window of the target). It does not change the transfer
picture: it is the size of the cross-session correction, not of the in-domain gap. **CORAL is nil to
harmful** — negative on ViewGauss, NJIT and both pooled lines, −0.12 on PanoSaliency (a tier-2 corpus
whose "position" is a direction vector, so second-moment matching to Beat Saber's covariance destroys
what it reads); its pooled figure lands in [−0.02, 0), a region the registration did not name and which
means *no gain, slight cost — route closed*. The ">+0.06 would reframe the seated gap as distribution
shift" outcome did not occur: the embedding cloud is already roughly where it needs to be, and the
seated gap is the corpora.

One seed per checkpoint, two checkpoints sharing a seed; the treatment/control agreement in sign on
every corpus is the consistency check the registration asked for. Not pursued further.
