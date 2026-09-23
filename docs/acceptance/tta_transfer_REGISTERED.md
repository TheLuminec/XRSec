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
