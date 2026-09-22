# e240 checkpoint on corpora it never saw — REGISTERED before running (2026-09-22 00:10, Coordinator, AVALON)

**User's request (2026-09-21 night):** score the e240 Nymeria-in-domain checkpoint on datasets it has never
seen — Across-XR, Questset, the smaller seated corpora — and mark which are in domain, since Beat Saber recurs.

**Checkpoints.** Treatment e240 (seed 1, 240 epochs, best 212; trained on BOXRR-23 minus 141 + who_is_alyx +
141 Nymeria; row AUC 0.7304 on its 48 held-out Nymeria users) and, as the paired comparison, **control
e240** (same seed and budget, no Nymeria; 0.5405). Both under `dyn` 10 s stride 5. Scoring on AVALON's CPU.
**Gate:** each checkpoint must first reproduce its known, twice-deterministic CPU figure on its own 48
held-out users — treatment 0.732900792, control 0.540551503 — within 1e-6, with `position_lookup_auc`
0.7232780555884043 exact; otherwise nothing from it is read.

**Protocol.** Each corpus scored on its own (own dataset id, `target_fit` normalisation, within-dataset
negatives, cross-session positives, 512 pairs per user, manifest seed from the checkpoint), reporting
verification AUC and EER beside `position_lookup_auc` and `amplitude_auc` (the training-free baselines
valid under `dyn`). The seven seated corpora are also scored **pooled** (one loader, within-dataset
negatives) for the figure the zero-shot programme reported. Across-XR and Questset get a second pass with
`cross_session_positives=false` (positives inside one recording = within one application, same sitting;
optimistic by construction, reported as such) so each carries a cross-application and a within-application
figure. 360_em yields 0 windows at `channels=full` and is skipped by the loader.

**Domain marks.** Training activities: Beat Saber (BOXRR), Half-Life: Alyx (who_is_alyx), daily life on
AR glasses (Nymeria). Every corpus below is unseen as a *corpus* (new people, rig, day); "in domain" means
the activity was in training.

| corpus | activity | domain |
|---|---|---|
| Across-XR | Beat Saber, Half-Life: Alyx, Superhot, Synth Riders, Social VR | **partly in**: 2 of 5 applications; cross-application positives span in/out |
| Questset | Beat Saber, Cooking Sim (group 1); Medal of Honor, Forklift Sim (group 2) | **partly in**: group 1's Beat Saber only; group 2 fully out |
| ViewGauss, Head_and_Gaze, VR_User_Behavior, Panonut360, PanoSaliency, EyeNavGS | seated 360-video viewing / navigation | **out** |
| NJIT_6DOF | room-scale walking | **out** (closest to Nymeria's locomotion scripts) |

**Registered outcomes, by where the number falls.**

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| treatment, seven seated corpora pooled | **0.58–0.64** (the zero-shot programme's 0.60–0.62 at 3,072 identities) | **< 0.55** (Nymeria training *hurt* transfer) or **> 0.67** (it helped — contradicting arm B's null) | 0.55–0.58 or 0.64–0.67: weakly moved; report beside the control |
| treatment − control, pooled seated (paired, same seed) | **within ±0.02** (arm B: activity diversity does not cross an activity boundary) | **beyond ±0.04** in either direction | ±0.02–0.04: weak, sign reported, not resolved at n = 1 |
| Across-XR, cross-application positives | **0.60–0.72** (two of five applications in training; the `raw` zero-shot arm read 0.70–0.73 verification on the 17 test users, `dyn` unmeasured) | **< 0.55** | 0.55–0.60 weakened; > 0.72 exceeding |
| Questset, cross-game positives, whole corpus | **0.55–0.70** | **< 0.52** | 0.52–0.55 weakened; > 0.70 exceeding |
| NJIT (walking, the Nymeria-like task) vs the seated mean | treatment − control on NJIT **≥ +0.02** larger than on the seated mean (locomotion learned on glasses helps walking) | ≤ 0 | 0 to +0.02: not resolved |
| training-free baselines | `position_lookup` per corpus within ±0.03 of the CLAUDE.md lookup table where one exists; `amplitude` 0.50–0.66 | outside | outside: the population or harness changed — stop |

Which outcome is strong: a treatment pooled figure above 0.67 or an NJIT-specific gain would be the first
data-side transfer this project has seen across an activity boundary and would need seeds 2–3 before it
is believed; the band holding is the expected, weak outcome (Nymeria buys in-domain, not transfer).
Single seed throughout — every figure here is a direction, not a measurement, until seeds 2–3 run.

## Result — 2026-09-22 00:31, both gates exact (3.0e-10 / 5.0e-10), 24 figures, 19 minutes on AVALON's CPU

`docs/acceptance/e240_transfer.json`; rows in `results/runs/avalon.jsonl` (`experiment` e240_transfer, `mode` rescore).

| corpus | domain | control e240 | treatment e240 | paired Δ | EER (t) | position lookup | amplitude |
|---|---|---|---|---|---|---|---|
| ViewGauss | out | 0.5898 | **0.6384** | **+0.049** | 0.403 | 0.938 | 0.568 |
| Head_and_Gaze | out | 0.5847 | **0.6131** | **+0.028** | 0.420 | 0.880 | 0.541 |
| VR_User_Behavior | out | 0.5469 | 0.5339 | −0.013 | 0.474 | 0.719 | 0.515 |
| NJIT (walking) | out | 0.5319 | 0.5457 | +0.014 | 0.468 | 0.645 | 0.593 |
| EyeNavGS | out | 0.5561 | 0.5569 | +0.001 | 0.462 | 0.494 | 0.500 |
| Panonut360 (tier 2) | out | 0.5437 | 0.5562 | +0.013 | 0.462 | 0.511 | 0.521 |
| PanoSaliency (tier 2) | out | 0.7422 | 0.7381 | −0.004 | 0.341 | 0.576 | 0.666 |
| **seated seven, pooled** | out | 0.6179 | **0.6210** | **+0.003** | 0.419 | 0.734 | 0.559 |
| Across-XR, cross-application | partly in (2/5) | 0.5735 | 0.5801 | +0.007 | 0.445 | 0.599 | 0.501 |
| Across-XR, within-application | partly in | 0.5987 | 0.6067 | +0.008 | 0.427 | 0.642 | 0.516 |
| Questset, cross-game | partly in (Beat Saber, g1) | 0.5474 | 0.5373 | −0.010 | 0.479 | 0.477 | 0.517 |
| Questset, within-game | partly in | 0.6275 | 0.6154 | −0.012 | 0.424 | 0.679 | 0.544 |

**Against the registration.**
- Treatment pooled seated **0.6210 — inside 0.58–0.64.** The control reads 0.6179, the zero-shot programme's 0.618 to three decimals at a different seed and identity count.
- Treatment − control pooled **+0.003 — inside ±0.02.** Arm B's null replicates at 240 epochs: Nymeria training does not cross the seated boundary.
- Across-XR cross-application **0.5801 — between (0.55–0.60): weakened**, not falsified. Within-application 0.607.
- Questset cross-game **0.5373 — between (0.52–0.55): weakened**, not falsified; the treatment is *below* the control by 0.010 here (and 0.012 within-game).
- NJIT: Δ +0.014 against a seated-mean Δ of +0.012 → **+0.001, "0 to +0.02: not resolved"**. The walking-corpus hypothesis is not supported at one seed.
- Baselines: every `position_lookup` within 0.011 of the CLAUDE.md table (ViewGauss 0.938/0.934, H&G 0.880/0.870, VR_UB 0.719/0.719, NJIT 0.645/0.653, PanoSaliency 0.576/0.583, Panonut 0.511/0.508, EyeNavGS 0.494/0.493); `amplitude` 0.50–0.67 (PanoSaliency 0.666 at the band's edge).

**Reading.** The Nymeria-trained model transfers to corpora it never saw at the same level as a model
without Nymeria — pooled +0.003 — which is the registered, expected outcome and the third time this
project has measured that a data-side lever does not cross an activity boundary. Two things are worth
more than the pooled null and both are single-seed directions: **ViewGauss +0.049 and Head_and_Gaze
+0.028**, the two best-conditioned seated corpora (real head pose, one sitting), are the same two that
identity count moved in the `dyn` curve; and **Questset moves the wrong way** (−0.010 cross-game),
the one corpus with a Beat Saber cell where the treatment gave up 141 BOXRR identities — consistent
with the swap costing a little Beat Saber coverage. **PanoSaliency's 0.74 and Panonut360 are tier-2
corpora** (a unit direction vector sits in the position slot), so those figures are gaze-direction
dynamics, not head motion, and the amplitude baseline alone reads 0.666 there. The training-free
position lookup beats both models on ViewGauss, Head_and_Gaze, VR_User_Behavior and NJIT, as it has
since 2026-09-04; under `dyn` the models cannot see it.

**Every figure is one seed.** Seeds 2–3 exist for both arms at 120 epochs and can be scored the same
way in ~40 minutes if the ViewGauss / Head_and_Gaze direction is to be believed or refuted.
