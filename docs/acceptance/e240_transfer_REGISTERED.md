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
