# Reverse-direction exposure: does multi-application exposure from Questset carry to Across-XR? — REGISTERED 2026-09-28, Coordinator; not run

**Question.** Exposure to four Across-XR applications carried to Questset titles in no training
corpus: +0.056 [+0.039, +0.073] over the Nymeria treatment (`exposure_breadth_REGISTERED.md`,
Amendment 3). If the lever is "people recorded in several applications" rather than something specific
to Across-XR (its lab, rig, population or application mix), exposure from another corpus should carry
the other way. User decision, 2026-09-28.

**Arm Q2.** The Nymeria treatment, seed-matched, with Questset group 2 swapped in for the last 30 BOXRR
training users, post-draw. Group 2 is 30 people × Medal of Honor + Forklift Simulator, titles in no
corpus we train on.
- Validation users and the evaluation set (48 held-out Nymeria) are the treatment's own, so the epoch is selected on the same people.
- Training identities stay 3,072.
- Questset group 1 is in no training set.
- Generator: `questset_exposure_lists.py`. Corpus: `build_questset_subset.py`, which builds `Questset_g2` as symlinks, so its normaliser statistics sit under their own name.
- Seeds 1, 2 and 3: three runs on Miami under `gated_launch.sh`, code identity `af7cf72022`.
- Dose: 13,412 windows, about 2.1 % of training windows, against breadth's 2.56 %. The exact figure comes from Miami's loader.

**Differences from the breadth arm, all stated before the run.** Each person here contributes 2
applications, not 4. There are 30 people, not 23, from a different lab. So even if the lever is general,
a smaller effect than breadth's is expected. **The falsifier is "no effect", not "smaller".**

**Scoring,** on the training GPU:
- Across-XR: the alignment harness on all 20 ordered cross-application cells, Schach's users 32-48, N=17. No normaliser flag: Q2 holds no Across-XR statistics, so Across-XR is target-fitted exactly as for the treatment.
- Questset group 1: the Questset harness at N=30, per user. Q2's statistics sit under `Questset_g2`, so `Questset` is target-fitted, as for the treatment.
- Every checkpoint gate must pass first, on its own 48 Nymeria users.

**Registered outcomes.** Seed-paired against the treatment, per-user differences averaged over seeds,
user bootstrap. Each row is scored by where the interval falls.

| row | band | falsifier | landing between means |
|---|---|---|---|
| 1 (primary): Q2 − treatment, Across-XR A1, all 20 cells, N=17 | +0.02..+0.08: exposure carries across corpora in both directions | whole interval < 0: Questset exposure does not carry; the Across-XR result is specific to that corpus | 0..+0.02 or straddling 0: not resolved; above +0.08: exceeds, report as such |
| 2: Q2 − treatment, Questset group 1, N=30 (unseen people and titles, same rig) | +0.02..+0.10 | whole interval < 0 | 0..+0.02: not resolved; above +0.10: exceeds |
| 3 (control): Q2 − treatment, the run's own 48 Nymeria users (`selected_test_auc`) | within ±0.02 in every seed: the swap left in-domain alone | outside ±0.02 in 2 or 3 seeds: the swap moved in-domain, so read row 1 with that | outside in exactly 1 of 3: noted as seed noise at that level, row 1 read as is |

**Which outcome is strong.** Row 1's falsifier. It would say the breadth result belongs to Across-XR,
and that changes what any acquisition, OpenNEEDS included, should be argued on. The band holding is
weaker: it is also what "any 30 new people from a new corpus help Across-XR" predicts. That reading has
one measured point against it: Nymeria's 141 people alone moved Across-XR by +0.025 [−0.007, +0.057].

**Convergence check:** categorical against the treatment's own seeds. If Q2 stops on patience where the
treatment did not, or the reverse, the difference carries a budget term.

## Amendment 1 — 2026-09-28, an instrument fact found before any row existed: the loader drew 8 validation users from Questset_g2

Miami stopped seed 1 at 40 minutes and voided it: no row written, marker `.void` with the reason. The
loader printed 3,064 training identities and 1,079 validation users, against the treatment's 3,072 and
1,071. Mechanism (`dataset.select_validation_users`): a corpus with no explicit validation user enters the
`val_user_fraction` draw. This generator pinned the treatment's 1,071, all BOXRR, alyx and Nymeria, and
named none from Questset_g2. So at 0.25 the loader drew round(30 × 0.25) = 8 of the 30: g2o1u10, g2o1u11,
g2o2u01, g2o2u03, g2o2u06, g2o2u08, g2o2u12, g2o2u14. The same 8 were drawn at every seed. Reproduced on
AVALON from the composed config.

**Fix: `val_user_fraction=0` in this arm's configs.** The validation set is built from the explicit list
whenever that list is non-empty, and the fractional draw only runs above 0. So the arm now validates on
exactly the treatment's 1,071 and trains on all 30 Questset people, 3,072 in total. The treatment never
drew anything either, because every corpus it trains on is covered by its explicit list. The recorded
`val_user_fraction` therefore differs from the treatment's 0.25, and it is inert there.

**Why the registration's own check missed it:** the generator asserted counts from its own list
arithmetic, and that arithmetic never passes through the loader's function. Both generators, this one
and `exposure_breadth_lists.py`, now resolve the split with `select_validation_users` itself. They assert
the validation set equals the pinned list and the training count is 3,072. The check was verified both
ways: at 0.25 it reports the 8 drawn users and 3,064; at 0 it reports none and 3,072. All ten breadth
configs pass unchanged, which confirms the breadth arm was never affected. Miami's chain now runs the
same resolution before each launch. **The general form: a count asserted by the code that wrote the
config is a claim about the writer, not about the reader.** Only the loader's own function says what the
run will train on.

## Result — 2026-09-29, three seeds: not resolved, and at most small

`questset_exposure_read.py`, `questset_exposure_read.json` (`07a57ce`), committed before interpretation.
- Every Q2 gate is 0.0e+00 on the training GPU. The treatment's Across-XR files are GPU for seed 1 and AVALON CPU for seeds 2-3, with gates at 4.0e-5 and 9.5e-4.
- Dose, measured: 13,412 / 650,777 = **2.06 %** (Miami, seed 1's loader).
- The validation windows equal the treatment's to the window (216,884 from 1,071), so epoch selection ran on the treatment's own people.
- Convergence: neither arm stopped on patience, so the arms are matched.

| row | three seeds | reading |
|---|---|---|
| 1 (primary): Q2 − treatment, Across-XR A1, all 20 cells | **+0.014 [−0.006, +0.035]** (seeds −0.004 / +0.034 / +0.013) | **spans the falsifier, not-resolved and band regions; mean in "not resolved"** |
| 2: Q2 − treatment, Questset group 1, N=30 | **−0.006 [−0.024, +0.014]** | spans the falsifier and not-resolved regions; mean just below 0 |
| 3 (control): own 48 Nymeria users, `selected_test_auc` | −0.015 / −0.021 / −0.019 | registered rule: one seed of three outside ±0.02, "noted" |

**What it says.** Neither the strong outcome (the falsifier) nor the band is established. Exposure from 30
Questset people in 2 applications carries to Across-XR by **at most about +0.035**. The forward direction
onto Questset read +0.029 [+0.011, +0.047] (breadth − treatment, three seeds), so the two directions'
intervals overlap. **"Exposure carries across corpora in both directions" is not earned, and neither is
"it is Across-XR-specific."** The design differences registered beforehand (2 applications against 4, 30
people against 23, 2.06 % against 2.56 %) all push towards a smaller effect, and this design cannot
separate them. Row 2 is the surprise: within Questset itself (same rig, unseen people), group-2 exposure
does nothing for group 1's titles.

**Row 3 carries a pattern the registered rule does not name, so it is stated rather than absorbed into
"noted".** All three seeds are negative by a similar amount (mean −0.018). So swapping 30 BOXRR identities
for these 30 Questset people **systematically costs about 0.02 on the treatment's own in-domain Nymeria
users.** The breadth arm's 23-person Across-XR swap did not (its Nymeria figures sit at the treatment's).
It is one-directional at every seed, and it is the only row here that is.

**Consequence for the acquisition argument (OpenNEEDS).** The measured lever is large on the applications
you are exposed to: +0.108 on Across-XR's held-out applications, at scale. It is small or unresolved on
another corpus: +0.03 forward, at most +0.035 reverse. An acquisition argued on "its applications will then
be covered" has a measured basis. One argued on "it will improve transfer to unseen corpora" has a small
one at best.
