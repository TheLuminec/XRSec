# Nymeria leave-script-out (LSO): unseen people doing unseen tasks on AR glasses — REGISTERED (2026-09-23, Coordinator)

**Question.** The in-domain arm showed the model reads a person's motion across activities it has seen
(constrained 0.669, three seeds). Does that survive activities it has *never* seen from anyone? This is
the cross-task claim on the target device, with 236 people.

**Held-out scripts, chosen by rule before anything ran:** among all 5-script subsets holding ≤ 25 % of
sequences, maximise the number of held-out people with ≥ 2 held-out-script recordings (so their
positives are cross-script by construction); tie-break on ≥ 1. Winner: **S10-Housekeeping, S13-Charades,
S2-Where_is_X, S5-Workout, S6-Dance** — 272 of 1,100 sequences (25 %), locomotion, a party game,
housework, exercise and dance. Every non-held-out user keeps ≥ 1 training script.

**Corpora** (`nymeria_lso_build.py`, hard links from the verified corpus, manifest with sha256 at
`nymeria_lso_manifest.txt`): `Nymeria_LSO_train` — 236 users, 828 recordings of the 15 training scripts;
`Nymeria_LSO_test` — the **25** of the 48 held-out people who have ≥ 2 held-out-script recordings, **53**
recordings of the 5 held-out scripts only.

**Arm (`configs/nymeria_lso_s{1,2,3}.yaml`, from the in-domain arm's reference lists):** train on BOXRR
minus the same 141 users + alyx + `Nymeria_LSO_train` with the same 48 people excluded and the same
1,071 pinned validation users → **3,072 training identities, 141 of them Nymeria, exactly the treatment's
people with 5 of their 20 scripts removed**. Evaluate on `Nymeria_LSO_test` (`test_dirs`,
`test_on_excluded=false`): 25 unseen people, 5 unseen scripts, cross-recording positives = cross-script.
`dyn`, 10 s, stride 5, 120 epochs, patience 15, seeds 1–3, otherwise the in-domain config.

**Reference figures on the SAME test set, no training needed:** the in-domain **treatment** checkpoints
(saw all 20 scripts from the 141 training people; never these 25 people) and the **control** checkpoints
(no Nymeria), seeds 1–3, scored on `Nymeria_LSO_test` through `nymeria_lso_score.py` — the standard
figure plus the constrained one (cross-script positives / same-script negatives), each checkpoint first
reproducing its own row's figure on its own held-out set. So three models on one test set:

| model | people | scripts | reads |
|---|---|---|---|
| control | unseen | unseen (never any Nymeria) | zero-shot floor |
| treatment | unseen | **seen** from other people | cross-person, same tasks |
| **LSO** | unseen | **unseen** from anyone | cross-person, cross-task |

**Registered outcomes** (constrained AUC on the 25, seed-averaged; the row AUC beside it):

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| treatment on LSO_test, constrained | 0.62–0.72 (the in-domain 0.66–0.68 on a 25-person subset of the same people, held-out scripts only) | < 0.58 | 0.58–0.62: these 5 scripts are harder than the corpus average; read LSO against it, not against 0.669 |
| **LSO − treatment, constrained, paired by seed** | **−0.06 to 0.00** — unseen tasks cost something, most of the motion cue survives | **< −0.10** — the cue is task-specific: the model learned how these people do *these* activities, not how they move | −0.10 to −0.06: substantial task-dependence; report as such |
| LSO − control, constrained | ≥ +0.10 | ≤ +0.03 — LSO reads as zero-shot; no cross-task signal | +0.03 to +0.10: weak cross-task signal |
| `position_lookup_auc`, `amplitude_auc` on LSO_test | identical across the three models within a seed (same users, same pairs) | any difference | — |

**Which outcome is strong.** The falsifier on the second row: it would say the credited in-domain result
is task-bound, which would change the paper's sentence from "how the person moves" to "how the person
does the activities it was trained on". The band holding says cross-task transfer within one device
exists and prices it. Note the asymmetry named in advance: a *large* LSO gain over control together with a
large deficit against treatment is consistent — motion generalises across tasks and the tasks still
matter — and both numbers are reported.

**MDD:** three paired seeds at the in-domain arm's constrained sd (0.013) resolve ~0.03; 25 people rather
than 48, so expect roughly 1.4× that. The bands are wider than the MDD; the −0.10 falsifier edge is not
close to the band's −0.06 edge but the interval, not the point, decides.

**Launch conditions on Miami:** rebuild the two LSO trees with `nymeria_lso_build.py --verify` on the
verified Nymeria corpus and match the manifest's sha256 per file; `nymeria_lso_lists.py --seed 1 --seed 2
--seed 3` and match the printed digests (held 8842e13112a7; V and dropB per seed as in the in-domain
JSONs); print the composed config before each launch (`experiment_name` nymeria_lso, `test_on_excluded`
false, `test_dirs` = LSO_test); `gated_launch.sh`, one seed at a time, rows pushed as they land; nothing
in `model/` pulled while a run is alive. Reference scoring of the six in-domain checkpoints on LSO_test
runs on Miami's GPU (same device as their rows) after the arm, or on AVALON's CPU with the documented
tolerance, whichever is idle.

## Amendment 1 — 2026-09-23, seed 1 launched on Miami; two unnamed regions named before any number exists

Miami's launch: rebuilt LSO trees byte-identical to the committed manifest (882 lines, diff empty), all
nine list digests reproduced, composed config asserted by the runner (`data_dirs` names
`Nymeria_LSO_train`, `test_dirs` names `Nymeria_LSO_test`, `test_on_excluded` false), and one check this
registration did not ask for and should have: **all 48 `exclude_users` paths resolve under
`Nymeria_LSO_train` and none under `Nymeria_Dataset`, all 25 test people are among them, none of the 25
is a validation user.** Had the exclusions still named the original corpus, nothing would have matched
and the 25 test people's *training-script* recordings would have been trained on — a leak no counter
reports. Verified on the composed config, not by reading the generator.

**Two regions the registration left unnamed, named now (seed 1 has no number yet):**

| quantity | region | **means** |
|---|---|---|
| LSO − treatment, constrained | **> 0.00** | removing five scripts acted as regularisation rather than deprivation: 15 scripts from the same people generalise to unseen tasks *at least as well* as 20. Not a gain to credit — the treatment is the like-for-like referent — but a finding that task breadth in training is not what carries the cue. Above +0.03, report it as exceeding and seed it before it is quoted. |
| treatment on LSO_test, constrained | **> 0.72** | these five scripts are *easier* than the corpus average for the treatment; read LSO against the treatment's own figure as before, and say so. |

Seeds 2–3 chained behind the same assertion gate; 2b4d7ba is pulled between seeds, never during one.

## Amendment 2 — 2026-09-23, RESULT: the falsifier did not fire — the in-domain cue is not task-bound

Rows and Miami's GPU scoring at 8663165 (`miami-server`); checkpoints on AVALON at
`exchange_from_miami/nymeria_lso/`, hashes verified both ends. All nine gates on the device that wrote
the rows: 0.0 to 6.1e-8. Constrained AUC on `Nymeria_LSO_test` (25 unseen people, 5 unseen scripts,
11,701 windows, 5,120 + 5,120 cross-script / same-script pairs):

| model | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| control (no Nymeria) | 0.5043 | 0.4989 | 0.5147 | **0.5060** |
| treatment (20 scripts seen from other people) | 0.6315 | 0.6805 | 0.6595 | **0.6572** |
| **LSO (15 scripts; these 5 never seen from anyone)** | 0.5947 | 0.6335 | 0.6153 | **0.6145** |

| registered quantity | measured | verdict |
|---|---|---|
| LSO − treatment, paired | −0.0368 / −0.0470 / −0.0442, mean **−0.0427, CI [−0.0558, −0.0296]** | **whole interval inside the band −0.06..0.00**; the −0.10 falsifier nowhere near. Clean. |
| treatment on LSO_test | 0.6572, every seed inside 0.62–0.72 | these five scripts are neither easier nor harder than the corpus; LSO is read against the treatment's own figure |
| LSO − control | mean **+0.1085, CI [+0.0510, +0.1660]** | point inside the ≥ +0.10 band, interval's lower edge in the +0.03..+0.10 region: **point-in-band, interval-straddling** — the interval decides, and at n = 3 this is the row the design resolves least well; falsifier (≤ +0.03) excluded |
| baselines across the three models within a seed | 0.7817/0.5373, 0.7803/0.5441, 0.7809/0.5395 — identical | same people, same pairs |

**The sentence this buys:** unseen people doing five tasks no training identity ever performed separate
at **0.61** against a zero-shot floor of **0.51**, having cost **0.043** against the same model tested on
tasks it had seen from other people. The credited in-domain result is about how the person moves, not
how they do the specific activities — the cross-task claim on the target device.

**A qualifier this result adds to the record, Miami's observation.** The control reads **0.506 — chance
—** here, where on the in-domain test set the same three checkpoints read 0.472, which was written up as
"8–9 SE below chance, anti-identifying once activity is matched". So that residual is a property of
*that* test construction (48 people, all 20 scripts, the arm's pair draw), not a constant of the
zero-shot model; the sentence carries its test set from now on. It does not touch the LSO reading — the
floor is the floor either way. Housekeeping: Miami's JSON labels two byte-identical control-s1
checkpoints (the `03ea8e2376` run and the `af7cf72022` acceptance) under one key; they scored identically
to every digit, a free consistency check, not ten distinct checkpoints.

Three seeds, one test set of 25 people, one sitting per participant — the usual caveats. A CPU
cross-check of the nine figures on AVALON follows for the cross-machine record.
