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
