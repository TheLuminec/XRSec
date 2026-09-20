# REGISTERED: retraining the zero-shot `dyn` arm after the Miami disks were lost

Written 2026-09-20 on AVALON, **before any run**, by the coordinator. DESKTOP-C executes.

## Why this arm and not another

The 23 gated checkpoints were Miami's and Miami died (`COORDINATION.md`, 2026-09-20). Every
*recorded figure* survives on `origin`; what is gone is the ability to **re-score** — for a
reviewer, and for the Questset arms A1/A2/A3, which are plumbing-complete and waiting on weights.

Of the 23, the **zero-shot `dyn` 10 s arm (3 seeds)** is the one to rebuild first:

- it is the **paper's headline** — cross-application rank-1 **0.234** at N=17 against Schach et al.'s
  published **0.180**, head-only and never trained on Across-XR;
- it is what the **Questset arms need**, and Questset exists to give the paper a second gallery size
  (N=30 beside Across-XR's N=17);
- it **trains on BOXRR + who_is_alyx only**, both of which DESKTOP-C already holds — so it needs no
  corpus transfer to start, unlike every other arm.

## The prediction, registered before the run

Seed-averaged cross-application **A1, rank-1 at N=17, Schach's own test users 32-48**, three seeds,
`dyn`, 10 s, `bilstm`, `identity_softmax`, BOXRR+alyx at 3,072 training identities, `epochs=120`,
`patience=15`, `embedding_dim=128`. Original arm: **0.234**, seed range **0.010**, user-bootstrap
CI **[0.181, 0.292]**.

| measured seed-averaged A1 | verdict |
| --- | --- |
| **below 0.19** | **FALSIFIER** — the arm does not reproduce off its original node |
| **0.19 to 0.21** | consistent with the arm, **outside its observed spread**: widens the reported interval, confirms nothing |
| **0.21 to 0.26** | **REPLICATES** |
| **0.26 to 0.28** | consistent, outside the spread: widens the interval, confirms nothing |
| **above 0.28** | **FALSIFIER** |

**band:** 0.21–0.26   **falsifier:** below 0.19 or above 0.28   **landing between them means:** the
seed average is consistent with the original arm but sits outside the spread the original three seeds
showed, so it **widens the interval we report** and is not scored as either a replication or a
refutation. *(The mandatory third line. This project has registered four bands whose measurement
landed in an unnamed region; the template line exists so that cannot happen silently.)*

## This is a REPLICATION, not a reproduction — three axes differ and are named in advance

Nothing here is expected to bite, but each is recorded so it cannot be reached for afterwards:

| axis | Miami (original) | DESKTOP-C (replication) |
| --- | --- | --- |
| device | RTX 4060 Ti, capability (8,9), **sm_89 absent from `arch_list`** — kernels ran through CUDA's compatibility path | RTX 5060 Ti, capability (12,0), **sm_120 present** — kernels run native |
| numpy | 2.5.3 | **2.4.2** |
| pair draw | — | `Generator` streams carry no stability guarantee across minors, so the manifest may differ |

CLAUDE.md prices cuDNN's BiLSTM at up to **7e-4 AUC** between devices and a *different pair draw* at
**1e-3 to 3e-3 AUC** — both far below this band. **The native-versus-compatibility split is the
larger unknown** and has never been measured here on an identification metric. If the arm lands in a
"widens" cell, that is the first thing to look at and it is **not** an excuse available after the
fact — it is written down now.

**Record `torch.cuda.get_arch_list()` and the capability tuple in the certificate**, not the device
name. The pair is what says which arithmetic path ran.

## Acceptance

1. **Three seeds, all of them, reported whichever way they fall.** A result that survived seeds run
   under a fixed rule that could have sunk it is worth more than one never tested that way.
2. **Each checkpoint writes its gate certificate to `docs/acceptance/`** and each run's row reaches
   `origin`. "It was gated" and "there is a committed certificate that it was gated" are different
   claims, and only the second survives the session.
3. **Push the numbers the moment they exist, before reasoning about them.** The standing rule, adopted
   after a 36-hour run was lost with nothing on `origin`.
4. Scoring needs the **Across-XR corpus**, which DESKTOP-C does not hold. AVALON serves it.

## The thing this must not recreate

If DESKTOP-C trains these and they live only on DESKTOP-C, **we have rebuilt the exact single point of
failure that has now cost this project twice.** DESKTOP-C has no working outbound transfer mechanism
(`COORDINATION.md`:693-703: no sshd, no share, both needing elevation). **Resolving that is a
precondition for the retrain being worth doing, not a follow-up**, and it is the user's call because
a listening service is.

---

## AMENDMENT 1 (coordinator, 2026-09-20, before any run)

**Why this is legitimate.** Both additions are **facts about the instrument, knowable without running
the experiment** — they are read out of Miami's own recorded env block and DESKTOP-C's interpreter.
Amending for that is allowed; amending because a number came back wrong is moving the line. **The
original table above stands unedited**, per the rule that a registration whose history is not visible
is not a registration. Raised by DESKTOP-C, which put the facts in its prerequisites certificate and
explicitly declined to edit someone else's registration — the correct handling.

**The axis table was incomplete: five differ, not three.** Read from the 23 Miami rows directly
(all 23 agree, no spread):

| field | Miami | DESKTOP-C | status |
| --- | --- | --- | --- |
| `code_identity` | `517cdaa57b` | `517cdaa57b` | **IDENTICAL — see below** |
| `torch_version` | **2.14.0+cu130** | **2.10.0+cu130** | **ADDED — a four-minor DOWNGRADE** |
| `python_version` | 3.13.15 | 3.12.10 | **ADDED** |
| `numpy_version` | 2.5.3 | 2.4.2 | registered |
| `device_capability` | 8.9, **sm_89 absent** from `arch_list` → compatibility path | 12.0, **sm_120 present** → native | registered |
| `cuda_version` | 13.0 | 13.0 | same |

**Torch is the addition that matters, and the direction is the point.** 2.14.0 → 2.10.0 is a
**downgrade of four minors in the library that owns the cuDNN BiLSTM kernel this arm trains.**
CLAUDE.md prices that kernel at up to **7e-4 AUC between devices** and says **nothing about between
torch versions** — so it is **unmeasured here rather than bounded**, which is a different thing from
small. **Ranked alongside native-versus-compatibility as a leading candidate** if the arm lands in a
"widens" cell, not below it. Python 3.13 → 3.12 is registered for completeness with no expectation.

**And the positive is worth more than either addition: `code_identity` is IDENTICAL.** All 23 Miami
rows record `517cdaa57b`; DESKTOP-C's tree computes `517cdaa57b`; **AVALON's tree computes
`517cdaa57b`** — three machines, two operating systems, and `model/` is uniformly LF on all of them
so the mixed-endings hazard that produced three identities for one codebase is absent. **So this is a
replication across hardware and dependencies and NOT across code**, which removes the axis that would
have been hardest to reason about and narrows what a "widens" outcome could mean.

## AMENDMENT 2: what the smoke run did and did not exercise

`rc=0` end to end on 276 identities for one epoch. **Paths exercised**: cache load (311 hits, 0
built), `encoding=dyn`, `identity_softmax` with `bilstm`, `normalize=per_dataset` over two corpora,
the validation draw (72 held out, 204 training classes), and **`evaluate()` on a corpus never trained
on** through `eval_normalize=target_fit` with AUC / EER / lookup / `position_lookup` / `amplitude`
all reported, then checkpoint, plot and results row.

**Paths NOT exercised, stated because a gate that omits this is how the last one passed**: scoring on
**CrossApplicationXR**, which DESKTOP-C does not hold (ViewGauss stood in as the unseen corpus); the
**full 4,096-identity index build**, so peak memory at real scale is unmeasured; and **`epochs>1`**,
so early stopping never ran. The 09-11 Rack run died the first time `validation_epoch_end` ever
executed, after a gate that had deliberately skipped Trainer callbacks — **state which path a gate
exercised, not which path passed.**

Identity count verified **post-draw through the loader's own `select_validation_users`**, not on the
pre-draw pool: pool 4,096 = 4,020 BOXRR + 76 alyx, draw 1,024, **training 3,072 on every seed** —
exactly the registered figure. That is the distinction that cost this project a matched arm once.
