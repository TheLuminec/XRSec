# Template verification: the evaluation protocol (fixed 2026-10-12, before any checkpoint was scored with it)

## Why this exists

Dr. Feng (email, 2026-10-12, point 2) described the deployment decision. A user stores an enrolment
embedding, and each new capture is compared with it to accept or reject the claim "this is the same
person". Most of what this project reports is something else:
- window-pair verification AUC (are two windows the same person?);
- rank-1 identification against a gallery.

Neither is the decision. This protocol computes that decision on the same embeddings and templates as our
rank-1 (A1) figures, so the three can be read side by side. It is a measurement, not a hypothesis test, so
it registers no bands. Any comparison made with it later registers its own.

## The decision

| element | definition |
| --- | --- |
| template | renormalised mean of a user's L2-normalised enrolment-window embeddings (`centroids`, as A1) |
| attempt | one probe window against one claimed template; cosine score |
| genuine attempt | probe against its own user's template |
| impostor attempt | probe against another user's template, every other user in the population |
| decision | accept when the score is at or above a threshold t |
| threshold | **never fitted on the people it is scored on**: users are split into two random folds; t is fitted on one fold (impostors drawn only from inside it) and applied to the other, then swapped; 20 repeats, seed 67 |

The probe is one window, 10 s for the 10 s models and 2 s for ball-throwing. That is the same unit as A1
and as Schach et al.'s single-window figure.

## What is reported, per condition

| field | meaning |
| --- | --- |
| `eer` | equal error rate over all users' attempts. Threshold-free, so it is comparable to AUC, but it is not a deployable figure |
| `crossfit_eer_threshold` | FAR, FRR and balanced accuracy at a threshold fitted at the EER on the other fold. **This is the deployment figure** |
| `crossfit_far1pct` | TAR at a threshold fitted for FAR = 1 % on the other fold, with the FAR actually realised on the held fold. The security-side operating point |
| `per_user_frr` | each user's false-reject rate at the cross-fitted EER threshold. The distribution the mean hides; Dr. Feng's per-user point |
| `rank1_per_user` | A1 from the same score matrix. It is the plumbing gate, below |

**Conditions.** Each condition is a choice of where enrolment and probe come from:

| corpus | conditions |
| --- | --- |
| Across-XR, users 32-48 | cross-application (20 ordered application pairs, pooled), within-application (first half / second half) |
| ball-throwing, 41 people | C0 same session, C1 same headset other day, C2 other headset |
| alyx, each seed's held-out two-date users | cross-day, same-day (first half / second half) |
| Nymeria | not yet: the script-matched gallery needs its own adapter (next version) |

## Gates

1. **The checkpoint gate**: each checkpoint reproduces its own recorded figure on the device used,
   1e-4 on cuda and 1e-3 on CPU.
2. **The plumbing gate**: the rank-1 computed from the very score matrices this protocol uses must equal
   the committed per-user rank-1 figures for that checkpoint and device exactly (gap < 1e-6). If it does
   not, the file is written as refused and nothing in it is read.
   - It passed in the AVALON CPU smoke test on alyx seed 1, at a gap of 0.0 against
     `alyx_cross_day_s1.json`.

The scoring core is fixture-tested (`template_verification_fixture.py`):
- a separable corpus gives EER 0 and TAR at 1 % FAR above 0.995;
- an identity-free corpus gives EER about 0.5 and TAR about equal to FAR;
- impostor counts are genuine × (N-1).

## What it is not

- **Not a replacement for the registered figures.** A1 and the constrained AUC stay the registered
  quantities of their arms. This protocol adds the deployment view of the same models.
- **Not comparable across corpora without N.** The impostor population is the corpus's own test users
  (17, 41 or 12-17), and a larger population makes the FAR = 1 % point harder.
- **Not pooled across corpora.** A deployment has one population, so every figure is per corpus and per
  condition.
