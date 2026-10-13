# Split audit: are test users and test content both unseen, pretraining included? (2026-10-12)

Dr. Feng's rule (email 2026-10-12, point 1): a test must leave out **users and content at the same time**,
and this must hold for pretraining as well. This file audits every result family this project reports
against that rule, from the registrations and CLAUDE.md. **Nothing was re-run.**

"Content" means what the people were doing:
- application, for Across-XR and Questset;
- script, for Nymeria;
- game and activity, for alyx, BOXRR and ball-throwing;
- video or scene, for the seated 360° corpora.

Status:
- **compliant**: both users and content are unseen, pretraining included;
- **users only**: the content (or part of it) was seen in training or pretraining;
- **pending**: compliance depends on the Who-is-Alyx / Across-XR overlap answer
  (`alyx_axr_overlap_REGISTERED.md`).

| result family | test users unseen? | test content unseen? | status | what to report as the compliant figure |
| --- | --- | --- | --- | --- |
| Pooled 5-fold verification (0.669) and the static-cue tables | yes (folds) | **no**: the same corpora and videos are in training | users only | none; retired as a headline already |
| Cross-corpus transfer, BOXRR + alyx → seven seated corpora (`raw` 0.672, `dyn` 0.60-0.62) | yes | yes (different corpora and activities) | **compliant** | as reported |
| BOXRR in-domain identification (0.862, 0.948) | yes (clean users) | **no**: Beat Saber is the training activity | users only | label "seen activity"; no compliant version exists |
| alyx in-domain (0.664 AUC) and alyx cross-day (0.483 / 0.743) | yes (validation users) | **no**: Half-Life: Alyx is in training | users only | the cross-day *cost* stands; levels are seen-activity levels |
| Nymeria in-domain (constrained 0.669, rank-1 0.555) and the GNN arm | yes (48 held out) | **no**: the same 20 scripts are in training. The constrained protocol hides activity at test, not in training | users only | **leave-script-out**: 0.6145 constrained, users and five scripts held out |
| Nymeria leave-script-out | yes (25) | yes (5 scripts removed from all training) | **compliant** | as reported; rank-1 not yet computed |
| Across-XR zero-shot (A1 0.234) | yes (32-48), **pending** overlap | partly: Beat Saber (BOXRR) and Half-Life (alyx) are in pretraining | users only on 2 of 5 apps | the **unseen-activity cells**: the six ordered pairs among Superhot, Synth Riders and Social VR, already in the committed files. **A1 0.191** (0.196 / 0.182 / 0.195) against 0.234 over all 20 cells. Plus the no-alyx retrain for the users side |
| Across-XR C2-lo (0.375; +0.119 against Schach) | yes, **pending** overlap | **no** by design: all five apps seen through users 0-22 | users only | it is the exposure arm, so label it so. The compliant counterpart is P3 / breadth |
| Across-XR P3 and exposure breadth (application X held out) | yes, **pending** overlap | yes for X cells, except X = Beat Saber or Half-Life (pretraining) | **compliant** for Superhot, Synth Riders, Social VR | the X-cells of those three |
| Questset zero-shot and breadth (+0.056) | yes | three of four titles unseen; Beat Saber seen (BOXRR) | compliant on three titles | the Beat Saber split exists (+0.020 on titles no corpus covers) |
| Questset reverse (Q2) | yes | group-1 titles unseen except Beat Saber | as above | as above |
| Ball-throwing cross-day / cross-headset (0.693, -0.131, -0.235) and the broad 2 s arms | yes (corpus never trained) | yes (throwing in no training corpus) | **compliant** | as reported |
| Rack 2023 re-run | their disjoint-subject split | **no**: one game | their protocol | reported as theirs; no claim of ours rests on it |

## What follows

1. **Every figure in the paper needs this column.** Draft C's Table 2 should say users-unseen and
   content-unseen per row, and the compliant figure should lead wherever one exists.
2. **The headline AR-glasses row should move to the compliant one**, leave-script-out, once its rank-1 is
   computed. The script-matched protocol needs a 25-user, five-script variant of the rank-1 adapter.
   Until then the in-domain row is "users only".
3. **Across-XR splits along application lines.** Three of the five applications are genuinely unseen
   content for every pretraining corpus we hold. Reporting those cells separately turns the zero-shot
   and breadth results into compliant ones without a new run. For zero-shot the compliant A1 is
   **0.191**, lower than the 0.234 headline because the Beat Saber and Half-Life cells were helping.
   Schach et al.'s 0.180 averages all 20 cells, so a like-for-like placement needs their same six cells
   from their per-cell JSON (step 4).
4. **Overlap remains the one open users-side question.** It touches every Across-XR row. The no-alyx
   retrain answers it for the claims whatever the authors say; the authors' answer settles it outright.
5. **No other cross-corpus participant overlap is plausible.** The corpora come from unrelated
   institutions (Meta, Padova, Kentucky, the seated-corpus labs). BOXRR is anonymous public players and
   cannot be checked. That is recorded as a limit, not a finding.
