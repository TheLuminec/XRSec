# Audit: does PAPER_DRAFT §5-§6 state a verdict its registration did not license?

**Run 2026-09-17 on LAPTOP-C (`xrsec-f9`), repo at `eac83fe`. READ-ONLY — nothing in
`docs/PAPER_DRAFT.md` was edited.** Commissioned after G1 (a stale verdict in PAPER_PLAN) to point the
same defect class at the document that actually goes out. Referent:
`docs/acceptance/across_xr_alignment_RESULTS.md`, whose verdict columns are treated as authoritative.

## Headline

**Every verdict the draft states on a resolved registration is earned.** P1 "held", both BEAT verdicts,
the alignment band exclusion, the P3 falsifier exclusion and its stricter threshold "not met" all match
RESULTS cell for cell. §6.8 and §6.9 are exemplary — three registered failures reported, an
unregistered observation labelled exploratory, and a coverage reading explicitly withdrawn.

**The defect is not a wrong verdict. It is a missing one.** A registered prediction **failed** and the
draft does not report it anywhere, while the half of the same registered pair that **held** is reported
twice and carries a Discussion claim. In a paper whose stated contribution 6 is *"negative outcomes
reported as prominently as positive ones"*, that asymmetry is the finding.

## F1 — a registered prediction failed and is absent from the draft (highest severity)

Amendment 4's addendum registered a **pair** of task-structure predictions. RESULTS scores them
opposite ways:

| registered | measured | RESULTS verdict |
| --- | --- | --- |
| rhythm games carry best | Beat Saber +0.084, Synth Riders +0.077 | **holds** |
| **Social VR carries least, below +0.03** | **+0.044 [−0.000, +0.084]; Alyx is least at +0.026** | **fails** |

RESULTS states the consequence in one sentence: *"the task-structure ordering **holds at the top and
not at the bottom**."*

**What the draft does with each half.** The rhythm half appears twice — §6.3 (*"the two largest gains in
both of our arms are the rhythm-game pair"*) and §7 (*"Task structure matters more than task count"*).
The Social VR half appears **nowhere**. `grep -n -i "social vr"` over the draft returns three hits: the
corpus description (line 185), the pretraining-coverage asymmetry (line 242), and a bare `+0.038` row in
the §6.4 table (line 555). **The draft never says that a prediction was registered about Social VR, nor
that it failed.**

**Robust to the re-seeding, checked both ways.** The prediction fails on *both* clauses under *both*
vintages of the number:

| | Social VR value | is it least? | is it below +0.03? |
| --- | --- | --- | --- |
| RESULTS seed-1 table | +0.044 | no (Alyx +0.026) | no |
| draft's seed-averaged table | +0.038 | no (Alyx +0.026) | no |

So this is not an artefact of which table is current. Note the draft's own §6.4 table contains the
refutation — Alyx at +0.026 sits below Social VR at +0.038 — so the data needed to report the failure is
already printed on the page; only the registration and its verdict are missing.

## F2 — §7 generalises the ordering to the end where it failed

§7, lines 806-810: *"a gallery collected in a rhythm game generalises well to another rhythm game **and
poorly elsewhere**."*

The first clause is the half that held. **The second clause is the half that failed**: Social VR was
predicted to carry least and carries about the mean, and the least-carrying application is Half-Life:
Alyx, which is a *covered* pretraining activity rather than a structurally distant one. RESULTS' "holds
at the top and not at the bottom" is precisely a denial of "and poorly elsewhere".

This is F1's consequence rather than a separate defect, and it is the one that would reach a reader: the
supportable claim is *"rhythm games transfer to each other unusually well"*, not *"cross-application
transfer is ordered by task structure"*. The second is the stronger, more quotable sentence and it is
the one the registration failed to establish.

## F3 — §6.4 states a causal reading whose licensing control it does not cite

§6.4: *"A related registered prediction also held: the unseen-application cell sits below the
seen-application cell, −0.036 [−0.054, −0.018]. **Training on all five applications was buying
something**, which is worth establishing rather than assuming."*

RESULTS registers that very row as **`P3 − C2-hi < 0 (direction, dose-confounded)`** and its verdict
reads *"holds; **two sufficient explanations** (the held-out application, and 20% less Across-XR
data)"*. On the draft's presentation the −0.036 is equally explained by the P3 arm simply having less
in-domain data.

**The claim is defensible and the draft omits its defence.** One row further down RESULTS carries the
control that defuses the dose explanation: `control P3(non-X) − C2-hi(non-X) within ±0.03` measured
**−0.009 [−0.025, +0.008]**, verdict *"20% less in-domain data at fixed exposure cost nothing measurable
on the seen cells, so the runs are comparable to C2-hi"*. That is what licenses the causal reading, and
it is not cited in §6.4. Severity: low — a one-clause fix, not a retraction. But as written a reviewer
meets a dose-confounded contrast presenting as a clean one.

## F4 — the field recommendation rests on a single seed (lowest severity, and partly disclosed)

§6.7's mechanism paragraph: *"the ceiling variant fits on the seventeen people it is then scored on and
reaches **+0.148**, while the honest variant fits on thirty-two other people and reaches **−0.008**"* —
and from that mechanism follows the paper's recommendation to the field (a corpus needs far more
multi-application participants).

**The pairing itself is sound and I verified it**: both values are C2-lo **seed 1**, same arm, same run
(`across_xr_alignment_RESULTS.md:223` and `:258`), so this is not the cross-arm comparison it might look
like. But +0.148 is **one seed of three at identical configuration** — the other two read −0.004 and
+0.001 — and RESULTS says of that cell: *"present in one of three runs — **reported as that, never as a
rate**"*.

The draft does disclose the run-dependence two paragraphs earlier, in the same section, which is why
this is F4 and not F1. The gap is that the mechanism sentence and the recommendation built on it quote
the single seed where the structure appeared, without the "one of three" qualifier travelling to the
point where the claim is made. This project's own record contains the matching lesson (an elegant
account of n=1 is not evidence about n=1).

## Verified earned — reported as prominently, because a clean negative is the result

| draft claim | RESULTS | verdict |
| --- | --- | --- |
| §6.1 P1 "held, inside the band" (0.234, band 0.18-0.35, falsifier < 0.12) | inside the band | **earned** |
| §6.1 zero-shot vs their model "unresolved", both metrics | UNRESOLVED, both | **earned** |
| §6.2 C2-lo BEAT, both directions (+0.119, +0.176) | BEAT, both | **earned** |
| §6.2 "the second landed 0.006 past its upper edge, which we declare rather than argue" | "an edge, not argued" | **earned** |
| §6.4 P3 falsifier excluded; stricter threshold (+0.030) **NOT met** at +0.021 | headline NOT made | **earned** |
| §6.7 "the registered band of +0.05 to +0.20 is excluded" | band excluded | **earned** |
| §6.7 `raw` seed +0.032 flagged as scoping the negative to `dyn` | same scope in RESULTS | **earned** |
| §6.8 mechanism failure, margin reversal, activity-diversity null | all three | **earned** |
| §6.9 "not registered in advance"; coverage corollary **withdrawn** | matches | **earned** |
| §5.5 "twenty-three gates passed, gaps 5.3e-8 to 2.9e-4" | **recomputed from the 23 certificates: 23 entries, 23 distinct checkpoints, all `passed: true`, min 5.280e-08, max 2.885e-04** | **earned, digit-exact** |
| §6.5 identity count −0.013 [−0.039, +0.013] | RESULTS claim 5: Z-676 − zero-shot −0.013 | **earned** |
| §5.5 Table 3 population note | names OUTLINE/RESULTS and spells out P2's trap | **G1 already fixed** |

## Non-findings, recorded so nobody re-checks them

- **The draft's §6.4 table disagrees with RESULTS' per-application table (lines 610-620) on Synth Riders
  (+0.065 vs +0.077), Social VR (+0.038 vs +0.044) and the pooled figure (+0.049 vs +0.053).** The draft
  is **more current, not stale**: it uses the two-seed values from the re-seeds registered in Amendment
  4's third addendum, and that RESULTS table is the seed-1 one it supersedes. RESULTS says so elsewhere
  (*"this read +0.053 on seed 1 until 2026-09-15"*). No action.
- **§6.7's +0.148 and −0.008 are the same arm and the same seed** (C2-lo seed 1), not a cross-arm
  comparison. See F4 for what is actually at issue.
- **§5.2's "they agree on every verdict"** holds for the headline contrasts: ZS UNRESOLVED under both
  metrics, C2-lo BEAT under both, and both ten-minute directions agree. The one registered item that
  went the other way (the D2 *level* band, 0.20-0.32, measured 0.199) is a level rather than a contrast
  and is reported in §6.8 as the failed mechanism.

## One suggestion, since Table 3 is still a placeholder

F1 is exactly what Table 3 exists to prevent: a registered prediction with no row. When it is populated
from OUTLINE/RESULTS, **the rhythm/Social-VR pair should appear as two rows, not one**, with the second
marked FAILS. A table of registered predictions that silently omits the ones that failed is worse than
no table, because it looks like the complete set.

## Scope and limits

- §5 and §6 only, as commissioned, plus the §7 passage that F2 depends on. §1-§4 and §8-§9 not audited.
- Verdicts are compared against RESULTS' own verdict columns; I re-derived only the gate range, which is
  computable from files on this node. Every other figure is taken from the certificates, not recomputed.
- This node holds no checkpoints and no corpora, so nothing here rests on a measurement I made.
