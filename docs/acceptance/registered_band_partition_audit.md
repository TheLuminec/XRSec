# Audit: do registered bands and their falsifiers partition the line?

**Run 2026-09-17 on LAPTOP-C (`xrsec-f9`), repo at `f09c1b0`, working tree clean.** Commissioned by
the Coordinator after three known violations of the CLAUDE.md rule *"a band and its falsifier must
partition the line"*, two of them occurring **after** the rule was written and read by both parties.
No GPU, no corpus: this is a reading pass over registration text and the harnesses that score it.

## Headline

**28 registrations carry an explicit decision rule. 13 partition the line cleanly. 15 leave at least
one unnamed region. 4 measurements have actually landed in an unnamed region.**

**No conclusion in this project rests on a verdict its registration did not license.** Each of the
four that landed in a gap was reported as something other than a pass or a fail — as a third
outcome, as "unresolved", or corrected within the day. That is the result and it should lead: the
rule is violated often in *drafting* and has not yet produced a wrong *conclusion*.

The one substantive defect found is **not** a bad verdict. It is a registered band that was measured
and then **never scored at all** (R14, within-application height) — silent rather than wrong.

## Gate - required before any clean cell in this table means anything

The Coordinator's condition: this pass must rediscover all three known violations independently.
**It found all three, plus one they did not name.**

| known violation | rediscovered? | where |
| --- | --- | --- |
| Within-application placement: band 0.80-0.95, falsifier < 0.65, measured **0.7525** | **YES** | R13 |
| Alignment kill condition: band >= +0.15, falsifier CI upper < +0.05, measured CI upper **+0.053** | **YES** | R2 |
| Questset static lookup: group 1 band had no outcome named below it, measured **0.092** | **YES** | R16 |
| *(not named by the Coordinator)* within-application **height** band never scored | **found** | R14 |

Stated plainly so the scope is auditable: **candidate discovery is mechanical; every partition
verdict in this table was read by hand**, because whether two regions meet is a reading question and
not a regex question. A registration phrased without any of the handle words is invisible to this
pass and I cannot bound how many of those exist.

## Population, counted independently of the rows displayed

`falsifier` occurs **136** times repo-wide (`grep -roi`, excluding `.git` and `.venv`), summed per
file: CLAUDE.md 22, across_xr_alignment_RESULTS.md 15, across_xr_alignment_REGISTERED.md 14,
PROGRESS_REPORT.md 13, COORDINATION_ARCHIVE.md 11, COORDINATION.md 9, PAPER_PLAN.md 7,
PAPER_DRAFT.md 6, questset_static_lookup.py 5, GENERALISATION_PROPOSAL.md 5,
nymeria_activity_analysis.py 4, PAPER_OUTLINE.md 4, questset_static_lookup.json 3,
questset_geometry.py 3, questset_geometry.json 3, questset_arms_REGISTERED_miami.md 3,
across_xr_within_application.py 2, across_xr_alignment_p2.py 2, step6_offset_trend.py 1,
step6_implied_rank1.py 1, sota_rack2023_reproduction_REGISTERED.md 1,
across_xr_within_application.json 1, across_xr_alignment_p3.py 1. **Sums to 136 exactly.**

**The scope given to me yields 106, not 136.** The commissioning message listed
`docs/acceptance/*`, PAPER_PLAN, PAPER_OUTLINE, GENERALISATION_PROPOSAL, COORDINATION and CLAUDE.md.
That omits **PROGRESS_REPORT.md (13), COORDINATION_ARCHIVE.md (11) and PAPER_DRAFT.md (6)** - 30
occurrences. **PAPER_DRAFT.md is the file the stated worry was about** ("possibly in the paper
draft"), so the scope as given excluded the document the audit was most for. Audited at the full 136.

Most occurrences narrate a registration rather than constituting one; deduplicated to **28 distinct
decision rules**.

## The table

`gap` is the width of the unnamed region between a band and its falsifier on the same axis.
"landed" is the column that matters.

| # | source | band as written | falsifier as written | partitions? | gap | resolved | landed | verdict earned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | REGISTERED.md:181 (P1/A1) | 0.18-0.35 | < 0.12 | **no** (both ends) | 0.06 below, open above | yes | 0.234, in band | **yes** |
| R2 | REGISTERED.md:181 (A2'-A1) | >= +0.15 | CI upper < +0.05 | **no** | **0.10** | yes | +0.026 [+0.000, **+0.051**] - upper end in gap | **yes** - band exclusion does not depend on the gap; recorded in Amendment 2 |
| R3 | REGISTERED.md:184 (A2-A1) | +0.05..+0.20 | < +0.05 | **no** (above band) | 0 below, open above | yes | +0.011 [-0.020, +0.041] | **yes** |
| R4 | REGISTERED.md:190 (A2-null) | <= A1 + 0.03 | binary, same line | **yes** | - | yes | null hurts | **yes** |
| R5 | REGISTERED.md:192 (A2-full) | <= A2 in expectation | interval-based | **yes** | - | yes | below A2 | **yes** |
| R6 | REGISTERED.md:195 (seed variance) | within 0.05 | binary | **yes** | - | yes | range 0.005-0.010 | **yes** |
| R7 | REGISTERED.md:287 (C1) | 0.10-0.25 | < chance + 0.03 = 0.089 | **no** | 0.011 below, open above | yes | 0.131 [**0.088**, 0.177], lower edge on the falsifier | **yes**, and RESULTS says exactly that |
| R8 | REGISTERED.md:293 (C2-A1) | +0.05..+0.20 | < +0.03 | **no** | 0.02 below, open above | yes | +0.089 [+0.048, +0.131] | **yes** |
| R9 | REGISTERED.md:373 (Amd 4, P3) | +0.02..+0.07, headline > +0.03 | <= 0, **(0, +0.02] named "unresolved"** | **YES - exemplary** | none | yes | +0.053 [+0.022, +0.083] | **yes**; headline explicitly NOT made |
| R10 | REGISTERED.md:520 (Amd 6, P2) | +0.00..+0.06 | < -0.03 | **no** (both ends) | 0.03 below, **open above** | yes | **+0.117 [+0.042, +0.192] - above the band, in an unnamed region** | **yes** where scored, see G1 |
| R11 | REGISTERED.md:527 (Amd 6, A0) | A0 gain > A1 gain | binary, either way | **yes** | - | yes | +0.223 > +0.117 | **yes** |
| R12 | REGISTERED.md:584 (Amd 7 screen) | sign flip named in advance | screen, not test | **yes** | - | yes | -0.028 [-0.045, -0.012] | **yes**; size against the -0.02 edge reported unresolved |
| R13 | within_application.py:271 (lateral) | 0.80-0.95 | < 0.65 | **NO** | **0.15** | yes | **0.7525 [0.7114, 0.7748] - WHOLLY in the gap** | **yes** - reported as a third outcome, not as pass or fail |
| R14 | within_application.py:271 (height) | 0.85-0.97 | **none registered** | **NO** - nothing named below 0.85 | open | measured | 0.8894 [**0.8483**, 0.9007] | **NOT SCORED - see G2** |
| R15 | questset_static_lookup.py:25 (group 2) | < 0.10 holds / 0.10-0.15 weakened / > 0.15 falsifier | as written | **YES - exemplary** | none | yes | 0.033 | **yes** |
| R16 | questset_static_lookup.py:19 (group 1) | "ABOVE 0.15" at N=30 | **none for below** | **NO** | open below | yes | **0.092 - in the unnamed region** | **corrected**: reported "held" 2026-09-16, fixed the same day to "NOT MET" |
| R17 | questset_geometry.py:13 (height) | > 0.65 | < 0.60, **0.60-0.65 named "weakened"** | **YES - exemplary** | none | yes | g1 0.718 held, g2 0.493 fired | **yes** |
| R18 | questset_geometry.py:13 (lateral) | 0.45-0.60 | none | **no** | open both ends | yes | 0.526 / 0.549, in band | **yes** |
| R19 | questset_arms_miami.md:17 (A1) | < 0.10 / 0.10-0.15 / **0.15-0.40** / > 0.40 | all four named | **YES - exhaustive** | none | **not yet run** | - | - |
| R20 | questset_arms_miami.md:31 (A2) | group 2 raw-dyn smaller | equal or larger | **yes** | - | **not yet run** | - | - |
| R21 | nymeria_activity_analysis.py:34 | +0.005..+0.03 | < +0.005 | **no** (above band) | 0 below, open above | yes | -0.0012 [-0.0045, +0.0020] | **yes** |
| R22 | step6_offset_trend.py:37 | <= +0.02 under all three maps | > +0.02 under all three; **split = "not resolved"** | **YES - exhaustive** | none | yes | +0.001 / +0.016 / +0.010 | **yes** (mechanism wrong, band right - CLAUDE.md records it) |
| R23 | step6_implied_rank1.py:19 | >= +0.06 | < +0.02 | **no** | **0.04** | yes | +0.109 / +0.116 | **yes** |
| R24 | GEN_PROPOSAL 9.11 (Nymeria) | 0.52-0.56 | < 0.51 and > 0.60 | **no** | **0.04** above | yes | in band | **yes** |
| R25 | GEN_PROPOSAL 9.11 (identity count) | +0.00..+0.02 | none | **no** | open | yes | +0.013 | **yes** |
| R26 | GEN_PROPOSAL 9.14 (additivity) | 0.618 point | < 0.606 and > 0.635 | **yes** (middle is the band) | none | yes | 0.6184 / 0.6176 | **yes** |
| R27 | GEN_PROPOSAL 9.14 (seed 2) | within +-0.006 of seed 1 | more than 0.01 from seed 1 | **no** | **0.004** | yes | +0.0007 / -0.0056 | **yes** |
| R28 | sota_rack2023_REGISTERED.md:113 | all cells in seed spread AND ordering holds | any cell outside OR ordering breaks | **yes** | - | **open** (seed 1 only) | - | - |

**Clean partitions (13):** R4, R5, R6, R9, R11, R12, R15, R17, R19, R20, R22, R26, R28.
**Landed in an unnamed region (4):** R2, R10, R13, R16.

## The two findings worth acting on

### G1 - PAPER_PLAN scores P2 against a registration that Amendment 6 superseded

`docs/PAPER_PLAN.md:182` records P2 as **"(no band) ... HELD"**. Amendment 6 registered an actual
magnitude band for that quantity: **+0.00 to +0.06, falsifier below -0.03**. The measurement is
**+0.117 [+0.042, +0.192]** - above the band, in a region Amendment 6 does not name.

The scoring harness is honest: `across_xr_alignment_p2.json` records
`"verdict_A1": "interval spans a registered edge: unresolved"`, and **`docs/PAPER_OUTLINE.md:269`
states it correctly** - *"falsifier excluded, size unresolved against the band edge"*. So the
paper-facing chain is right and **PAPER_PLAN's row is the stale outlier**.

Why it still matters: `docs/PAPER_DRAFT.md:450` holds the placeholder
`[TABLE 3: registered predictions and verdicts - prediction, registered band and falsifier, measured
value with interval, verdict]`. If that table is populated from PAPER_PLAN's three-row summary, P2
enters the paper as *"no band, held"* when a band existed and the result sits outside it. **Populate
Table 3 from PAPER_OUTLINE and RESULTS, not from PAPER_PLAN.** Severity: low today, high the moment
Table 3 is filled from the wrong source.

### G2 - a registered band that was measured and never scored

`across_xr_within_application.py:271` registers **two** bands - lateral `[0.80, 0.95]` and height
`[0.85, 0.97]` - with a falsifier for lateral only. The verdict block at lines 302-306 prints
`band ... CONTAINED?` and `falsifier ... FIRED?` **for lateral alone**. Height is computed
(P = 0.8894, CI [0.8483, 0.9007]), written to JSON, and never compared to its band anywhere in the
repo. No document claims a verdict on it.

Two consequences. **By the point estimate it would read "held"** - 0.8894 sits inside [0.85, 0.97].
**By this project's own rule, read against the interval, it would not**, because the CI's lower edge
0.8483 falls below the band's lower edge. So the unscored band is one whose two scoring conventions
disagree, which is where silence is least safe.

This is the same defect as the Questset group-1 miss (R16), in the same file family: a harness that
scores one registered line and prints a verdict which reads as the verdict for the registration. The
Questset harness was repaired on 2026-09-16 with a comment explaining why;
`across_xr_within_application.py` was not, because nobody looked at it again. **A fix applied where a
defect was found is not a fix applied where the defect is.**

## What did not turn up, recorded because a clean negative is the result

**No resolved registration reports a pass or a fail for a region its registration never named.** The
four gap landings were each reported as something other than a verdict:

- **R13** (0.7525): CLAUDE.md says *"The band was excluded and the falsifier did not fire: two
  outcomes were registered and the data chose a third."* Named as a registration defect, not scored.
- **R2** (+0.051 upper): Amendment 2 exists solely to record the defect, and argues correctly that
  excluding a band at >= +0.15 does not depend on where the gap sits.
- **R16** (0.092): mis-scored as "held" and **corrected the same day**, with the mechanism recorded
  - a one-line verdict computed over half a registration.
- **R10** (+0.117): scored "unresolved" by the harness and by PAPER_OUTLINE; only PAPER_PLAN's stale
  row disagrees (G1).

## Recommendations

1. **Populate PAPER_DRAFT Table 3 from PAPER_OUTLINE and RESULTS.** Correct `PAPER_PLAN.md:182` to
   state Amendment 6's band and the "unresolved against the band edge" verdict.
2. **Score the height band in `across_xr_within_application.py`**, or strike it from the `registered`
   dict. A band in the record that no code reads is a claim nobody will check.
3. **Make the partition check mechanical rather than attentional.** Every clean case here (R9, R15,
   R17, R19, R22) shares one property: the registration **names the middle region in prose**. Every
   defective one names two endpoints and lets the middle fall out. A registration template with a
   mandatory third line - *"landing between X and Y means: ___"* - turns this from something to
   remember into something missing from a form.
4. **Where a harness scores a multi-line registration, score every line.** R14 and R16 are one bug in
   two files and only one was fixed.

## Scope and limits

- Mechanical candidate discovery, **manual adjudication**. A registration using none of the handle
  words is invisible to this pass.
- `COORDINATION_ARCHIVE.md` (11 occurrences) and `PROGRESS_REPORT.md` (13) were read for *new*
  decision rules and contained none absent from the table; they narrate registrations scored
  elsewhere.
- Resolution status is taken from the certificates and results documents in the repo.
  **No number in this audit was recomputed from data** - this node holds neither the checkpoints nor
  the corpora, and says so rather than implying verification it did not perform.
