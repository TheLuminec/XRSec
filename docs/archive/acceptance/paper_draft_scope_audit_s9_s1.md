# Audit: §9 and §1 for the H1 shape — the claim someone would prefer to be true

**Run 2026-09-17 on LAPTOP-C (`xrsec-f9`), repo at `bbf6db1`. READ-ONLY — `docs/PAPER_DRAFT.md` was not
edited.** Both versions audited: `main` and `origin/review/paper-draft-2026-09-17`. Target was the **H1
shape** — a design target rather than a null, a mechanism rather than an observation — on the reasoning
that a conclusion and an introduction are where attractive phrasings collect.

## Headline

**H1 was not at three sites. It is at five, and two are still live on `main`** — §9's second paragraph and
**§1's Contribution 4**, the latter in the contributions list and labelled *"with a mechanism"*, the
mechanism being the one RESULTS ruled out. The review branch has both correct.

**And the reason the fix missed them is worth more than the two sites.** All four of the corrected
phrasings now return **zero hits** in the draft, so a completeness check run with the fix's own search
terms comes back clean while the claim is live twice more in different words. The family to sweep is the
**claim**, not the string — and a conclusion and a contributions list are precisely where a claim is
restated in fresh words.

**One further finding sits in the abstract on both versions** and is the paper's own framing sentence
contradicting §5.4.

## I1 — H1 at a fourth site: §9 paragraph 2 (`main` only)

`main` §9:955-958:

> "…does not carry on a static-free encoding, and **the reason is a property of the available data rather
> than of the method**: **thirty-two people recorded in two or more applications is not enough** to fit a
> transformation that generalises to new ones. **That is the most actionable output of this work.** The
> field's cross-application results are limited by corpora, and **the corpus that would move them** is one
> in which many people are recorded in many applications."

Against `across_xr_alignment_RESULTS.md:116` — *"so **more correspondences did not help**"* — every
emphasised clause is the refuted mechanism: the failure is attributed to the **count**, the count claim is
promoted to *"the most actionable output"*, and the design target is asserted as one that *would* move the
field.

**The review branch is correct and its wording is the supportable one:**

> "The map it fits is **person-specific**: it helps, when it helps at all, only the people it was fitted
> on… **Whether** a far larger fitting population would recover a transformation that does transfer **is
> untested here**, and it is the most actionable **question** this work leaves: the corpus that **could
> answer it**…"

**Merge instruction: §9 takes the branch wholesale.** Unlike §7, which is mixed, there is nothing in
`main`'s §9 paragraph 2 worth preserving.

## I2 — H1 at a fifth site: §1 Contribution 4 (`main` only)

`main` §1, contribution 4:

> "**A negative answer, with a mechanism**, to the reference study's own proposed fix. Train-user-only
> orthogonal alignment … does not carry on a static-free encoding, and **the reason is a property of the
> corpus rather than of the method: at most thirty-two people are recorded in two or more applications**,
> and no amount of pretraining raises that number (Section 6.7)."

This is the most load-bearing position in the paper for this claim — a numbered contribution, promising *a
mechanism*, where the mechanism given is the one G17 excluded. The final clause ("no amount of pretraining
raises that number") is true and is deployed in support of the count-based reading.

**The branch is correct**: *"The fitted map is person-specific: it helps, when it helps at all, only the
people it was fitted on, and the corpus offers at most thirty-two participants outside the evaluation
split (users 0–31) to fit on, a number no amount of pretraining raises."* Same facts, no causal claim.

## I3 — the family sweep matched the phrasing, not the claim (methodological; the durable half)

Measured rather than asserted. Grepping `docs/PAPER_DRAFT.md` at `bbf6db1` for the phrasings the fix
corrected:

| search term | hits in the draft |
| --- | --- |
| `would need far more` | **0** |
| `concrete design target` | **0** |
| `actionable specification` | **0** |
| `far more multi-application` | **0** |

All four exhausted — and the claim is live at lines 955-958 and in contribution 4, worded *"is not
enough"*, *"the most actionable output"*, *"the corpus that would move them"*, *"a property of the corpus
rather than of the method"*. **A completeness check using the terms you just fixed returns a clean
result while the claim survives.**

**A second mechanism compounds it: the draft is hard-wrapped at ~100 characters, so claims straddle line
breaks and defeat single-line greps even when the phrase is right.** `grep "property of the available
data rather than of the method"` returns nothing on a file that contains exactly that sentence, because it
breaks after *"a property of"*. Confirming I1 required a multiline pattern. Any sweep of this document
for a claim needs `-U`/multiline or a whitespace-tolerant pattern, or it under-reports silently — the
failure direction that reads as "already fixed".

This extends the rule adopted this morning. *"When a defect is found in one harness, grep the family
before closing it"* is right, and the family for a **prose claim** is not a string set: conclusions,
abstracts and contribution lists exist to restate, so they hold the same claim in deliberately different
words. The check that would have worked here is a sweep for the **subject** — every mention of
*thirty-two*, of *correspondences*, of *alignment* — rather than for the sentence that was edited.

## I4 — the paper's framing sentence asserts what §5.4 shows is overstated (BOTH versions; in the abstract)

`main` abstract, **first sentence**:

> "Motion-based identification in extended reality **is close to solved within a single application** and
> **collapses across applications**."

The branch's abstract (line 30) is the same claim: *"is near-solved within an application and
collapses"*. The framing recurs at §1:72 (*"That collapse from near-solved to roughly three times chance
is the gap this paper addresses"*) and at §9:948 on both versions.

§5.4:432, the paper's own structural finding:

> "…the drop from 83.1 to 18.0 **overstates the cross-application collapse** by whatever the overlap is
> worth."

So the paper establishes that the collapse it opens by asserting is overstated, because the
within-application figure it rests on contains gallery–probe frame overlap on every query window. This is
the H1 shape — **the larger collapse is the more motivating one** — and simultaneously the F2 shape, a
summary contradicting its own detail, at the highest-exposure sentence in the document.

**Scoped fairly, because two nearby things are fine.** Quoting the reference study's 83.1% *with
attribution* is legitimate, and the abstract's second sentence does attribute it correctly ("The
reference study … reports 83.1%"). §1's account of why their figures are *not* inflated by placement or
anthropometry is generous and correct. The defect is narrow: sentence 1 of the abstract, and §1's *"That
collapse … is the gap this paper addresses"*, state the collapse in the paper's own voice with no
qualifier, 300 lines before the paper qualifies it.

Supportable version: *"reported as close to solved within a single application"*, or *"near-solved by
within-application evaluations that cannot avoid gallery–probe overlap (§5.4)"* — which costs a clause
and makes §5.4 a promise the introduction keeps rather than a correction it walks into.

## Verified clean

§9 paragraph 1 and the rest of §1 are earned:

| claim | status |
| --- | --- |
| §9 "identifies unseen users … better than a system using the head and both hand controllers", +0.119 / +0.355 | earned; correctly qualified as the **exposed** arm |
| §9 "Without any exposure the comparison is unresolved, and we report it as unresolved" | exemplary |
| §9 "a behaviour-only assessment understates the total risk — extends … rather than contesting it" | earned |
| §9 "conceals both the most exposed individual and the least" | earned from the per-user range |
| §1 contribution 1 "a head-only beat" | earned — scoped to the exposed arm, with zero-shot declared unresolved in the same bullet |
| §1 contribution 3 "The stricter registered threshold … was not met, and we say so" | earned |
| §1 "Every published cross-application result **we are aware of** … uses the head together with both hand controllers" | correctly hedged, and consistent with Baldoni et al. |
| §1 "eighteen per-frame features are four head-rotation features and fourteen from the two controllers" | matches the reference study |
| §1 "Their figures are consequently *not* inflated by placement or anthropometry" | correct, and volunteered in their favour |
| §1's two properties of the reference design | match §5.4 and §6.4 |

## Scope and limits

- §9 and §1 in full, on both versions. The abstract was read because I4 originates there.
- **§2, §3 and §4 remain unaudited.** §2 (related work) is where I would go next for this shape — it is
  where a field's state gets characterised, and I4 shows the characterisation is already carrying more
  than §5.4 supports.
- §5-§6 covered by `paper_draft_verdict_audit.md`; §7-§8 by `paper_draft_scope_audit_s7_s8.md`.
- Nothing recomputed from data. I1, I2 and I4 rest on comparing the draft's text against RESULTS and
  against §5.4; I3 is a measured grep result.
