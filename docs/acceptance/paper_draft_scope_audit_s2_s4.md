# Audit: §2 related work, and a spot-check of §3-§4

**Run 2026-09-17 on LAPTOP-C (`xrsec-f9`), repo at `895af60`. READ-ONLY — `docs/PAPER_DRAFT.md` was not
edited.** §2 in full on both versions (`main` and `origin/review/paper-draft-2026-09-17` — §2 is identical
on both), plus a targeted §3-§4 pass and a re-verification of §1's bullets after `895af60` changed §1.

Two additions to the hunt were requested and both were applied: check each related-work
characterisation against the citations in the same paragraph, and flag anywhere prior work is credited or
discredited **in our voice** on the basis of our reading of their released artefacts rather than their
published text.

## Headline

**The predicted finding did not materialise, and that is worth stating first: 83.1% is NOT restated in
§2.** The prediction was a fourth unqualified restatement; §2 never quotes the figure at all. Nothing to
fix on that axis.

**Three findings, and all three are about provenance rather than about numbers.** §2 asserts one
measurement the paper never reports, describes one cited work's protocol in a way two of our own internal
records disagree about, and makes two claims about another group's artefacts with no provenance — in a
paper whose §5.2 documents its own provenance to the digit. §2 is the least evidenced section of the
draft, and it is the one making claims about other people's work.

**§3-§4 are clean on these axes**, and the H3 hedge is now consistent at all three sites.

## J1 — "we measure the ordering to invert in our setting" reports a measurement the paper does not contain

§2, *Encodings*:

> "Their ordering was obtained with a full body frame derived from the head *and* both controllers. A
> head-only rig cannot construct that frame, and **we measure the ordering to invert in our setting: on
> pooled head-only corpora the raw encoding beats the best body-relative alternative by a wide margin**…"

**The paper reports no such experiment.** Grepping the draft for `BRA`, `BRV`, `body-relative` and the
claim's own words returns the §2 sentence, §1's description of *their* encoding, §4.2's `dyn`-vs-`raw`
contrast, and §8's mention of their encoding — and **no results anywhere for the SR/BR/BRV/BRA family**.
There is no number, no interval, no section reference and no self-citation attached to "we measure".

Three things compound it, from `CLAUDE.md`'s record of where that result came from:

| | the measurement as made | this paper |
| --- | --- | --- |
| corpora | 8 pooled seated corpora (PanoSaliency, ViewGauss, VR_User_Behavior, …) | BOXRR-23, who-is-alyx, Across-XR, Questset |
| identities | 419 | 3,072-4,096 |
| metric | verification AUC (0.7284 raw against 0.5970 bra) | rank-1 identification |

So *"in our setting"* names a setting that is not this paper's: different corpora, different scale, and a
different metric. `docs/PAPER_OUTLINE.md:112` shows the intent was modest — *"one sentence, to motivate
why `dyn` rather than BRV"* — but the sentence as written claims data, and a reviewer who asks for it will
find the paper has none.

**The repair is already in the same sentence.** *"A head-only rig cannot construct that frame"* is a
design argument that needs no measurement and carries the paragraph on its own; the clause after it is
what creates the exposure. Either drop the empirical claim, or cite it with its corpora, metric and scale
as prior unpublished work and accept that it is then a claim about a different setting.

## J2 — the Rack et al. 2023 characterisation discredits a cited protocol, and two of our own records disagree

§2: *"Rack et al. (2023) contributed both a dataset and a protocol: 'Who is Alyx?' records seventy-six
players … and reports cross-session identification of users **seen** during training in the high
seventies."*

`docs/LITERATURE_BRIEFING.md:207` agrees: *"A: 71 users, single 20 s window, 76.6-78.3% rank-1 (**seen
users**, cross-session)."*

`docs/acceptance/sota_rack2023_reproduction_REGISTERED.md`, written from their config and code, does not:

- 63 users used (76 − 8 dropped − 5 with ≠ 2 sessions, "arithmetic closes")
- split **train 27 / validation 9 / test 27** — three disjoint groups
- window 500 frames at 15 FPS = **33.3 s**
- *"The values below are the authors' own stated figures in their prose, **for the 27 test subjects**."*

**The two records describe different quantities** — 71 users against 63, a 20 s window against 33.3 s — so
the most likely resolution is that the paper contains both a seen-user dataset benchmark and a
disjoint-split experiment, and §2 is describing the first. **I cannot resolve it here: this node holds no
PDFs** (`find -iname '*.pdf'` returns 0).

Why it still needs one check before shipping. As written, §2 attributes to that paper *only* a seen-user
protocol, which **discredits it in our voice** — and it is the specific criticism this project levels at
its own lineage, so a reader will take it as a pointed one. If the same paper also reports figures on 27
held-out test subjects, then the sentence is true of one experiment and misleading about the work, and it
matters twice over because this project is **currently reproducing that disjoint-split experiment**. One
read of the Frontiers text settles it; it is not settleable from the draft.

**The contrast inside §2 is the sharpest form of this.** The *Cross-application* paragraph gives Baldoni
et al. (2025b) a careful three-clause qualification — test users seen during training, the exact
train/test split within each recording, head plus both controllers, and an explicit *"not a competitor to
ours and is not treated as one"*. That paragraph is the model. The Rack sentence gives a cited work's
protocol one word.

## J3 — two claims about another group's artefacts, with no provenance, in a paper that is scrupulous about provenance

§2, *Embedding alignment*, makes two claims in our voice that rest on material outside their paper's text:

1. *"Their released evaluation code contains an `orthogonal_procrustes`-based multi-domain alignment,
   **which corroborates the method independently of the text**."* — a **credit** to prior work resting on
   our reading of their repository. `docs/PAPER_OUTLINE.md:119` names the file
   (`evaluation/helpers/compute_transformation_matrix.py`); **the draft omits it.** No acceptance
   certificate in `docs/acceptance/` records this verification, and it is not checkable on this node
   (`external_sota/` is absent here, being gitignored).
2. *"Note that this material appears only in the journal version of their paper; the preprint does not
   contain it."* — a claim about another group's **publication history**, carried in
   `PAPER_OUTLINE.md:119` as **"[DISCREPANCY, gap G1]"**, with no version identifiers, dates or arXiv
   version number in the draft.

**The finding is the asymmetry, not either claim.** §5.2 documents its own provenance to the digit —
"every per-user value in all thirty-five cells at a maximum absolute difference of 0.0", the index
correspondence "reconstructed rather than assumed", 463,996 embeddings accounted for. §2 then makes two
claims about the same group's artefacts with no file path, no commit, no version. Same paper, same group,
two standards. Both are probably correct; neither is checkable as written, and the second is flagged
internally as an unresolved discrepancy.

## J4 — the predicted finding, reported as not found

The hunt expected 83.1% restated in §2 without the self-match qualification. **It is not there.** §2
quotes 18.0% → 52.3% and 30.8% → 94.3% (their alignment result, correctly attributed and correctly
described as disqualified by them), Rack's "high seventies", and Baldoni's ">95% within-game". The only
within-application figure in §2 is Baldoni's, and it is thoroughly qualified. §2 is clean on this axis and
the prediction fails.

## §3-§4 spot-check — clean, and one pattern now consistent

| claim | status |
| --- | --- |
| §3.2 "the largest XR motion corpus in existence **cannot support** a cross-application study" | **earned, and measured rather than asserted** — CLAUDE.md records the exhaustive key-set partition of all 4,716,986 index documents: 92,103 users with Beat-Saber-labelled recordings, 13,746 with unlabelled, **overlap zero**, and 105,849 against the release's own 105,852 users, so the finding is about BOXRR-23 and not about our copy. The Tilt Brush tarball inspection (no HMD device) closes the remaining branch |
| §3.2 "the only corpus in **our pretraining set** with cross-day structure" | correctly scoped |
| §3.3 "Across-XR remains the only fully crossed cross-application corpus **we know of**" | hedged |
| §3.1 "no arm ever trains on them" (users 32-48) | matches §4.5 and the arm table |
| §4.2 `dyn` "Anything that survives it is behaviour" | supported by the encoding's definition; §6.6 audits the `raw` counterpart rather than assuming it |

**H3 is now consistent at all three sites** — §3.3 "we know of", §7 "we are aware of", §8 "we are aware
of" — and §7's alignment paragraph now reads *"Whether a corpus with more would support the method is
**not established**"*. Both earlier findings are fixed throughout rather than at the site where they were
found.

## §1 re-verified after `895af60`

The commit changed the abstract's first sentence (now *"is **reported as** close to solved … and **as**
collapsing"*), added the §5.4 forward reference to §1, and rewrote Contribution 4. **§1's other bullets
are untouched by that diff**, so the clean verdicts in `paper_draft_scope_audit_s9_s1.md` stand for
contributions 1, 2, 3, 5 and 6. Contribution 4 now states the person-specific mechanism and says whether
more participants would help is "not settled here" — J1's sibling defect is gone from that bullet.

## Scope and limits

- §2 in full (identical on both versions), §3-§4 by targeted scan for superlatives, causal connectives and
  first-person empirical claims rather than line by line, §1 re-verified by diff.
- **J2 is unresolved and not resolvable here** — no PDFs on this node. It is a flag for one check against
  the Frontiers text, not a finding that the draft is wrong.
- **J3 is unverifiable here** — `external_sota/` is gitignored and absent. Reported as missing provenance,
  not as a false claim.
- Nothing recomputed from data; this node holds no checkpoints or corpora.
