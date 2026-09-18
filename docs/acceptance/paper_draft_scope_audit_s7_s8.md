# Audit: is any sentence in PAPER_DRAFT §7-§8 wider than the result under it?

**Run 2026-09-17 on LAPTOP-C (`xrsec-f9`), repo at `bec1309`. READ-ONLY — `docs/PAPER_DRAFT.md` was not
edited.** Commissioned to hunt the **F2 shape** (a sentence wider than its result) rather than the F1
shape (a wrong verdict), with mechanisms and field recommendations as the priority targets.

**Both versions audited this time.** Last pass measured `main` and F1 turned out to be already fixed on
`review/paper-draft-2026-09-17`. So every finding below records which version it applies to. The two
differ by 291 insertions / 195 deletions across the draft.

## Headline

**§8 is the best-scoped section in the paper and needs nothing.** Every limitation is named, quantified
where quantifiable, and several are volunteered against interest (the headline arm choice being post hoc,
the single published baseline, the posture result being single-corpus).

**§7 is where the overclaims are, and the most useful finding is that §7 contradicts §8 twice.** In both
cases §8 has the careful phrasing and §7 the wide one, so two of the four findings below can be fixed by
copying a sentence that is already in the document.

**One finding runs the opposite way to F1: `main` is WORSE than the review branch**, and its §7
recommendation to the field is contradicted by RESULTS' own mechanism sentence. That makes §7's merge
resolution **mixed** — main's rhythm fix is better, the branch's alignment paragraph is better — so
neither side can be taken wholesale.

## H1 — main's alignment recommendation contradicts RESULTS' own mechanism (highest severity; `main` only)

`main` §7, *Alignment as a corpus specification*:

> "A corpus built to support this method **would need far more**; that is a **concrete design target** for
> a future collection, and it is available now because the route was run honestly rather than assumed to
> fail."

RESULTS says the opposite about why the honest fit failed (`across_xr_alignment_RESULTS.md:115-116`):

> "…not because the fit is rank-deficient … but the **test-fitted advantage**: A2′ fits on the 17 people it
> is scored on and reaches +0.148; A2 fits on 32 *other* people and reaches −0.008, **so more
> correspondences did not help**."

That is the G17 conclusion the project committed at `37a5083` — *the honest fit fails for the test-fitted
advantage, not the rank argument*. If the failure is that the map does not transfer **across people**,
then more multi-application participants are not established to fix it, and "would need far more" is a
recommendation to the field resting on the mechanism that was explicitly ruled out.

**The review branch already fixes this**, and its wording is the supportable one:

> "**Whether more would help is not established** — the map fitted on those thirty-two did not transfer to
> seventeen others, which a larger fitting population **might or might not** cure — but the subsampling
> test of Section 6.7 decides it on embeddings that already exist…"

**Consequence for the merge.** §7 needs **mixed** resolution: keep `main`'s narrowed rhythm-pair sentence
(the F2 fix) and take the **branch's** alignment paragraph. Taking either side of §7 wholesale
reintroduces one of the two defects. This is the mirror image of the §6.4 situation already recorded — the
branch is better there, main is better on the rhythm sentence, and now the branch is better again on
alignment.

## H2 — a device-class conclusion with no device in the corpus (both versions)

§7, *The second is that head-only suffices*:

> "**AR glasses without hand tracking are therefore inside the scope of this risk, not outside it**, which
> is not how the device class is usually discussed."

**There is no AR-glasses data anywhere in this paper.** Verified by enumerating the corpora rather than
by impression:

| role | corpora | device class |
| --- | --- | --- |
| pretraining (§3.2) | BOXRR-23 (4,020 users, Beat Saber), who-is-alyx (76 players) | VR |
| evaluation (§3.1) | Across-XR (49 users, five applications) | VR |
| second corpus (§3.3) | Questset (60 participants) | VR |

On `main`, Nymeria — the project's only AR-glasses corpus — appears **solely in the reference list**
(entry 14) and in no experiment. The head-only *scope* is motivated by AR glasses in §1 and §2, which is
entirely legitimate; **§7's "therefore" converts a motivation into a conclusion** across a device boundary
the paper never crosses. The inference needs AR-glasses head tracking to carry comparable identity signal
to a VR headset's, which is plausible and untested here.

**Scoped carefully, because the branch contains a tempting over-reading.** The branch's §6.8 describes
Nymeria as *"daily-life activity recorded on AR glasses"* and reports swapping it into training as
**−0.0012 [−0.0045, +0.0020]**, a null with the whole registered band above the interval. That is **not**
evidence that AR-glasses head motion is unidentifiable — it measures whether AR-glasses *training data*
improves cross-domain transfer, a different question. I raise it only because on the branch the paper now
contains one AR-glasses experiment returning a null in §6.8 and an AR-glasses scope conclusion in §7, and
a reader may connect them even though they do not bear on each other.

Supportable version: the risk is available from head tracking alone, so *a device class that tracks only
the head is not excluded by sensor set* — a statement about what the sensor set does not rule out, rather
than a finding about AR glasses.

## H3 — §7 asserts what §8 correctly hedges, and the paper's own bibliography contradicts it (both versions)

§7: "the fully crossed corpus that makes **the whole cross-application literature** possible offers
thirty-two of them."

§8, four paragraphs later: "Across-XR is the only fully crossed cross-application corpus **we are aware
of**."

The §8 hedge is right and §7 has no hedge — but the wider problem is that the claim is contradicted by the
paper's own citations. §2 cites **Baldoni et al. (2025)** reporting roughly 30% cross-application
identification, and **Baldoni et al. (2025b)** on Questset (>95% within-game, nothing above 0.30
cross-game), plus reference 9, *"Cross-application identification across two applications by
classification"*. Those are cross-application identification results obtained on a corpus that is **not**
fully crossed and is not Across-XR. So a cross-application literature demonstrably exists without
Across-XR, by this paper's own related-work section.

Fix is available in the document: §8's phrasing.

## H4 — an empirical claim about deployments with no result under it (both versions)

§7: "Any deployment in which head position reaches an application — **which is most of them, since
position is what the runtime needs** — carries that term…"

The parenthetical is a quantitative claim about deployed XR applications. Nothing in this paper measures
what fraction of deployments expose head position to application code, and no citation is attached. The
reasoning offered ("position is what the runtime needs") is an architectural argument, not a measurement.
The sentence works without the quantifier — "any deployment in which head position reaches an application
carries that term" is exactly as useful and is not a claim about a population nobody counted.

## H5 and H6 — two smaller ones (both versions)

- **"which is not how the device class is usually discussed"** (§7): an unevidenced claim about the
  field's discourse. §2 does not establish it. Either cite it or drop it; it adds nothing the preceding
  clause does not.
- **"The second is that head-only suffices"** (§7): an absolute heading over a properly hedged sentence
  ("*Non-trivial* cross-application identification is available from head tracking alone"). "Suffices" for
  what is unstated, and §8 explicitly says no sensor-set claim is available in either direction. The body
  text is fine; the heading is wider than it.

## Verified clean — and §8 is the model

§8 needs no change. Specifically checked and earned:

| §8 claim | status |
| --- | --- |
| "eleven of twenty resolved, none lost" | matches §6.3 |
| "Across-XR is the only fully crossed … **we are aware of**" | correctly hedged (and see H3) |
| "never a different day"; Questset one sitting | matches the corpus records |
| "seed variance is arm-specific … 0.010 … against 0.029" | matches RESULTS |
| "the headline arm choice is post hoc relative to the first registration" | volunteered against interest |
| "We compare against a single published system" + why that one | volunteered, with the reason |
| "the per-user rank-disagreement observation is exploratory" | matches §6.9 |
| "the posture scoping result is single-corpus" | matches |

§7's non-findings, checked so nobody re-checks them:

- **"the applications in these corpora that share a posture are the majority"** — defensible and
  countable: Across-XR's five applications are all played standing (measured), Questset is three standing
  titles and one seated, so of the ordered application pairs in these corpora the large majority share a
  posture. No action.
- **"the first data-side lever *we* have measured"** — correctly scoped to this work, with the mechanism
  explicitly declared unestablished. This is the paragraph the rest of §7 should be written like.
- **"No sensor-set claim is available from this work in either direction, and we do not make one"** —
  earned for the within-application gap, which is what it is about. It sits in tension with H6's heading
  but not with the BEAT result, which is a like-for-like comparison against a released model.

## Scope and limits

- §7 and §8 in full, on **both** `main` (`bec1309`) and `origin/review/paper-draft-2026-09-17`. §9
  (Conclusion) **not audited** — it is the obvious next place for the F2 shape, since a conclusion is
  where scoped results get restated without their qualifiers.
- §1-§4 remain unaudited; §5-§6 were covered by `paper_draft_verdict_audit.md`.
- Nothing recomputed from data; this node holds no checkpoints or corpora. H1 rests on comparing the
  draft's text against RESULTS' own mechanism sentence, and H2 on enumerating §3's corpora.
