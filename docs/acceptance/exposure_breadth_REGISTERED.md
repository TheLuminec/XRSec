# Experiment 2 — exposure breadth across applications, at scale — REGISTERED (2026-09-23, Coordinator; setup only, not run)

**Premise.** The only data-side lever ever measured to cross an activity boundary is *exposure*: P3 trained
on four Across-XR applications for 23 people and carried +0.049 rank-1 to the fifth, unseen application
(495 identities, no Nymeria). Breadth of people did nothing (identity count), breadth of corpora did
nothing (arm B, Nymeria transfer). This asks whether breadth of **tasks** does, at the scale the project
now trains at: every application we hold at once, with one held out.

**Arms, one run per held-out Across-XR application X ∈ {superhot_vr, half_life_alyx, beat_saber,
synth_riders, social_vr}, seed 1** (`configs/exposure_breadth_<X>_s1.yaml` from
`exposure_breadth_lists.py`):

| | training identities | tasks in training |
|---|---|---|
| **breadth-X** (new) | BOXRR + alyx (the treatment's post-draw lists minus 23 BOXRR; seed 1: 2,848 + 60) + Nymeria 141 + Across-XR 0–22 = **3,072** | Beat Saber, Alyx, 20 daily-life scripts, 4 Across-XR applications |
| P3-X (exists, Miami) | BOXRR 415 + alyx 57 + Across-XR 0–22 = 495 | Beat Saber, Alyx, 4 applications |
| in-domain treatment (exists) | 3,072 | Beat Saber, Alyx, 20 scripts — no Across-XR |
| zero-shot / Z-676 (exist) | 3,072 / 495 | Beat Saber, Alyx |

Identity matching by the post-draw rule: the treatment's lists (48 Nymeria held out, 1,071 pinned
validation, 141 BOXRR dropped) plus **23 more BOXRR training users dropped** (the last 23 in sorted
order after the 141) to make room for Across-XR 0–22; Across-XR 23–31 are the explicit validation users
for that corpus (Schach's own validation split, and it keeps the fractional draw off the corpus); Across-XR
32–48 are excluded and evaluated (`test_on_excluded=true`, beside the 48 Nymeria held-out, per-dataset
AUC recorded). Corpus for training: `CrossApplicationXR_LOAO_<X>` (X's sessions absent from every user,
`build_loao_corpus.py`). **The unit is the X-cells** — rank-1 at N=17 on users 32–48 with X as gallery or
probe — scored by the programme's own harness (`across_xr_alignment.py`, `--normalizer-dataset`), pooled
over the five X, paired on cells and users; Questset cross-game rank-1 (A1 protocol, N=17 and N=30) as
the second, fully unseen instrument.

**Dose, stated:** Across-XR at ≈ 16.7k of ≈ 780k training windows (≈ 2 %), against P3's 11.6 %. So a
null against P3 has two explanations (breadth does not help; the objective barely saw the applications)
and only the first licenses the conclusion — **a `balance_identities=cap` companion run on one X is the
registered follow-up if the null lands**, not an amendment after the fact.

**Registered outcomes** (rank-1, X-cells pooled over five X, user bootstrap):

| quantity | band | falsifier | landing between means |
|---|---|---|---|
| breadth-X − P3-X | **+0.00 to +0.06** — scale and daily-life tasks add a little to exposure | **< −0.03** — more and broader data *hurts* the held-out application (dose or interference) | −0.03 to 0.00: not resolved; run the cap companion before reading |
| breadth-X − treatment on the same X-cells | **≥ +0.03** — exposure to four applications carries, as P3 showed, on top of Nymeria breadth | **≤ 0** — with Nymeria in training, Across-XR exposure adds nothing: breadth of daily-life tasks already covered it | 0 to +0.03: unresolved at one seed |
| non-X cells, breadth-X − C2-lo-like full exposure | within ±0.03 | outside | — (control: removing one application leaves seen ones where they were) |
| Questset cross-game rank-1, breadth-X vs zero-shot | **+0.00 to +0.05** | **< −0.03** | +0.05 to +0.10: breadth transfers to a fully unseen corpus — seed it |

Which outcome is strong: the second row's falsifier — it would say Nymeria's 20 scripts already supply
what application exposure supplies, which collapses "task breadth" into one lever with two sources; the
first row's falsifier would say scale interferes. Five runs, one seed each (the unit is the application,
as in P3); ~90 min each on Miami. Not launched: this registration and the generator are the setup.
