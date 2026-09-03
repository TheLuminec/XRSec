# Literature briefing for the XRSec Coordinator

From: research agent (lit review). Read date: 2026-09-03.

I read four documents end to end and checked every claim below against the current
repo (`configs/config.yaml`, `model/`, `docs/ROLLING_WINDOW_PROPOSAL.md`, `CLAUDE.md`)
so that nothing here is advice you have already implemented.

| # | document | what it is |
| --- | --- | --- |
| T | `CSE_Thesis_KeLyu_2025.pdf` | Miami MS thesis, advisor Xianglong Feng. Siamese + TimesNet + LSTM + 1D-CNN on head motion. **This is our repo's direct ancestor.** |
| N | `nsf_ERI_2023_ARVRsecurity.pdf` | Feng's NSF ERI proposal. **Source of `paper_gnn_bilstm`** (Ga/Gp GNN + BiLSTM + two attention layers). |
| A | `frvir-04-1272234.pdf` | Rack et al. 2023, "Who is Alyx?" — the dataset we just wrote a converter for, plus its published benchmark. |
| X | `frvir-7-1743491.pdf` | Schach, Rack, McMahan, Latoschik 2026 — cross-application identification, 49 users x 5 VR apps, **leave-users-out protocol like ours**. |

A and X are the same Wurzburg group and are the closest thing this field has to a
methodological standard. T and N are our own lineage. The two sets disagree, and the
disagreement is the most useful thing in this briefing.

---

## 1. Provenance: the retired 0.85 is explained

`CLAUDE.md` records the historical 0.85 as "configuration lost, never reproduced, the
corrected protocol cannot account for it." The two ancestor documents account for it
completely. **Neither T nor N ever holds out users.**

- **T §3.3.2 / §5.1:** "we split the dataset into training (80%) and testing (20%)
  sets randomly", per video, where "each video contains sensor data from all users
  who watched it." Every test user is a training user. Headline overall accuracy:
  **0.9852** (Dataset 1) and **0.8364** (Dataset 2, which is our VR_User_Behavior).
- **N §3.1.1:** Case A = all users' segments pooled then randomly split → **>90%**.
  Case B = train on some videos, test on held-out *videos* → **~80%**. Case B holds
  out content, not people.

So 0.85 sits exactly between N's Case A and Case B, on the seen-user protocol, and
Dataset 2's 0.8364 in T is a near-exact match. The recommendation in `CLAUDE.md` — do
not treat it as a target — is correct, and can now be stated more strongly: **it is a
seen-user number and is not comparable to anything we report.** It is worth one line
in the README so it stops being mysterious.

Two corroborations of our own noise-floor work fall out of T:

- T Table 5.6, Dataset 2, per-video overall accuracy ranges **0.5615 to 0.8857** on
  one model, one protocol, one dataset. That is a 0.32 spread driven purely by which
  content is held out — the same phenomenon as our measured 0.114 spread over which
  *users* are held out, and larger. Content is a second variance axis we do not
  currently stratify on.
- T §6.1 states plainly that the model "exhibits notable limitations when extended to
  multi-class user identification" and §6.2 proposes metric learning, triplet loss,
  and prototypical networks as *future work*. **We already did that.** `identity_softmax`
  (+6.5, 5/5 folds, three backbones) is precisely the fix the thesis identified and did
  not implement. That is a result worth foregrounding, not a footnote.

---

## 2. Input encoding is a confounded axis in our sweeps

`CLAUDE.md` reports "extractor architecture: ~0, spread under 1 point" across three
backbones. Rack et al. report the opposite about *encoding*, holding architecture fixed:

| encoding (A §4.1.2, §5) | what it removes | result |
| --- | --- | --- |
| SR — scene-relative raw | nothing | worst |
| BR — body-relative | scene position/yaw | better |
| BRV — body-relative velocity | static pose | better still |
| BRA — body-relative acceleration | velocity offset | **best** (their headline model) |

A §5: "the acceleration encoding (BRA) yields superior performance compared to BR and
BRV... abstracting non-motion-related information can enhance the generalization
capabilities of the model. We hypothesize that more sophisticated encoding techniques
could lead to even better identification models."

In our repo, encoding is **not a config axis** — it is baked into individual extractors
(`motion_tdnn` and `motion_gram` derive kinematics via `model/extractors/_kinematics.py`;
`bilstm` and `paper_gnn_bilstm` consume raw channels). So "architecture ties" is
measured across models that also differ in encoding, and "encoding wins" has never
been measured with architecture held fixed. Those are the same experiment run
backwards, and neither answers the other.

**Proposal:** promote encoding to a top-level config key alongside `channels` and
`center_position` — `encoding: raw | br | brv | bra` — applied in the dataset layer so
every extractor sees the same transform. Then one sweep with `sweep.folds` over
{3 extractors} x {4 encodings} answers both questions at once. Note that
`center_position=true` is already a partial BR (it removes translation but not yaw),
so we have one point on this curve and it is the *worst* one in Rack's ordering.

This also bears directly on the `center_position` result. Our centred arm is measured
at exactly one encoding — the weakest one in the published ordering — so "movement-only
sits near chance" is a statement about raw centred coordinates, not about movement as
such. BRV and BRA are the encodings under which the field's movement-based results were
obtained, and we have never run either.

---

## 3. Our windows are 5-10x shorter than every published result

| source | window | rate | duration |
| --- | --- | --- | --- |
| us (`configs/config.yaml`) | 40 frames | 20 Hz | **2 s** |
| N §3.1.1 (our ancestor) | 10 frames | 10 Hz | 1 s |
| A §4.2.1 | 300 frames | 15 Hz | **20 s** |
| X §5.4 (similarity) | 450 frames | 30 Hz | **15 s** |
| X §5.5 (classification) | 600 frames | 30 Hz | **20 s** |

A §4.2.1 cites Rack et al. 2022 for the design rule directly: *"it is preferable for
samples to cover a longer duration with a lower frequency than a shorter duration with
a higher frequency."* We are on the wrong side of that rule, and the ancestor papers
are the reason — we inherited a 1-second window from N and doubled it.

**This is not the same lever as `mode=curve`, and I want to be precise about why.**
Averaging k independent 2-second embeddings reduces variance; it cannot manufacture
features that span 20 seconds. Exploration rhythm, revisit patterns, the cadence of
turning to re-orient — those live at timescales our extractors never see. So the
`ROLLING_WINDOW_PROPOSAL` note that a flat k-curve would indicate "error dominated by
between-session variation" has a third possible reading: **the per-window
representation has a low ceiling that averaging cannot raise.**
`sample_time: 10, sample_rate: 10` keeps `seq_len` at 100 (2.5x current cost) and
separates the two hypotheses.

---

## 4. Enrollment matters far more than probe length — and our k-curve is symmetric

A §4.3, Figure 6, is the most actionable single result in these papers:

| | |
| --- | --- |
| 1 min enrollment, 10 min use-time | **10%** |
| 10 min enrollment, 1 min use-time | **54%** |
| 1 min enrollment, 25 min use-time | 13% |
| 5 min enrollment, 1 min use-time | 37% |
| 25 min enrollment, 5 min use-time | 71% |

"if there is too little enrollment data, the model cannot make accurate predictions,
even with more use-time data."

`model/templates.py` builds a template of k windows on **both** sides. If the same
asymmetry holds for verification, the reference side is worth far more than the probe
side, and a 2-D sweep over `(k_ref, k_probe)` would find a much better operating point
than the symmetric diagonal — at no extra forward passes, since the embeddings are
already computed once. This looks like a one-parameter change to an existing,
already-approved feature.

Related, from X §5.4.1: their template is the **unit-norm mean direction (extrinsic
spherical mean)** that maximises average cosine similarity within a user. `templates.py`
L2-normalises then means then renormalises for cosine heads, which is the same thing to
first order — worth confirming, and worth noting we independently landed on the right
construction.

---

## 5. Per-dataset embedding rotation (GOPA) — a paper-shaped opportunity

X §6.2.5 is the most interesting result in the newest paper. Embedding spaces learned
for different VR *applications* turn out to differ **only by an orthogonal
transformation** — rotation/reflection, no scaling or translation. Aligning them:

- mean within-user cosine similarity **0.798 → 0.963**
- cross-application accuracy **18.0% → 52.3%** single-window, **30.8% → 94.3%** at 10 min

They then disqualify their own result (X §9): the rotations were fitted on the test
users, so it is "a diagnostic upper bound, not a deployable, generalizing solution."
X §8 names the fix as future work: *"develop and evaluate protocols that learn
orthogonal transformations only on training/validation users and then apply them to
unseen test users."*

**We are unusually well positioned to do exactly that.** Our corpus is 7 datasets =
7 domains, our folds are already stratified by dataset, and `val_user_fraction` already
gives us a training-disjoint user group to fit on. The honest version of their
experiment is a natural fit for our protocol, and the negative result is publishable too.

Two related notes:

- Our `normalize=per_dataset` corrects input offset and scale. Nothing in the pipeline
  corrects **embedding-space orientation**, which X says is what actually differs.
  That is a plausible reason the +11 from normalization plus within-dataset negatives
  did not go further.
- X §8 also names **Domain-Adversarial Neural Networks** (gradient reversal on an
  auxiliary dataset classifier) as the other route. That is a small addition to
  `identity_train.py` and is the training-time analogue of `within_dataset_negatives`,
  attacking the same shortcut that was worth 11 points when we closed it at pair level.

---

## 6. Sampling: we alias high-rate data as well as duplicating low-rate data

`ROLLING_WINDOW_PROPOSAL.md` already records that nearest-neighbour lookup fabricates
motion *below* the native rate (ViewGauss 50.5% duplicate frames). The papers show the
other half of the same bug. T §4.1.2 decimates by **averaging every sample inside each
window** rather than picking one; A §4.2.1 and X §5.2 **resample** to a fixed rate.

Nearest-point selection at 20 Hz against a 250 Hz source (NJIT) or 125 Hz (EyeNavGS)
discards 92% of the samples and aliases head tremor into the retained signal. So the
same one-line fix — bin-average or interpolate instead of nearest — addresses both the
duplication and the aliasing, and it is a prerequisite for any derived-velocity or
BRA encoding to mean anything (§2 above depends on it).

---

## 7. We should report a metric the field can compare against

Everything we report is pairwise: accuracy at `logit > 0`, AUC, EER. Everything they
report is **rank-1 closed-set identification among N held-out users**, plus a
sequence-accuracy curve over observation time:

- A: 71 users, single 20 s window, **76.6-78.3%** rank-1 (seen users, cross-session).
- X: **17 unseen users**, single 15 s window, **83.1%** rank-1 within-application,
  **78.5%** averaged over all applications; **100%** at a 10-minute sequence.

X's split — 23 train / 9 validation / 17 test, disjoint users — is our protocol exactly.
That makes their 78.5% the *directly comparable* external number, and we currently have
no figure that can be placed next to it. Adding rank-1 / CMC over the held-out fold to
`model/metrics.py` costs one function over embeddings we already compute, and it turns
"0.669 pairwise on unseen users" — which no reviewer can situate — into a number on the
field's own axis.

Worth also noting X §7.2's caution before we celebrate any such number: their model was
trained on 23 users, and "in realistic settings with substantially more users, the
embedding space would likely be more densely populated, which could in turn reduce
identification accuracy." Rank-1 must always be quoted with N.

---

## 8. What the papers confirm we already have right

Short list, so these don't get re-litigated:

- **Cross-session positives.** A's whole design rationale (§2.2) is that single-session
  datasets "cannot be tested how well a model would recognize the same person on a
  different day." Our 1.1-1.6 point correction is the right call and the right size.
- **Leave-users-out.** X uses disjoint user splits and treats it as the only meaningful
  protocol for a pretrainable model. T and N do not. We are aligned with the current
  literature and ahead of our own lineage.
- **Validation-selected epochs.** A §4.1.4: "We save a snapshot of each model at its
  validation highpoint... since performance can already start declining." Same fix,
  same reason.
- **Seed/fold variance.** A §4.3: their CNN varies 1.7 points over 10 seeds, but they
  measured Liebers et al.'s setup varying by **15 points**, and A §2.2 reports ±8 points
  from reseeding alone on a small dataset. Our insistence on `sweep.folds` is standard
  practice in this specific literature, not over-caution.
- **Similarity/embedding learning over classification.** X §6.3: the classification
  model gets 43.2% where the similarity model gets 78.5%. Our `identity_softmax` +
  cosine is the right family.
- **Multi-window aggregation.** Confirmed as the single largest post-hoc gain in the
  field: A goes 77% → 95% (2 min) → 99% (7 min) purely by majority-voting windows;
  X goes 83% → 100% within-application at 10 minutes. `mode=curve` is aimed at real
  headroom. See §3 and §4 for the two caveats on how to spend it.

---

## 9. Ranked suggestions

1. **`encoding: raw|br|brv|bra` as a first-class axis** (§2), swept against the three
   extractors with `sweep.folds`. De-confounds our strongest negative result and puts
   the `center_position` finding on the encoding the field actually uses.
2. **Asymmetric `(k_ref, k_probe)` in `mode=curve`** (§4). Near-free; literature says
   the asymmetry is large.
3. **`sample_time: 10, sample_rate: 10`** (§3). Tests whether the per-window
   representation, not the evidence budget, is the ceiling.
4. **Rank-1 / CMC over held-out users in `metrics.py`** (§7). Makes us comparable.
5. **Bin-averaged resampling** (§6). Prerequisite for 1 and 3.
6. **Training-user-only GOPA alignment, or a DANN head** (§5). Highest ceiling, most
   work, and the most obviously novel relative to X.
7. **One README line retiring the 0.85 as a seen-user number** (§1).

Numbers 2 and 7 are close to free; 1 is where I would spend the first real compute.

---

### Caveats on this briefing

- I have read the papers, not run anything. Every repo claim above is from source
  inspection; none of the proposed experiments has been executed.
- A's 76.6-78.3% and 95%/99% figures are **seen-user, cross-session** (train session 1,
  test session 2, same 71 people). Only X's 78.5%/83.1% are leave-users-out and
  therefore comparable to ours. I have tried to keep that distinction explicit
  everywhere; if a number appears without a protocol tag, treat it as suspect.
- A and X both track head **and both hand controllers**. We track the head only, by
  design. Their absolute accuracy figures are therefore an upper bound on what a
  head-only rig should be expected to reach, and the comparison in §7 should be read
  with that in mind. Their *methodological* findings — encoding order, window length,
  enrollment asymmetry, embedding-space rotation — are about the pipeline rather than
  the sensor set and transfer regardless.
- T's Dataset 2 is 48 users / 9 videos with quaternions, which matches our
  VR_User_Behavior. T's Dataset 1 (50 users, 10 videos, Euler angles) is a different
  corpus we do not appear to hold.
