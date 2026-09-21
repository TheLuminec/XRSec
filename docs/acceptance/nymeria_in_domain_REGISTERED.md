# Nymeria in-domain `dyn` arm — REGISTERED before the corpus is closed (2026-09-21, Coordinator, AVALON)

**Question (the user's, 2026-09-20):** *"Can't we use dyn and train on it to learn the movement
patterns without just looking at where they are?"* Every Nymeria figure this project holds is
**zero-shot**: `dyn` 0.53–0.55 on Nymeria at every window length and identity count measured, with
Nymeria never in any training set; and arm B (Nymeria 50 *in* training) was scored on the seven seated
corpora, never on held-out Nymeria users. **No row anywhere measures what a model learns about
AR-glasses daily-life motion when it is trained on that motion and tested on people it never saw.**
This arm measures that, under `dyn`, so the recorded-position location match (0.73, the shared SLAM
map) is removed by construction and whatever the model scores is behaviour.

## Corpus, as it will exist when this runs

The full released Nymeria v0.0: **236 participants, 1,100 head sequences** — every `recording_head`
in the 2026-09-20 URL index, which agrees sequence-for-sequence with the HuggingFace
`dataset_metadata.json` (1,100 / 236). *The paper's 264 participants and 1,200 sequences are the
collected set; 236 / 1,100 is the released set, and no export can reach the other 28.* Stage 3 is
streaming on AVALON as this is written (637 sequences, ~477 GB in, ~8 GB out); it adds the 5
single-sequence participants (walter_park, trevor_riley, david_ramirez, michael_griffin,
vincent_bell) and every further sequence of the other 231. Verification per sequence: size and sha1
against the index, gravity read from the raw trajectory before deletion, |q|, monotonic time, rate,
LF endings, local +Y → world-up, and the camera-rgb `T_Device_Camera` per device serial (the direct
test of the device-frame constant, which was derived from three devices).

**Corpus facts that qualify every number from this arm.** One sitting per participant, so a positive
pair is **cross-activity within one sitting** and this arm cannot pay the 1.1–1.6 point cross-session
cost. Sequences per participant now 1–8 (median 5), so within-user positives span several scripts.
CC BY-NC 4.0; no DUA machinery attaches; the corpus may travel to Miami.

## Design

Two arms, paired by seed, three seeds each, one code identity, identical everything except
`drop_users`:

| | data_dirs | exclude_users (`test_on_excluded=true`) | drop_users |
|---|---|---|---|
| **treatment** | BOXRR-23 + who_is_alyx + Nymeria | 48 Nymeria users, fixed list, seed 67 draw | none |
| **control** | same | the same 48 | **every other Nymeria user** (188) |

So both arms score **the same 48 held-out Nymeria users** (never trained on, never validated on), and
the control is the zero-shot model on exactly that population. `encoding=dyn`, 10 s, `window_stride=5`,
`bilstm`, `identity_softmax`, `embedding_dim=128`, `normalize=per_dataset`, `within_dataset_negatives`,
`cross_session_positives`, `epochs=120`, `early_stopping_patience=15`, `val_user_fraction=0.25`,
`exclude_users` **passed explicitly** (never the config default), `max_users` unset.

**Known nuisance, stated rather than met later:** the validation draw is over each arm's own pool, so
the treatment selects its epoch partly on Nymeria validation users and the control cannot. That is
part of what "the domain is in training" means and is not removed. Dose: Nymeria ≈ 190k of ≈ 900k
windows at 10 s stride 5 (~21%) — stated so a null is a result about the treatment and not the dose.

**MDD:** three paired seeds, multiplier 2.48×; at a paired sd of 0.012 the design resolves ~0.03, at
0.020 ~0.05. The band below is wider than that; the falsifier edge is not, and is scored on the
interval.

## Registered outcomes — headline is verification AUC on the 48 (the user's metric)

| quantity | **band** | **falsifier** | **landing between them means** |
|---|---|---|---|
| treatment `selected_test_auc` | **0.60–0.72** | **< 0.57** | 0.57–0.60: learnable but weaker than the alyx precedent (0.664 at 10 s); report as *weakened*, check the dose line above before reading |
| treatment − control, paired | **+0.05 to +0.15** | **CI upper < +0.02** | +0.02 to +0.05: a real but small in-domain gain; *not resolved* unless the interval clears zero |
| control `selected_test_auc` | 0.52–0.57 (replicates the zero-shot rows) | > 0.60 | 0.57–0.60: the richer per-user sequence set has made zero-shot Nymeria easier; say so before crediting the treatment |
| `position_lookup_auc` on the 48 | 0.68–0.78 (the 0.73 location match, both arms identical) | outside | outside means the population or the harness changed — stop and find which |
| `amplitude_auc` on the 48 | 0.50–0.56 | — | training-free dynamics baseline; the model must clear it or it learned nothing amplitude does not give |

**Which outcome is strong.** The falsifier is the informative half: daily life on AR glasses is 20
scripts, and if cross-activity positives from one sitting carry no learnable signature at ~188
training identities, that is the first evidence that activity *diversity within a person* defeats the
behavioural cue — the mirror of arm B, where diversity across people did nothing. The band holding is
weaker: it says the in-domain gain seen on Beat Saber and Alyx also appears on glasses, and it cannot
separate "the model learns gait/head dynamics" from "it learns the person's activity mix"; scoring
rank-1 by script pair (gallery script ≠ probe script) is the follow-up that separates them, and is
**not** registered here.

**Secondary, reported beside the headline:** EER on the 48 (the operating-point number the user's
1:1 on-device framing needs), and rank-1 at N=17 via `mode=curve` with `require_cross_session`,
implied-vs-measured per the Gaussian check.

## Launch conditions on Miami — none of this runs until every line holds

1. **The user has ruled on the volume.** Miami's kernel journal shows `ntfs3(sdc3)` MFT
   sequence errors on records 4a182–4a184 (MANIFEST.sha256 and feng-ms-7b51.results.jsonl) as recently
   as 2026-09-21 00:32, 62 lines this boot, rate-limited — bounded to two files, not spreading, and
   **not repaired**. Nothing writes to sdc3 before the user says so.
2. **The corpus gate on the files received, per file by sha256**, then a loader count on Miami that
   matches AVALON's (users, windows, shape) — the same gate Questset passed there.
3. **A hard memory cap on the job, not a pre-check:** launched under
   `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, cap = MemAvailable at launch
   minus 8 GB, **after** a fixture run shows the cap kills a deliberately bloating Python and lets a
   small one through (both directions). Pre-check `MemAvailable ≥ 30 GB` from `/proc/meminfo`, never RSS.
4. **Nothing else of ours on the node while it runs** — the queue unit holds 0 pending and stays that way.
5. **One marker per seed with `rc=`**, a watcher that fires on `.failed` as well as done, and **the
   result row pushed to origin the moment a seed lands**, before anyone reads it.
6. Both arms' `num_train_identities`, `num_drop_users`, `num_excluded_users` and `eval_split` read
   back from the rows and match this table before any number is quoted.

Amendments below this line are amendments, dated; the registration above is not edited.

---

## Amendment 1 — 2026-09-21, before any run, from Miami's review of the registration

Both points are facts about the design, knowable without running it, so amending is legitimate;
the original text above stays as written.

**A1. Three rows left an unnamed region at the top or bottom end** — the partition defect this project
has recorded twice, reproduced here by its own coordinator. Named now:

| quantity | region | **landing there means** |
|---|---|---|
| treatment AUC | **> 0.72** | above the alyx (0.664) precedent and approaching BOXRR in-domain (0.845): daily-life motion on glasses would be *more* identifying than a rhythm game. Report as exceeding, and do **not** credit it until `position_lookup_auc` and `amplitude_auc` on the same 48 are confirmed inside their bands and the script-pair follow-up (gallery script ≠ probe script) has been run — an activity-mix cue would land here too |
| treatment − control | **> +0.15** | larger than any in-domain gain this project has measured; check the control row first — a control that fell below its band produces this delta for the wrong reason |
| control AUC | **< 0.52** | zero-shot Nymeria is **at chance on this population**, which the 0.53–0.55 rows never were on theirs; report as a finding about those rows (population-dependent) before reading the treatment against it |

**A2. The delta conflated "Nymeria in training" with "+188 training identities".** The control dropped
188 users and the treatment kept them, so the arms differed in identity count as well as domain — and
identity count is the one data-side lever measured to move in-domain results a great deal. Arm B met
this exactly and solved it by **swapping at fixed count**. Adopted here:

| | trains on | identities |
|---|---|---|
| **treatment** (amended) | BOXRR minus the **last 188 BOXRR *training* users** + alyx + 188 Nymeria | matched |
| **control** | BOXRR (all) + alyx, `drop_users` = the 188 Nymeria | matched |

The 188 BOXRR users are removed from the **post-draw training list**, not the pre-draw pool — the
New Gen correction: pin the validation draw (`val_user_fraction=0` with the pipeline's own 25 % draw
written to a file and passed as `validation_users`), then `drop_users` the last 188 BOXRR training
users in sorted order. Verified on the lists the loaders hold before either arm runs: equal training
identity counts, BOXRR training users nested (treatment ⊂ control), alyx identical. Cost to the
treatment: 188 of 4,020 BOXRR identities, cross-activity for the held-out users, where transfer is
already saturated (3,072 → 4,096 moved zero-shot by 0.001). Registered: this swap moves the treatment
by **less than 0.01** relative to the unswapped design; if the unswapped treatment is ever run and the
gap exceeds 0.02, the identity-count reading was live and the amendment was necessary rather than tidy.

The registered bands and falsifiers are unchanged by A2; what changes is the sentence the delta
supports — *"a Nymeria identity is worth more than a BOXRR identity to a Nymeria held-out user"* rather
than *"adding Nymeria helps"*, which is the sharper and the defensible one.

**A3 (launcher, not registration).** Condition 3 now reads `gated_launch.sh` as reviewed: marker
directory required and absolute (off sdc3), free-memory check before the fixture, the positive control
itself capped, lock + active-scope + heavy-python guards that refuse when their probes are absent, and
`oom_kill` / `peak_mb` read from the job's own cgroup beside `rc` so 137 is never asserted as a cause.
**What it bounds is one job, not the machine** — the honest sentence is "one job cannot eat the box",
and condition 4 (the queue unit) remains a user decision, not a mechanism.

## Amendment 2 — 2026-09-21, the user's ruling on launch conditions 1 and 4

**Condition 1 is satisfied by the user's decision, not by a repair.** Shown Miami's kernel journal
(62 `ntfs3(sdc3)` MFT-sequence lines this boot on records 4a182–4a184, last at 00:32, bounded to two
files, not spreading), the user ruled: *"It may just have been because of a power outage breaking the
drive, for now we will keep going as usual. With Miami repaired."* Recorded as a ruling with the
evidence beside it; the journal reading stands as a fact about the volume. The corpus lands on Miami
and the arm runs there.

**Condition 4 becomes a mechanism.** The user authorised disabling the queue unit (*"Disable if you'd
like but it's in a safe environment"*); Miami stops and disables `xrsec-queue.service` so nothing
ungated can start beside the capped job, and records the command to re-enable it.

## Amendment 3 — 2026-09-21, the swap count is 141, not 188, and the validation draw is pinned

A fact about the instrument, read from `select_validation_users`: the fractional validation draw is
**one draw over the pooled candidates of every `data_dir`**, `count = round(pool × 0.25)`, seeded from
the run seed. So Amendment 1's "drop 188 BOXRR training users" was the pre-draw number — the very error
the New Gen correction names. Post-draw, the treatment's 188 non-held-out Nymeria users split into
**47 validation + 141 training** (round(188 × 0.25) = 47, fixed for every seed; which 47 varies), so the
identity-matching swap removes **141** BOXRR users from the treatment's training list, not 188.

**Pinning.** Both arms pass the **control's** validation draw explicitly as `validation_users` (BOXRR +
alyx users only, since the control drops every non-held-out Nymeria user before the draw). Explicit users
remove those two corpora from the fractional draw, so in the treatment only Nymeria is drawn — 47 of 188 —
and BOXRR/alyx validation is identical across arms by construction. Then:

| | training identities | BOXRR training | alyx training | Nymeria training |
|---|---|---|---|---|
| control | **3,072** | 3,015 | 57 | 0 (188 dropped) |
| treatment | **3,072** | 3,015 − 141 = 2,874, a **subset** of the control's | 57, identical | 141 |

The 141 dropped are the **last 141 BOXRR training users in sorted order** (BOXRR users not in the pinned
validation list), chosen by rule rather than by draw so the nesting is checkable by eye. All lists are
produced per seed by `docs/acceptance/nymeria_in_domain_lists.py` on the node that runs the arm, from
the pipeline's own `select_validation_users`, and the counts above are asserted by that script against
the directories the loader will read — the counts a row reports (`num_train_identities`,
`num_drop_users`, `num_excluded_users`) must then match them before any figure is quoted.

The held-out 48 are fixed at `docs/acceptance/nymeria_in_domain_heldout48.txt` (drawn once, seed 67,
over the 231 participants with ≥ 2 sequences; the 5 single-sequence participants train only, since their
only positives would be same-recording). Sequences per held-out user: 2–8, median 5.

## Amendment 4 — 2026-09-21, from Miami's dry run of the generator (all facts about the instrument)

1. **Every list is now explicit on both arms.** The treatment's 47 Nymeria validation users were a
   runtime draw, and a numpy `Generator` stream is not stable across feature releases (AVALON numpy
   2.4.3, Miami 2.5.3) — the count was safe (round(188 × 0.25) = 47), *which* 47 was not. The treatment
   config now carries V + the 47 (1,071 `validation_users`), so the pipeline draws nothing at runtime.
2. **The lists travel as committed reference files by NAME**, `nymeria_in_domain_lists_s{1,2,3}.json`
   (`<corpus>/<user>`), drawn once on AVALON; every other node rebuilds its configs with
   `--from-reference`, which asserts each named directory exists there and reproduces the digests. The
   digests in the JSON supersede the two quoted in an earlier message to Miami (the digest is now over
   sorted names; `dropB` is unchanged, `V` is not comparable to the old value).
3. **The held-out 48 are asserted absent** from both arms' validation, drop and training lists, and
   validation and drop are asserted disjoint — asserted, not inferred, per the vacuous-guard rule.
4. YAML entries are quoted, so a directory name carrying a colon or a space errors rather than silently
   composing a different config.
5. **On Miami the interpreter is `.venv313/bin/python`** (3.13.15). `.venv` there is Python 3.14, on which
   Hydra 1.3.6 cannot parse its own arguments, so a run issued with `.venv/bin/python` would fail at
   startup inside the gated scope with an rc that has nothing to do with the experiment. Miami
   re-checked it live. The run shape in condition 5 is issued with `.venv313`.

Digests (over sorted user names): Nymeria users `21a122db402a` (236); seed 1 V `72a5ff18e4a6`, Vtreat
`e0dc503a5c88`, dropB `666edb78f6a4`; seed 2 V `1e079dc28f2d`, Vtreat `7028540b705d`, dropB `e4db0aa32be6`;
seed 3 V `eeea7263f192`, Vtreat `65168fd64c37`, dropB `46300540e09b`; dropC `8f0ffa0ab435` and held
`8842e13112a7` on every seed. Both arms 3,072 training identities, BOXRR nested, alyx identical, on all three.
