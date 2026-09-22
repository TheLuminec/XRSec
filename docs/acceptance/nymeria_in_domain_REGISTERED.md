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

*Digest convention (Miami, same day, after reproducing all 15 digests and 13 list invariants
independently from the JSON): the digest is the first 12 hex characters of sha256 over the **bare user
names, sorted, joined by newline** — not the `<corpus>/<user>` strings the JSON stores (hashing those
gives e.g. `ea9de44af445` for seed 1's V, not `72a5ff18e4a6`). A node that hashes the full string will
report a mismatch and blame its corpus; the arithmetic is in `digest()` in the generator.*

## Amendment 5 — 2026-09-21 05:20, corpus closed; the loader figures both nodes must reproduce

The corpus this arm runs on is now fixed: **236 participants, 1,100 sequences**, manifest
`nymeria_manifest_avalon.txt.gz` (1,102 files under `users/`). Loader on AVALON, `channels=full`:

| setting | users | windows | shape |
|---|---|---|---|
| 5 s @ 20 Hz, `raw` (the standard corpus check) | 236 | **244,019** | (244019, 7, 100) — two independent cache builds agree |
| **10 s @ 20 Hz, stride 5, `dyn` (this arm's setting)** | 236 | **242,919** | (242919, 7, 200) |

Miami reproduces both before seed 1 (the second one as the first line of seed 1's own stdout). **Dose,
restated from the measured index:** ≈ 1,029 windows per Nymeria user at the arm's setting, so the 141
Nymeria training users supply ≈ 145k windows beside ≈ 495k from BOXRR + alyx (the zero-shot arm's
519,211 scaled to 2,931 of 3,072 identities) — **≈ 23 % of training windows**, revising the ≈ 21 %
estimate in the design section. Read the row's own loader lines for the exact figure.

*Correction to Amendment 5 (Miami, same hour, read from `model/dataset.py:203`): the loader prints
**one pooled line** for all `data_dirs` after filtering `exclude_users`, `drop_users` and `keep_users`,
so seed 1's stdout cannot show "242919 samples from 236 users" — its Nymeria contribution is 141 users
(treatment) or 0 (control) inside a three-corpus total. The 10 s gate is therefore a **standalone
Nymeria-only build** at the arm's setting, exactly like the 5 s one, whose own print must read
`Loaded 242919 samples from 236 users`. Both standalone gates run under `gated_launch.sh` with markers,
which also yields an index build's `peak_mb` **before** seed 1 rather than after. Seed 1's pooled line
is reported verbatim beside the row's `num_train_identities` / `num_drop_users` / `num_excluded_users`
as a cross-check, not as the Nymeria gate. My sentence was an inference about the instrument that a
grep would have settled.*

## Amendment 6 — 2026-09-21 ~08:00, seed 1: control landed, treatment OOM-killed by the cap, identity step

**Control s1 (identity `517cdaa57b`, row on `miami-server` at d6c453e):** `selected_test_auc` **0.5415**,
`position_lookup_auc` 0.7233, `amplitude_auc` 0.5081, `eval_positive_fraction` 0.500, pooled loader
lines 519,211 / 3,072 (training — exactly the zero-shot arm's count, an independent corroboration of
the composition), 47,796 / 48 (held-out), 167,128 / 1,024 (validation). Bands: control 0.52–0.57
**holds**, position lookup 0.68–0.78 **holds**, amplitude 0.50–0.56 **holds**. **`best_epoch` = 120 of
120: right-censored**, the validation-selected epoch is the last one, so 0.5415 is a lower bound on
what this arm reaches under a longer budget. The programme's zero-shot arms sat at 116–118 of 120
under the same budget. The budget stays at 120/15 for comparability; **if the treatment censors too,
a longer-budget pair is a registered follow-up**, not a change made after seeing a number.

**Treatment s1: `rc=137 oom_kill=1 peak_mb=32768`, killed at the 32 GB cap 56 s in, during the index
build, before any epoch.** The machine was untouched (MemAvailable back to 44 GB). Miami measured
the cause by sampling the cgroup rather than by inference — and corrected its own earlier reading
that the 14.8 GB gate peak was mostly page cache: Nymeria-only 10 s stride 5 read **max anon 12,219 MB
against 1,302 MB of cache**, and the same build under `raw` read 3,004 MB anon. **`encoding=dyn` cost
~4x `raw` at index build**; the pooled control build peaked at 30,752 MB of 32,768. Certificate:
`docs/acceptance/nymeria_in_domain_memory_miami.md`. **The cap was not raised** — fitting a job by
shrinking the guard is the guard's failure mode, and the user's instruction forbids it.

**The fix is in the encoding, and it is proven bit-identical.** `apply_encoding` now works in blocks
of 4,096 windows (`ENCODING_BLOCK_WINDOWS`), with the previous body kept verbatim as `_encode_block`;
every encoding is per-window, so nothing changes. Measured on AVALON under `gated_launch.sh`,
Nymeria-only at the arm's setting:

| | before | after |
|---|---|---|
| output tensor sha256 (242,919 × 7 × 200, float32) | `cecf77bd9fc9483f` | **`cecf77bd9fc9483f`** |
| `window_mean_positions` / amplitudes / dataset ids / session ids / start times | 5 shas | **all 5 identical** |
| max RSS | 13,554 MB | **4,629 MB** |
| cgroup peak (page cache included) | 14,551 MB | **4,535 MB** |

plus a unit test asserting `torch.equal` between block-wise and whole-tensor encoding for every
encoding on both channel sets at a block size that does not divide the window count. **This moves
`code_identity` from `517cdaa57b` to `03ea8e2376`.** Two logging repairs ride in the same step:
`num_train_identities` is now written on the `identity_softmax` path (it was silently absent from
every identity-trained row — Miami: 0 of 25 in its shard — because `WindowDataset` carries the count as
`num_classes` and no `sample_index`), and the generator sets `experiment_name` (the logger's key;
`experiment` composed but was inert, so the s1 row reads `xrsec`). `eval_split` lives in the checkpoint,
not the row, by design; condition 6 reads it from there.

**Acceptance for the identity step, before the pair is read:** Miami re-runs control s1 under
`03ea8e2376` and it must reproduce **0.5415** to same-device cuDNN run-to-run precision (the gap is
reported as a number; digit-identical is the strongest form) with the same three loader lines; that
reproduction is what certifies the memory change touched no numerics on the GPU path. Then treatment s1.
The `517cdaa57b` control row stays in the shard as the pre-step record; the pair that is read is the
two rows under `03ea8e2376`. Both markers' `peak_mb` go on record.

## Amendment 7 — 2026-09-21, the control re-run under 03ea8e2376 was VOID: a 2 s `raw` run wearing the arm's name

**Miami caught it before reporting a number.** The re-run's row read `sample_time` 2, `encoding` raw,
`seq_len` 40, `best_epoch` 2, `selected_test_auc` 0.5985 — every value the config default supplies —
because the regenerated configs held no `encoding`, `sample_time` or `sample_rate` key. **Cause, mine:**
the `experiment` → `experiment_name` edit to the generator placed a `#` remark on the dict's first line,
which commented out the three keys that followed it on that line. `py_compile` passed, Hydra composed,
and both Miami and I checked only the key that had changed. **This is the project's recurring bug in its
textbook form — a config default silently standing in for the intended experiment and returning a
plausible number — produced by the coordinator while fixing a logging key, one step after the
identity step whose acceptance the run was meant to be.** 0.5985 at epoch 2 *is* a sensible 2 s raw
result on those users; nothing in it says "wrong experiment".

**What the void run does establish**, tested by Miami rather than inferred: under `03ea8e2376`, `dyn` is
still applied and still removes the static cue (per-window |mean position| 5.29e-10 against 0.742 for
`raw` on one alyx user at 10 s stride 5); the standalone 10 s stride-5 `dyn` Nymeria gate re-ran MATCH
at 242,919 / 236 with **`peak_mb` 4,419 against 14,798** before the fix; identity prints `03ea8e2376`
with an all-LF tree; all 15 list digests unchanged. And the sample cache is unaffected: encoding is
deliberately outside its key because `apply_encoding` runs on the assembled index after per-user
caching; the "0 hit, 3,072 built" that alarmed was the 2 s key naming entries that had never existed.

**Fixes.** The generator now refuses to write a config unless every fixed key reads back from the
written YAML with its intended value and every list has its expected length; the remark is on its own
line. Verified on AVALON through Hydra itself (`main.py --cfg job`): both s1 configs compose with
`sample_time` 10, `sample_rate` 20, `encoding` dyn, `window_stride` 5, `epochs` 120, patience 15,
`experiment_name` nymeria_in_domain. **Rule from here, both nodes: print the composed config (seed,
sample_time, sample_rate, encoding, window_stride, epochs, patience, extractor, objective) immediately
before every launch and paste it beside the result.** Verify the artefact the run consumes, never the
edit that produced it.

**The void row** (experiment `nymeria_in_domain`, `sample_time` 2, `encoding` raw, `code_identity`
`03ea8e2376`, Miami's shard) is pushed with its commit message marking it VOID rather than deleted —
the record shows the run happened — and **any analysis of this arm must filter on `sample_time == 10`
and `encoding == "dyn"`, not on `experiment` alone**: the `sweep_id`-mixture trap, in a new key.

The acceptance sequence of Amendment 6 restarts from control s1.

## Amendment 8 — 2026-09-21, the identity step is accepted: control s1 reproduces digit-identical

Under `03ea8e2376`, composed config printed and checked first: `selected_test_auc`
**0.541536678870519 against 0.541536678870519, gap exactly 0.0**; acc, position lookup, amplitude,
lookup and `best_epoch` identical to every recorded digit; `seq_len` 200; 120 `dyn` NOTE lines; loader
lines 519,211 / 3,072, 47,796 / 48, 167,128 / 1,024. Marker `rc=0 oom_kill=0 peak_mb=11,436` against
30,752 before the fix, wall clock 4,790 s against 4,840 s. **The memory change is proven numerics-free
on the GPU path by reproduction, which is what the acceptance was for.** Treatment s1 is running under
the same cap.

**Condition 6, as it can actually be met for this arm.** `num_train_identities` is still absent from
the row, and the reason is one layer below my repair: `results_log.py` lists the field in `FIELDS` and
never copies it out of `history` (the only field in that file appearing once, not twice), and
`eval_split` was never wired into the row. My `train.py` `elif` cannot fire and carried a wrong
mechanism. **Decision: not fixed now.** A one-line logger fix moves `code_identity` again, mid-pair, to
record a number already given twice — the loader's own stdout line and the generator's asserted lists,
which refuse if the counts disagree. For this arm, condition 6's identity count is read from the run
log's loader line and the generator printout (both quoted beside each row), `num_drop_users` /
`num_excluded_users` from the row, and `eval_split` from the checkpoint. The logger fix, the dead
`elif`'s removal and its comment's correction go into the next identity step after seed 3.

## Amendment 9 — 2026-09-21, seed 1 pair complete; the budget follow-up registered before it is needed

| seed 1, 48 held-out Nymeria users | control | treatment | paired delta |
|---|---|---|---|
| `selected_test_auc` | 0.5415 | **0.7082** | **+0.1667** |
| `selected_test_acc` | 0.5047 | 0.6411 | +0.1364 |
| `best_epoch` / budget | 120 / 120 | 118 / 120 | both censored |
| `position_lookup_auc` | 0.7232780555884043 | 0.7232780555884043 | **byte-identical** |
| `amplitude_auc` | 0.508104446861479 | 0.508104446861479 | **byte-identical** |
| training identities (log + generator) | 3,072 | 3,072 | matched |
| test line | 47,796 windows / 48 users | 47,796 / 48 | identical |
| `peak_mb` under the cap | 11,436 | 11,745 | (was OOM at 32,768) |

Rows on `miami-server` at 402443e, pushed before reading. **Against the registration:** treatment
inside its band (0.60–0.72, near the top); the delta lands in the region Amendment 1 named — above
+0.15 — whose instruction was to check the control first, and **the control sits inside every one of
its bands** (AUC, position lookup, amplitude), so the delta is not manufactured by a depressed control.
The byte-identical training-free baselines are the `lookup_auc`-moved diagnostic used in the confirming
direction: both arms scored the same 48 users on the same pairs. `lookup_auc` on the encoded windows
differs (0.5024 vs 0.5036), as it should — that column is `dyn` rounding residue of the model's input.

**One seed. No interval yet; seeds 2 and 3 decide whether +0.17 is the effect or a draw.** Caveats
attached to any quotation: both arms budget-limited; the treatment selects its epoch partly on Nymeria
validation users and the control cannot (registered nuisance); positives are cross-activity within one
sitting and cannot pay the cross-session cost; the script-pair follow-up (gallery script ≠ probe script)
that separates "learns head dynamics" from "learns the person's activity mix" is not yet run.

**Budget follow-up, registered now, run AFTER seed 3 and not instead of it.** Both arms censor at 120,
so the registered condition is live. Seed 1, both arms, `epochs=240`, `early_stopping_patience=15`,
everything else identical, composed config printed. **Prediction:** the control is at its zero-shot
level (0.53–0.55 across every scale measured) and moves by **0.00 to +0.01**; the treatment moves by
**0.00 to +0.03**; the paired delta stays within **±0.03 of +0.167**. **Falsifier:** the delta shrinks by
more than 0.05 (the 120-epoch gain was partly the treatment converging faster, not learning more) or
the control rises above 0.57 (the zero-shot ceiling was the budget, which would re-open the zero-shot
rows). **Landing between:** delta moves by 0.03–0.05 — report as budget-sensitive, quote the 240-epoch
pair as the headline with the 120-epoch pair beside it. Cost: ~2 × 160 min on Miami.

## Amendment 10 — 2026-09-21, seed 2 replicates; the script-pair follow-up REGISTERED before seed 3 lands

| seed | control | treatment | paired delta | budget (c / t) |
|---|---|---|---|---|
| 1 | 0.5415 | 0.7082 | +0.1667 | 120 / 118 of 120 |
| 2 | 0.5269 | **0.7263 — above band, NOT credited** | +0.1994 | 116 / 120 of 120 |

Rows at 87ad92f (`miami-server`). Seed 2's control inside all three bands; its treatment above 0.72, the
region whose instruction is *report as exceeding and do not credit until the script-pair follow-up has
been run*. Miami applied that instruction as written. Population proof holds again and its signature is
the right one: `position_lookup_auc` / `amplitude_auc` byte-identical **within** a seed (0.7279 / 0.5133)
and different **between** seeds — the pair manifest is seeded per run, the 48 people are fixed. Three of
four arms at or within two epochs of the cap: "both arms budget-limited" is a standing qualifier on the
delta, not a footnote.

**The follow-up, registered now.** The live alternative to "the model learned how these people move"
is "the model learned which activities these people did": with 1–8 sequences per participant across 20
scripts, a random negative pair (two people) is also, most of the time, two *different scripts*, so an
activity cue makes negatives easy and inflates AUC through them. The test removes that cue by
construction: on the same 48 held-out users and the same treatment checkpoints, score **positives only
across different scripts** (same person, script A vs script B) and **negatives only within the same
script** (two people doing the same script), balanced. If the model reads activity, this AUC collapses
toward the control's; if it reads the person's motion, it survives.

| quantity | **band** | **falsifier** | **landing between them means** |
|---|---|---|---|
| treatment AUC, cross-script positives / same-script negatives, per seed | **≥ 0.65** — the gain is motion, the treatment is *credited* at its constrained figure | **< 0.58** — the gain was the activity mix; the treatment is **not** credited and the negative becomes the headline | 0.58–0.65: partly activity mix; report both figures, credit only the constrained one |
| control AUC under the same protocol | 0.50–0.56 | > 0.60 | 0.56–0.60: the zero-shot model reads something script-specific; say so before reading the treatment |
| standard-protocol reproduction (the gate) | each row's `selected_test_auc` within 1e-3 on CPU | outside | outside: the harness feeds the checkpoint something else; nothing from it is read |

Which outcome is strong: the falsifier — an activity-mix reading would void two seeds of an above-band
result and is the one that most needs reporting. Harness: `docs/acceptance/nymeria_script_pair.py`,
built on `score_nymeria.py`'s gate (reproduce the recorded figure on the recorded users first), script
labels from HuggingFace `dataset_metadata.json` joined on `<participant>/act<N>`, run on AVALON (CPU)
against the checkpoints copied from Miami, all six rows once seed 3 lands. Reported per seed, whichever
way it falls.

## Amendment 11 — 2026-09-21 13:46, script-pair harness gated; the control landed in a region Amendment 10 did not name — named here BEFORE the treatment's figure is read

**Gate:** seed 1 control, recorded 0.541537, rescored on CPU **0.541610, gap 7.4e-5** (tolerance 1e-3 = the
documented CPU-vs-GPU band, up to 7e-4 recorded; not slack); recorded-position lookup 0.7233 and
amplitude 0.5081 reproduce to the digit. Harness `docs/acceptance/nymeria_script_pair.py`, run under
`gated_launch.sh` (peak 1.4 GB), 48 users / 47,796 windows, every window carries a script label, no user
skipped (every held-out user has ≥ 2 scripts).

**Seed 1 control under the constrained protocol: 0.4729** on 12,288 + 12,288 pairs; the unconstrained
shape on the same embeddings reads 0.5369 (beside the row's 0.5415, the manifest draw apart).
**Below the 0.50–0.56 band, and Amendment 10 named > 0.60 and 0.56–0.60 but nothing below 0.50** —
the partition defect this file records for the third time, at the bottom end this time, mine. Named
now, with the treatment's figure not yet computed (its run is launching as this is written; the commit
timestamp orders them):

| control constrained AUC | **means** |
|---|---|
| **< 0.50** | the zero-shot embedding's small residual signal on Nymeria is *activity*, not person: two people doing the same script look more alike to it than one person across two scripts, so removing the activity cue takes it below chance. The 0.53–0.55 zero-shot rows were therefore partly activity. **Consequence for reading the treatment: its constrained figure is read on the registered absolute lines (≥ 0.65 credited, < 0.58 activity mix), not as a delta from a control that this protocol inverts.** |

The treatment's registered lines are unchanged.

## Amendment 12 — 2026-09-21 13:51, script-pair follow-up on seeds 1–2: BOTH treatments CREDITED

All four gates pass on CPU (gaps 7.4e-5, 8.8e-4, 2.5e-4, 4.0e-5, tolerance 1e-3). 48 users, 47,796 windows,
12,288 + 12,288 constrained pairs per checkpoint, no user skipped. `docs/acceptance/nymeria_script_pair.json`.

| seed | arm | row AUC | unconstrained, same embeddings | **constrained: cross-script pos / same-script neg** | verdict |
|---|---|---|---|---|---|
| 1 | control | 0.5415 | 0.5369 | **0.4729** | < 0.50: activity-reversed (Amendment 11) |
| 1 | treatment | 0.7082 | 0.7122 | **0.6622** | **≥ 0.65 — credited: motion** |
| 2 | control | 0.5269 | 0.5268 | **0.4660** | < 0.50: activity-reversed |
| 2 | treatment | 0.7263 | 0.7215 | **0.6787** | **≥ 0.65 — credited: motion** |

**Reading.** With the activity cue removed by construction — every positive is one person across two
different scripts, every negative is two people doing the same script — the treatment still separates
unseen Nymeria users at **0.66–0.68**, against a zero-shot model that reads **0.47** on the same pairs.
The activity mix was worth about **0.04–0.05** of the row figures (unconstrained minus constrained on the
same embeddings), so the row figures are credited at their constrained values, not at 0.71/0.73, and
seed 2's above-band 0.7263 is now *exceeding, credited at 0.679*. **The gain is the person's motion, under
`dyn`, on real AR glasses, across activities, on people never seen** — the sentence the user's question
asked for, on two seeds, with the alternative excluded by measurement rather than argued away.

**The control's inversion is a finding about the zero-shot rows.** Both controls fall below 0.50 under
the constrained protocol: the zero-shot embedding's residual signal on Nymeria is activity, not person,
and the 0.53–0.55 zero-shot figures this project has carried for Nymeria were partly that. Reported as
such; it does not change any zero-shot conclusion (all were "not resolved above 0.55") but it names what
the residual was.

Seed 3's pair runs through the same harness the moment its files arrive. Caveats unchanged: two seeds,
no interval; both arms budget-limited (240-epoch pair registered); one sitting per participant.

*Addendum to Amendment 12 (Miami, same hour).* At 12,288 + 12,288 pairs the Hanley–McNeil standard
error near chance is ~0.0037, so the controls sit **8.2 and 9.3 SE below 0.50** — real, and not a
harness fault. The sign carries the mechanism: below chance means the zero-shot embedding scores a
same-script pair from two different people as **more** similar than a cross-script pair from one
person. That is an active activity cue pointing the wrong way for identification once the script is
matched, not merely an absent person cue — so the sentence to write is *"the zero-shot residual on
Nymeria is anti-identifying once activity is matched"*, not *"0.53–0.55 was partly activity"*, which
invites *"so it was partly person too"*. With one caution against the tempting corollary: it does **not**
say the zero-shot model carries no person signal, only that under this pairing the activity cue is
stronger than whatever person signal remains. The treatment figure to quote is the constrained one
(0.662 / 0.679), never 0.7263. And the harness bug's lesson, recorded because it is the Rack seed-1
shape again: **a gate is evidence about the path it exercises — run the cheapest instance of every
path once before the expensive one, not only the cheapest instance overall.**

## Amendment 13 — 2026-09-21 16:35, three seeds complete; the script-pair follow-up credits all three

Rows at c8210b9 (`miami-server`). All six gates pass on CPU (gaps 4.0e-5 to 9.5e-4, tolerance 1e-3).
`position_lookup_auc` / `amplitude_auc` byte-identical across arms within every seed.

| seed | control (row) | treatment (row) | paired | control **constrained** | treatment **constrained** | verdict |
|---|---|---|---|---|---|---|
| 1 | 0.5415 | 0.7082 | +0.1667 | 0.4729 | **0.6622** | credited |
| 2 | 0.5269 | 0.7263 | +0.1994 | 0.4660 | **0.6787** | credited |
| 3 | 0.5386 | 0.7177 | +0.1791 | 0.4762 | **0.6658** | credited |
| **mean** | 0.5357 | 0.7174 | **+0.1817**, sd 0.0165, **95 % CI [+0.1407, +0.2227]** | 0.4717 | **0.6689** | **+0.1972**, sd 0.0134, **CI [+0.1639, +0.2306]** |

**Against the registration, by where the intervals fall.** Falsifier (CI upper < +0.02): nowhere near.
Registered band +0.05 to +0.15: the interval lies almost entirely above it; its top edge (0.15) sits just
inside the interval's lower end (0.1407), so this is **not written as "band excluded"** — it is "the delta
exceeds the registered band, with the band's edge inside the interval". All three controls inside
0.52–0.57, so the above-band region's instruction (check the control first) is discharged three times.
Treatment rows inside 0.60–0.72 on seeds 1 and 3, above on seed 2 — all three **credited at their
constrained figures** (0.662 / 0.679 / 0.666, mean **0.669**), which clear the registered 0.65 line on
every seed. Activity share (unconstrained minus constrained on the same embeddings): 0.050 / 0.043 /
0.047. The controls read 8–9 SE below chance on every seed under the constrained protocol.

**The sentence this arm supports:** *training on Nymeria lifts verification of never-seen Nymeria users
under `dyn` from 0.536 to 0.717 (paired +0.18, CI +0.14 to +0.22, three seeds); about 0.05 of that is
which activities the person did, and the remainder — 0.669 with every positive across two scripts and
every negative within one — is how the person moves, on real AR glasses, across daily-life activities.*
Quote 0.669 as the capability figure, never 0.717 or 0.726.

**Power, recorded because the design was better than it needed to be:** paired sd 0.0165 (rows) and
0.0134 (constrained) at n = 3 give MDDs of ~0.024 and ~0.033; the effect is 6–8× the MDD. The interval is
wide relative to the band because of the effect's size, not under-powering.

**Standing caveats:** five of six 120-epoch runs selected epoch 116–120 (the 240-epoch seed-1 pair of
Amendment 9 is running on Miami, configs differing from the seed-1 pair by exactly `epochs`); one
sitting per participant, so no cross-session cost is paid; the treatment selects its epoch partly on
Nymeria validation users. The logger identity step (Amendment 8) follows the 240-epoch pair.

## Amendment 14 — 2026-09-21, the 240-epoch pair: censoring resolved, delta holds; e240 constrained figure registered before its checkpoints arrive

Rows at c272f81. Seed 1, both arms, `epochs=240`, `patience=15`, configs differing from the 120-epoch
pair by exactly one line:

| | 120-epoch | **240-epoch** | shift | Amendment 9 line |
|---|---|---|---|---|
| control | 0.5415, best 120 of 120 | **0.5405**, best 141, stopped at 156 | −0.0011 | +0.00..+0.01 — **marginally outside**, reported as such; a thousandth is run-to-run noise here |
| treatment | 0.7082, best 118 of 120 | **0.7304**, best 212, stopped at 227 | +0.0222 | +0.00..+0.03 — inside |
| paired delta | +0.1667 | **+0.1900** | +0.0233 | within ±0.03 — inside |
| falsifier (delta shrinks > 0.05 or control > 0.57) | | | | nowhere near: the delta grew |

**Both arms stopped on patience** (156 − 141 = 227 − 212 = 15, the CLAUDE.md recovery when the config
field is blank), so **neither is budget-limited and the 120-epoch figures were not an artefact of where
training stopped.** The informative detail is which arm used the room: the extra budget was worth ~0 to a
model with no Nymeria in training and +0.022 to one with it — a second, independent signature of the
same effect. Peaks 9.9 / 11.1 GB under the cap.

**Registered now, before the two e240 checkpoints reach AVALON:** the script-pair protocol on them.
Treatment constrained **band 0.66–0.72** (at or above the three 120-epoch figures, 0.662–0.679);
**falsifier < 0.62** (the extra epochs bought activity, not motion); between 0.62–0.66: the gain was
partly activity, credit at the constrained figure. Control constrained < 0.50 as on every seed; > 0.56
would be new and reported.

**Record defect for the identity step, found by Miami:** `epochs` reads None on every row of this arm,
so the budget is recoverable only as `epochs_run − best_epoch`; **select the 240-epoch pair on
`epochs_run > 120`, never on `epochs`.** The identity step now covers four items: `num_train_identities`
(computed, dropped by the logger), `eval_split` (never wired — a compact digest and counts go into the row,
the lists stay in the checkpoint), `epochs` (None), and the dead `elif` in `train.py`. Acceptance: control
s1 reproduces digit-identical under the new identity on Miami with the four fields populated on the row.

## Amendment 15 — 2026-09-21 ~21:30, the logger identity step: `03ea8e2376` → `af7cf72022`

Logging only, one step, after seed 3 and the 240-epoch pair, as Amendment 8 decided. `results_log.py`
now copies `num_train_identities` out of history (it was in `FIELDS` and nowhere else), records `epochs`
and `early_stopping_patience` from the config (they read None on every row of this arm), and writes
`eval_split_digest` — 12 hex characters over the split's `<corpus>/<user>` names and flags, machine-
independent, so two nodes holding one split carry one digest while the lists stay in the checkpoint.
The dead `elif` in `train.py` and its wrong comment are gone. A unit test asserts the **values arrive on
a written row** (Miami's point: not that the lines exist) and that the digest is invariant to the
absolute root and sensitive to one swapped user. Suite: 499 passed. **Acceptance, on Miami:** control s1
at 120 epochs under `af7cf72022` reproduces `0.541536678870519` digit-identical on the same device, and
the row shows `num_train_identities` 3072, `epochs` 120, `early_stopping_patience` 15, a 12-character
`eval_split_digest`, `num_drop_users` 188, `num_excluded_users` 48 — the four fields populated and
non-null, said which. Until that lands, rows under `af7cf72022` are not compared with rows under
`03ea8e2376`.
