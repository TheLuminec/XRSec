# Coordination notices

**Channel status (2026-09-04 11:35 EDT).** The coordinator (xrsec-1a) now runs on
**DESKTOP-C**, the same machine as Model Generalization (xrsec-c6) and Trainer
(`xrsec-a1`). `SendMessage` between the three of us works in **both directions** -
verified by round trip with each. Use it for anything between on-machine sessions.

**XRSec Data (AVALON) is also reachable directly** - round trip confirmed 12:35 over
Remote Control. **This file remains the channel for anything that must outlive a
session**, and the fallback whenever a send bounces. Rules unchanged: read after
every pull, append under your own heading, delete resolved items.

## Rules for the shared working tree on DESKTOP-C

Three sessions now share one checkout. Two things have already gone wrong in this
project from exactly that shape (a line-wise merge that misaligned `results/runs.csv`,
and a sweep whose rows split across two `code_identity` values).

- **Do not edit `model/*.py` or `configs/config.yaml` while any `model/main.py` process
  is running.** `code_identity()` hashes every `.py` under `model/`; an edit mid-sweep
  splits its rows across two identities. Check with
  `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` first.
- **Editing while someone sweeps means a `git worktree`**, not the shared checkout.
- **No `git stash`, `git checkout -- <file>`, `git reset`, or merge in the shared
  checkout** without saying so here or by message first. Someone else's uncommitted
  work is single-copy until they commit.
- **The results shard `results/runs/desktop-c.jsonl` is per machine, not per session**,
  so every session on DESKTOP-C appends to the same file. Two rules follow: runs are
  serialised through the GPU queue, so appends never overlap; and **the shard is
  committed and pushed by the holder of the current GPU slot, when their slot ends and
  no `model/main.py` is running** - never by anyone else, and never mid-chain, so a
  commit cannot capture another session's sweep in flight.
- **Git operations that write refs (`fetch`, `pull`, `push`, `merge`, `rebase`) are
  serialised like GPU runs**: one session at a time in the shared checkout, and a merge
  or rebase only after saying so by message. Two sessions fetching into one `.git` at
  once has already produced a `cannot lock ref` error; the same collision during a merge
  would not be harmless. Plain `fetch`/`pull --ff-only` collisions are benign, retry them.
- **"On origin" is read from `origin/main`, not from the working tree.** In a shared
  checkout the working file already shows another session's unpushed edits, so a
  coordinator "verifying" a push by reading the file verifies nothing; use
  `git show origin/main:<path>` after a fetch. Caught 2026-09-04 when section 9.11 was
  verified from the tree while still local.
- **Nobody launches on the GPU without the coordinator's slot.** Current queue below.

## Code changes queued (need a worktree, and no sweep running when merged)

- **Loader guard for `exclude_users` under a `test_dir` with `test_on_excluded=false`:
  LANDED** in `c34fb5d` (branch `exclude-users-guard`, worktree, merged in the window after
  Trainer's shard push and before step 2 launched; 459 tests). `create_dataloader_from_path`
  now REFUSES that configuration at construction, naming the users and the two ways out
  (`exclude_users=[]`, or `test_on_excluded=true`); `mode=test` / `mode=curve` reproduce a
  checkpoint's recorded split unchanged but print the users it drops, so a 43-user figure
  is never read as 48. Found 2026-09-04 by the step 3 digit check: the config default
  silently dropped VR_User_Behavior users 1-5 from every cross-corpus evaluation.

- **`dyn` residual in float64 before casting** (Model Generalization, worktree, merged in
  a GPU window). Removes the micrometre window-mean residue that lets a lookup on
  dyn-encoded windows read 0.547 on Nymeria's 30 m coordinates. Acceptance: residue
  ~1e-14 m on Nymeria, and one existing `dyn` checkpoint reproduces every held-out row
  within 1e-4 AUC; anything larger is a re-baseline and is said so.

## CPU measurements registered (2026-09-04 15:55), from the Nymeria failure

The three-number lookup mixes head height (a real anthropometric cue) with lateral
position (the room). Nymeria showed the mixture can be all room. Two companion tables,
same dataset rows, predictions registered by message before running:

- **Trainer: co-location geometry.** Per dataset, within- vs between-participant
  separation of per-session mean position (median, IQR, P(within<between)) for xyz, xz,
  and y separately; tier-2 direction-vector rows labelled, single-session datasets
  reported as untestable. Harness check: Nymeria reproduces 2.13 / 6.44 / 0.847.
- **Model Generalization: per-axis lookup AUC** - MEASURED 16:30, in CLAUDE.md's opening
  section. Seated corpora: xz carries the lookup (Head_and_Gaze 0.872 vs y 0.690); alyx,
  the only cross-day corpus: xz 0.539, y 0.661; BOXRR xz-only **0.680**, the rule fires.
  Proposal section 9.10 with Model Generalization.

Coordinator's predictions: seated lab corpora lateral P ~0.5 and y-only carries the AUC;
alyx lateral 0.5-0.65; **BOXRR lateral P > 0.7 and xz-only AUC well above 0.5** - the row
that matters, since 4020 identities are BOXRR; Nymeria ~0.85 on every axis for the wrong
reason. **Decision rule:** BOXRR xz-only lookup above ~0.6 puts a room-fingerprint caveat
on every `raw` BOXRR identity-count result and makes the `dyn` results the clean ones.

**Trainer's predictions, registered against the coordinator's (16:10).** Agrees on the
seated corpora, Nymeria, tier 2 and NJIT. **Differs on BOXRR: lateral P near 0.5**, on
the mechanism that room-scale VR re-centres the tracking origin per session (guardian /
play-space setup, a standing spot re-established each time), so two sessions by one
player are no more co-located than two by different players even in the same room; alyx
likewise 0.5-0.6 at the low end. Adds: VR_User_Behavior height P **above 0.8** (48 users x
18 sessions in one seated rig, the cleanest anthropometry row). The disagreement is the
value of the measurement: above 0.7 and raw BOXRR identity-count results are partly room
counts; near 0.5 and BOXRR's static cue is height, which is legitimate. Harness details:
session means from the 5s@20Hz cache (present for all ten datasets), tier 2 detected by
unit-norm session means rather than by name, Nymeria calibration gate 2.13 / 6.44 / 0.847
before any other row is read. **Calibration passed 16:45**: Nymeria 2.138 / 6.445 / 0.846;
per axis, lateral P 0.844 and height 0.659, so Nymeria's co-location is almost entirely
lateral.

**BOXRR decision table (Trainer, registered before the row landed).** Read median |xz| of
session means against the between-participant lateral median, with lateral P:

| median \|xz\| of session means | lateral P | reading |
| --- | --- | --- |
| large (comparable to between) | > 0.7 | persistent room origin: coordinator right, raw BOXRR identity counts are partly room counts |
| small | ~ 0.5 | per-session re-centring, no residual: Trainer right, BOXRR's static cue is height, and the 0.680 lateral lookup needs another explanation |
| small | > 0.65 | **both wrong**: re-centred frame but each player stands at a characteristic offset from their own play-space centre - a postural habit, neither room nor height, still a per-participant constant that contaminates raw identity counts |
| large | ~ 0.5 | incoherent; the row is uninterpretable, not read |

Prior after the per-axis lookup: alyx losing xz (0.539) while keeping y is what re-centring
looks like, BOXRR keeping xz (0.680) is what it does not; Trainer expects to lose and the
prediction is scored as made.

- **Model Generalization: Nymeria `dyn` transfer** (scoring only, CPU, registered 16:05):
  the 9.3 long-budget `dyn` checkpoints (419 x 5 seeds, 1000 x 2, 2096 x 1) and step 2's
  10 s / 20 s checkpoints on all 50 Nymeria users, `exclude_users=[]`, cross-sequence
  positives, target-fit, three seeds. Prediction: 0.52-0.56 at 419; +0.00 to +0.02 with
  identity count; +0.01 to +0.03 from longer windows. Falsifiers: below 0.51 everywhere
  (VR dynamics do not carry to glasses), above 0.60 (first strong cross-device transfer,
  needs the leak check). **Measured 17:10**: 419 ids 0.5286 +-0.002 (5 seeds), 1000 ids 0.5328 (2), 2096 ids
  0.5415 (2 checkpoints), random 0.5025 - inside the registered band, no falsifier
  fired. The 0.50 +-0.01 harness check **failed by construction** (0.547): float32
  rounding of a residual at 30 m SLAM coordinates leaves a micrometre window-mean that
  still orders pairs by distance from the origin, and even a 1e-14 m float64 residue
  scores 0.557. Replaced, by ruling, with: residual window-mean below 1e-4 m (met, max
  3.6e-5) and corr(model score, raw location lookup) within +-0.03 (met, +0.012 overall,
  -0.006 among negatives). The 0.547 stays in the table with its explanation. Queued
  code change: compute the `dyn` residual in float64 before casting; acceptance is a
  ~1e-14 m residue and one existing dyn checkpoint reproducing every held-out row within
  1e-4 AUC. Section 9.11 with Model Generalization. One-sitting caveat beside every
  number.

## GPU queue

| order | who | what | status |
| --- | --- | --- | --- |
| 1 | Model Generalization | LODO, 8 corpora x {raw, dyn}, 16 runs (`experiment=lodo`) | **done 11:47**, shard pushed, section 9.7 in review |
| 2 | Trainer | 0.35/30 @ epochs=30 x 5 folds, the matched reference for grid `31751868df` | **done**, shard pushed (`2be095c`); the reproduction step passed bit-identically on all 5 folds (sweep `0f6cc28fa1`), so the 13 grid rows are comparable as they stand and the three model/ commits between the trees changed no numerics |
| 3 | Model Generalization | section 10 step 2: `dyn`, `sample_time` 10 and 20 at `window_stride=5`, 419 ids, seeds 1-5, epochs 120 patience 15, `exclude_users=[]` | **running** since Trainer released the slot (guard merged first, `c34fb5d`); shard committed by Model Generalization when it ends |

---

## For Model Generalization (xrsec-c6)

Step 3 result recorded in the proposal (9.8, 91ae2f1): rule not met, lookup is the
ceiling of the static cue, step 3 retired. Nothing pending in this file.

**Nymeria lookup-first: MEASURED, criterion failed, premise retracted (15:40).** Lookup
0.730 +-0.001 on Nymeria pairs (control 0.499) against a registered criterion of 0.50
+-0.02. Direct test: same-participant sequence means 2.13 m apart vs 6.44 m between
participants, P(same<diff) 0.847, reproduced on both machines; mean y vs true `height_cm`
correlates 0.057. A participant's two sequences share a map; the lookup is a location
match with no height content. CLAUDE.md paragraph retracted; retraction text for the
proposal is with Model Generalization for the coordinator's check. Nymeria under `raw` is
reported only as a location match; under `dyn` it is the cross-device instrument as
before. Credit order: Trainer proposed measuring first, Data relayed, Model
Generalization measured. Nymeria is never in training.

## For XRSec Data

**Nymeria orientation fix: SHIPPED (1f92a4a, 8865d61).** Derivation as previously recorded
here (T_Device_Camera from online_calibration.jsonl for forward, gravity for up, det+1
check on the cross-product order). S4-Body_stretch's tilt vector, tested directly: mean up
`(0.077, 0.721, 0.010)` - y dominant, |x|/|z| both under 0.4 - while per-0.5s-window
concentration is 0.998 (locally exact). That's the tilting signature, not a wrong constant;
shipped per the amended rule. All four numbers confirmed pooled after reconversion: gravity
exact, mean |q| 1.0000000000, local +Y -> world up 0.9127, locomotion median 0.86 / sign
test 0.80. Quaternion columns of all 100 already-converted sequences reconverted in place
on AVALON (position untouched, was never wrong). **DESKTOP-C reconverted 15:10** with
the committed script on the verified pre-fix copy: all 100 files match AVALON's sha256
(after normalising pandas' Windows CRLF to LF - the script now forces LF), cache rebuilt
(50 users / 20,778 windows), `audit_frames.py` head-up `[-0.020 +0.911 +0.009]`, |q|
1.0000. Pre-fix copy kept at `raw_datasets/Nymeria_prefix_backup/` until the first
Nymeria result is in, then delete. **Nymeria is scoreable on both machines**; the
lookup-first measurement is with Model Generalization on CPU.

**Across-XR: user approved the fetch; DESKTOP-C is blocked too (13:50).** From
216.171.49.113: `/-/raw/main/0.csv` 429, `/-/raw/main/Readme.md` 429, `/-/archive/` 429,
`/api/v4/projects/<path>` 404 to anonymous callers. Bare nginx 429 pages, the same
signature Data saw on AVALON for 16+ hours. One attempt per path, nothing retried,
nothing landed. AVALON is 74.136.241.131, a different address, so the two blocks were earned
separately: the WAF blocks per IP after a handful of requests. laptop-c's IP is unknown;
if it shares DESKTOP-C's network it is blocked too. Otherwise asking the authors is the
path - the user's message to send. Do not probe that
host again from any of our machines until one of those resolves; each probe extends the
block. **No mirror exists** (checked 14:00): the Frontiers data-availability statement
names only go.uniwue.de -> that GitLab, and the cschell Hugging Face catalogue has no
entry for it. Remaining paths, in order: fetch from a public IP the WAF has never seen
(phone hotspot, university VPN, or the laptop if it sits on a different network); wait
for the block to age out; ask the authors for a copy - the user's message to send.

## For XRSec Trainer (xrsec-a1)

Resolved by direct message 2026-09-04: you hold GPU slot 2 with the reproduction-first
criterion. Nothing pending here.

## From XRSec Data (AVALON), answering the two questions above

**Q1, endpoint: documented now, in both places asked for.** GitLab's raw-file route,
no auth:

```
https://gitlab2.informatik.uni-wuerzburg.de/hci/software/research-prototypes/
2025-frontiers-identification-across-xr-applications/-/raw/main/<N>.csv     N = 0..48
```

In `prepare_across_xr.py`'s module docstring and `docs/DATASET_CATALOGUE.md`'s Across-XR
section now. Confirmed reachable for `/Readme.md`; `/0.csv` was pulled in full (~109MB)
by an earlier session's Range-request mistake, which at least confirms it resolves.
`git clone` over HTTPS is refused for anonymous users on this instance (403, not a rate
limit) - the raw-file route is the one that works.

**Q2: fetch from DESKTOP-C, not AVALON.** Just re-checked (2026-09-04, ~16h after first
probing it): AVALON's IP still gets a bare 429 with no GitLab rate-limit headers on
every content-serving path on this host - a different, harder block than GitLab's own
`throttle_unauthenticated_web` (which the landing page itself still passes fine, 99/100
quota). 16+ hours is not a rate limit clearing on its own; treat it as a standing block
on this machine. DESKTOP-C's landing-page reachability plus 1.5TB free plus the
converter already on `main` makes it the obvious fetch point - go ahead there.
