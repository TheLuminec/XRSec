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

**Nymeria, lookup first (added 12:20, from Trainer via Data).** "The static lookup sits at
chance on Nymeria" is a prediction from the per-recording SLAM-origin argument, not a
measurement. It is measured in the step 3 harness before any model is scored on Nymeria:
(1) `audit_frames.py` row for Nymeria (position not unit-norm, local +Y at world up ~0.95,
mean |q| 1.0000, else stop); (2) lookup AUC on Nymeria pairs, three seeds, random control,
plus same-participant vs different-participant mean-position distance. Criterion: lookup
within 0.50 +-0.02 and same-participant distance no smaller than different-participant.
At or above 0.55 the premise is wrong and CLAUDE.md's "clean instrument" paragraph is
retracted before anything else is built on it. Every Nymeria number carries the caveat:
cross-activity positives within one sitting, so it cannot pay the cross-session cost.
Nymeria is never in training.

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
