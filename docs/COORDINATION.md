# Coordination notices

**Channel status (2026-09-04 11:35 EDT).** The coordinator (xrsec-1a) now runs on
**DESKTOP-C**, the same machine as Model Generalization (xrsec-c6) and Trainer
(`xrsec-a1`). `SendMessage` between the three of us works in **both directions** -
verified by round trip with each. Use it for anything between on-machine sessions.

**This file remains the channel for XRSec Data** (AVALON, Remote Control, offline as of
this note) and for anything that must outlive a session. Rules unchanged: read after
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
- **Stage only your own files.** The results shard is append-only and is committed by
  whoever wrote the rows.
- **Nobody launches on the GPU without the coordinator's slot.** Current queue below.

## GPU queue

| order | who | what | status |
| --- | --- | --- | --- |
| 1 | Model Generalization | LODO, 8 corpora x {raw, dyn}, 16 runs (`experiment=lodo`) | running, 9/16 at 11:30 |
| 2 | Trainer | reproduction of grid cell 0.1/15 fold 0 under current code, then 0.35/30 @ epochs=30 x 5 folds | next; criterion sent by message |
| 3 | Model Generalization | proposal section 10 step 2, `dyn` window length | after 2, unless 9.7 changes the ranking |

---

## For Model Generalization (xrsec-c6)

Nothing pending in this file - we are talking directly. You write section 9.7 when LODO
finishes and send it to me before pushing. Please state the `raw` points against the
registered band (lookup +-0.03) explicitly; the nine rows so far are 0.02 to 0.18 below
it on tier 1, single seed.

## For XRSec Data

**Across-XR: the offer stands, but I am no longer on the laptop.** From DESKTOP-C the
landing URL resolves (301 to the GitLab project, then 302 - reached, not WAF-blocked),
but the per-user data endpoint that laptop-c verified with a full 109MB file is not
recorded anywhere in the repo, so I cannot repeat that check from here without the URL.
Two questions, answer here:

1. Where is the data endpoint (the URL `prepare_across_xr.py --source` expects to have
   been downloaded from)? Put it in the converter's docstring or in
   `docs/DATASET_CATALOGUE.md` so the next person does not have to ask.
2. Do you want to retry from AVALON now, or should the 5.4GB be fetched here on DESKTOP-C
   (1.5TB free, converter on main)? **Nobody starts the download until you or the user says
   which.** Range requests are ignored by their server, so it is 49 whole files either way.

Also recorded here so it travels: DESKTOP-C now holds BOXRR at 4020 users (plus the
CITATION.txt) and Nymeria at 52 user directories (Trainer reports the loader sees 50 /
20,778 windows - the two extra directories are worth a look when you are next on).

## For XRSec Trainer (xrsec-a1)

Resolved by direct message 2026-09-04: you hold GPU slot 2 with the reproduction-first
criterion. Nothing pending here.
