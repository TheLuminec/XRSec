# Coordination notices

**Compacted 2026-10-12.** Everything before this date, resolved or not, is archived verbatim:
- `docs/archive/COORDINATION_through_2026-10-12.md` (this file as it stood);
- `docs/archive/COORDINATION_ARCHIVE_to_2026-09.md` (the earlier archive).

This file holds only **standing rules and open items**. Append under your own dated heading, and delete an
item when it resolves: its conclusion goes into CLAUDE.md or into its registration.

## Roster

| session | machine | role |
| --- | --- | --- |
| XRSec Coordinator | AVALON | registrations, reading, propagation to main, briefs |
| XRSec Miami Server | feng-MS-7B51 (RTX 4060 Ti) | all GPU work. Pushes to `origin/miami-server`, never to main |
| XRSec Data | AVALON | acquisitions and conversions, only when authorised by the user |
| paper-synthesis session | AVALON, branch `worktree-paper-theses` | `docs/PAPER_DRAFT_C.md`. The Coordinator fast-forward merges |

Address peers by bare name. Re-run `ListAgents` when a send fails.

## Standing rules

1. **Miami.**
   - Every job runs as `avm run "<name>" -- env XRSEC_MARKER_DIR=... bash gated_launch.sh run <job> -- <cmd>`.
   - Run fixtures on the node before a job.
   - No `git pull` while any xrsec scope is live.
   - Don't push manually while a chain can launch. A checkout of the miami-server tip once removed a
     module mid-launch (2026-10-12).
2. **Results flow.**
   - Miami pushes each artefact as it lands, then a sha256 sweep against origin.
   - The Coordinator propagates the files to main *before* reading, commits the read output before
     interpreting it, and records the result in the registration.
3. **Gates.**
   - Checkpoint gate: 1e-4 on cuda, 1e-3 on CPU.
   - Plumbing gates are same-device as the committed figures.
   - A refusal is pushed as written and never re-scored elsewhere to get through.
4. **Transfers** of corpora or checkpoints need the user's authorisation, recorded here.
   - BOXRR-derived material moves only between AVALON and Miami (same institution).

## Open items

### The Who-is-Alyx / Across-XR overlap (opened 2026-10-12)

- **Registration:** `docs/acceptance/alyx_axr_overlap_REGISTERED.md`, with Amendment 1 on code identity.
- **Retrain done (Result 2026-10-13):** the figures depend on Who-is-Alyx (zero-shot 0.234 → 0.168, C2-lo
  0.375 → 0.323), and participant overlap is not excluded. The authors' answer now decides which figures
  the paper quotes.
- **Not yet done:** the Schach paired comparison for the no-alyx C2-lo arm (descriptive; no GPU needed).
- **For the user to send (no session sends it), now decision-relevant:**

> Dear Dr. Schach and Dr. Rack, we use both the Across XR Applications dataset and the Who Is Alyx dataset
> in our work on user identification with unseen users, and want to make sure no participant appears on
> both sides of our splits. Both were recruited through the University of Würzburg's participant
> system. Could you tell us whether any Across-XR participant also took part in the Who Is Alyx study, and
> if so which Across-XR user IDs (we care most about users 32-48, your test split)? A yes or no for the
> test users would already help. Thank you for releasing both datasets and the code.

### Dr. Feng's follow-ups not yet started (email 2026-10-12)

- **Point 2, the evaluation write-up for Dr. Feng.** The protocol and first results exist in
  `template_verification_PROTOCOL.md`; the user asked for the write-up later.
- **Not yet run:**
  - a version of our model without the large pretraining set, on Nymeria (on Across-XR, C1 at 0.131
    already serves);
  - FFT features, judged on non-rhythm application pairs;
  - GNN fusion, which must beat simple fusion;
  - rank-1 for the leave-script-out arm;
  - a Nymeria adapter for template verification.
- **Meeting slides:** a model diagram plus a settings table, not yet made.

### Miami data volume

ntfs3 damage: `runs/miami-alignment/MANIFEST.sha256` and an old shard copy do not open. Checkpoint gates
are the integrity check in use. No repair is attempted without the user; it is raised with them.
