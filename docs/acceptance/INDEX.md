# Acceptance index (2026-10-12)

One line per programme. Each registration holds its design, amendments and Result section; the reading
output (`*_read*.json`) is what its result rests on.

**Do not move or rename files here.** Harnesses import each other by name, and gates look checkpoints up
by recorded paths. Superseded *documents* live in `docs/archive/acceptance/`; superseded *results* stay
here, because registrations cite them.

| programme | registration | status | key result | reading output |
| --- | --- | --- | --- | --- |
| Across-XR cross-application, alignment, C1/C2/P2/P3 | `across_xr_alignment_REGISTERED.md`, `across_xr_alignment_RESULTS.md` | closed | zero-shot A1 0.234, C2-lo 0.375; alignment closed | `across_xr_alignment_*seed*.json`, `*_aggregate.json` |
| Schach et al. paired, per user | (Amendment 8 of the above) | closed | C2-lo +0.119 [+0.050, +0.192] on their metric | `schach_paired.json`, `schach_per_user.*` |
| Exposure breadth (hold one app out at scale) | `exposure_breadth_REGISTERED.md` | closed | breadth − treatment +0.108; Questset +0.056 | `exposure_breadth_read3.json`, `*_a3_*.json` |
| Questset arms and reverse exposure | `questset_arms_REGISTERED_miami.md`, `questset_exposure_REGISTERED.md` | closed | A1 0.179 / 0.217; reverse +0.014, not resolved | `questset_exposure_read.json` |
| Nymeria in-domain (AR glasses) | `nymeria_in_domain_REGISTERED.md` | closed | constrained AUC 0.669 against control 0.472 | `nymeria_script_pair.json` |
| Nymeria leave-script-out | `nymeria_lso_REGISTERED.md` | closed | LSO − treatment −0.043 [−0.056, −0.030] | (in the registration) |
| Nymeria rank-1 | `nymeria_rank1_REGISTERED.md` | closed | treatment 0.555, control 0.172 (falsifier fired) | `nymeria_rank1_read.json` |
| Nymeria, `paper_gnn_bilstm` | `nymeria_gnn_REGISTERED.md` | closed | GNN worse: −0.146 constrained, −0.246 rank-1 | `nymeria_gnn_read.json` |
| e240 transfer, test-time adaptation | `e240_transfer_REGISTERED.md`, `tta_transfer_REGISTERED.md` | closed | transfer +0.003; centring +0.01-0.03 | `e240_transfer.json`, `tta_transfer*.json` |
| alyx cross-day | `alyx_cross_day_REGISTERED.md` | closed | 0.483 against 0.743 same day | `alyx_cross_day_read_gpu.json` |
| Ball-throwing cross-day and cross-headset | `ballthrowing_cross_day_REGISTERED.md` | closed | C1 0.693, day −0.131, headset −0.235 | `ballthrowing_cross_day_read.json` |
| Broad 2 s (Nymeria null, raw audit, tilt, window) | `broad_2s_REGISTERED.md` | closed | raw session-bound; tilt account not supported | `broad_2s_read.json`, `ballthrowing_tilt_lookup.json` |
| Rack et al. 2023 re-run | `sota_rack2023_reproduction_REGISTERED.md` | closed, **FAIL** | 0.850 / 0.566 / 0.174 against 0.99 / 0.89 / 0.25 | `rack2023_read.json` |
| Template verification (deployment decision) | `template_verification_PROTOCOL.md` | first results in | TAR at 1 % FAR: 0.50 / 0.17 / 0.06 | `tv_*.json`, `docs/results/template_verification.md` |
| Split audit (users + content unseen) | `split_audit_2026-10-12.md` | reference | compliant rows listed | — |
| Who-is-Alyx / Across-XR overlap | `alyx_axr_overlap_REGISTERED.md` | closed, **BAND failed** | alyx contributes: zero-shot −0.066 (0.234 → 0.168), C2-lo −0.052 (0.375 → 0.323) | `alyx_axr_overlap_read.json` |
| Numerics-touching code changes | `README.md` | reference | dyn float64, line endings, amplitude baselines | per-change files |

Everything else here is a harness, a fixture, a gate certificate (`*_gate.json`), a manifest or a list file
belonging to one of these rows.
