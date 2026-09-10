# The unseen-users guard: dormant for a week, then verified in both directions — 2026-09-10

`assert_evaluation_users_are_unseen` (model/dataset.py) exists to refuse evaluating a model
on identities it trained on. From its introduction in `9204036` (2026-09-03) to `4be7187`
(2026-09-10) it **never once ran on real data**: it read `user_dirs` off
`SiameseDataset.sample_index`, a `SampleIndex`, which copies a curated subset of
`SampleDataset`'s attributes (sample_time, sample_rate, seq_len, num_users, num_channels,
channels, dataset_names, user_dataset_ids) and never carried `user_dirs`. The early return
`if not train_users or not test_users: return 0` is commented `# nothing recorded; nothing
to check`, so the failure path reads as a deliberate benign branch. Its seven tests built a
`SimpleNamespace(sample_index=SimpleNamespace(user_dirs=[...]))` — a fixture with a
property the subject lacks — and passed.

**Blast radius, from code and history, nothing re-run.** One caller:
`create_dataloader_from_path`'s training branch, reached by every `mode=train` and
`mode=sweep` loader build that uses `test_dirs` or `test_on_excluded=true` (the random pair
split skips it by design). No other reader of `user_dirs` anywhere in the tree. On every
such row since 2026-09-03, "evaluation users were unseen" rests on configuration —
different corpora in `data_dirs` and `test_dirs`, or `build_folds` exclude lists — and on
nothing else. **Unverified, not false**: no wrong configuration has been found, and no
re-run is warranted. The configuration the guard exists for — train and test drawn from the
*same* corpus separated only by user list — first occurs in this project with the matched
arm C2, and the fix landed in the identity step immediately before it. Dormant cost is
zero right up until it is not.

## Direction 1 — it blocks when it should (real object)

Probe on the fixtures corpus, both users in both sets, before the fix:

```
user_dirs seen by the guard: [] []
loaded users: 2 2
GUARD DID NOT FIRE on 2 fully overlapping users (returned 0)
```

After the fix (`SampleIndex.user_dirs = list(sample_dataset.user_dirs)`): `ValueError:
2 of 2 evaluation users were also trained on`. Pinned by
`test_guard_fires_on_a_real_overlapping_dataset`.

## Direction 2 — it passes valid work (the two real shapes, 2026-09-10, this node)

A guard that has never run has never been shown not to block valid work, and it was about
to run on every loader build on three machines. Two real loader builds, no training,
`docs/acceptance/guard_both_directions.py`:

| shape | guard returned | train / val / test users | directories seen |
| --- | --- | --- | --- |
| `sweep.folds`: who_is_alyx, `build_folds` fold 0 of 5, `test_on_excluded=true` | **0** | 46 / 15 / 15 (fold list 15) | train 46, test 15 |
| cross-corpus: `data_dirs` BOXRR-23 + alyx, `test_dirs` CrossApplicationXR, `exclude_users` 32-48, `test_on_excluded=true` | **0** | 3072 / 1024 / **17** | train 3072 |

The load-bearing detail is a **zero from non-empty sets**: the guard saw 46 and 15 real
directories and found no overlap, which is a different claim from the empty-set zero it
returned for a week. The fold build carries the real weight - same corpus both sides,
separated only by user list. **The cross-corpus shape cannot fail by construction**
(BOXRR+alyx and Across-XR share no directory), so its zero proves the sets are populated
and the paths resolve, and nothing about overlap detection. And the one configuration in
which train and test users genuinely overlap - the random pair split (`test_dirs` empty,
`test_on_excluded=false`) - **skips the guard by design**, because that is a deliberate
seen-user protocol; a reader finding the guard silent there is seeing intent, not a second
dormancy.

The second shape is the zero-shot instrument of `across_xr_alignment_REGISTERED.md`: 3072
training identities is the 9.14 split to the identity, and 17 is Schach's test split. Paths
on this node pass through a symlink (worktree `processed_datasets` -> the main checkout's)
and the resolved-path comparison produced no spurious overlap. Pinned for the fixture
corpus by `test_guard_passes_on_real_disjoint_datasets`.

## Direction 2, the shape the fix exists for — one corpus, three disjoint lists

The configuration this project had never run, and the reason the guard stopped being free:
train, validate and test drawn from the *same* corpus, separated only by user lists, through
the `validation_users` path that is new in the same identity step. Loader builds only,
`docs/acceptance/guard_matched_shapes.py`, asserting on the directories the loaders hold:

| arm | guard returned | train / val / test users | directories seen | Across-XR ids held |
| --- | --- | --- | --- | --- |
| C1: Across-XR alone | **0** | 23 / 9 / 17 | 23 / 9 / 17 | train 0-22, val 23-31, test 32-48 |
| C2: BOXRR + alyx + Across-XR | **0** | 3095 / 1033 / 17 | 3095 / 1033 / 17 | train 0-22, val 23-31, test 32-48 |

C2's 3095 = 3072 + 23 and 1033 = 1024 + 9: the 25% draw applied to BOXRR and alyx and left
Across-XR alone, as `select_validation_users` specifies. No list intersects another.

## The rule

The fixture and the subject must agree on the property under test. The pandas fixture
earlier the same day *lacked* a property the real object had (an int frame where the real
one is float, so the multi-block write never triggered); this one *had* a property the real
object lacked. Neither direction is caught by "the tests pass".
