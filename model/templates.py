"""
Multi-window template scoring: aggregate k windows per side before comparing.

Verification from a single 5-second window is the hardest possible operating point,
and no real system works that way - it aggregates evidence. This builds a template
from k windows of one session, does the same for the other side, and compares the
two, reporting a curve over k with k=1 reproducing the existing single-window number.

The whole curve costs ONE forward pass. Embeddings for every window are computed
once, and each k is then a matter of indexing and averaging them, so an entire
k-curve can be produced from an already-trained checkpoint with no retraining. That
is what makes this cheap enough to run on every checkpoint we have.

Two rules keep it honest:

- The k windows on a side come from ONE session, and the two sides of a positive pair
  come from DIFFERENT sessions, exactly as `cross_session_positives` requires. A
  template drawn across sessions would average away the session variability that
  cross-session evaluation exists to expose, and would flatter the score.
- Embeddings are L2-normalised before averaging when the model scores by cosine,
  then the mean is renormalised. Averaging unnormalised embeddings lets one
  large-magnitude window dominate the template. For a `diff_linear` head the plain
  mean is used instead, because that head was trained against unnormalised
  embeddings and rescaling them would move the operating point it learned.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from metrics import pair_metrics


def _sessions_for_user(sample_index, user_index: int, minimum: int) -> list[torch.Tensor]:
    """Window indices grouped by session for one user, keeping groups of >= minimum."""
    windows = sample_index.user_sample_indices[user_index]
    if windows.numel() == 0:
        return []

    sessions = getattr(sample_index, "window_session_ids", None)
    if sessions is None or sessions.numel() == 0:
        return [windows] if windows.numel() >= minimum else []

    groups = []
    user_sessions = sessions[windows]
    for session in torch.unique(user_sessions):
        group = windows[user_sessions == session]
        if group.numel() >= minimum:
            groups.append(group)
    return groups


def generate_template_manifest(
    sample_index,
    pairs_per_user: int,
    k: int,
    k_probe: int | None = None,
    match_ratio: float = 0.5,
    seed: int | None = None,
    within_dataset_negatives: bool = True,
):
    """
    Build pairs whose sides are each k window indices drawn from one session.

    Returns {x1_indices, x2_indices, labels, anchor_user_ids} where x1/x2 are
    (pairs, k). Users whose sessions are too short to supply k windows are skipped
    and counted, because a template quietly built from fewer windows than requested
    would make the k-curve report a k it did not use.
    """
    rng = np.random.default_rng(0 if seed is None else int(seed))
    user_dataset_ids = getattr(sample_index, "user_dataset_ids", None) or [0] * sample_index.num_users

    eligible = {}
    # The reference side may carry more windows than the probe side, so a session has
    # to be long enough for whichever side lands on it.
    k_probe = k if k_probe is None else int(k_probe)
    for user in range(sample_index.num_users):
        groups = _sessions_for_user(sample_index, user, minimum=max(k, k_probe))
        if groups:
            eligible[user] = groups

    positives_target = int(round(pairs_per_user * match_ratio))
    negatives_target = pairs_per_user - positives_target

    x1, x2, labels, anchors = [], [], [], []
    skipped_users = sample_index.num_users - len(eligible)
    single_session = 0

    def draw(group: torch.Tensor, size: int) -> list[int]:
        picked = rng.choice(group.numel(), size=size, replace=False)
        return [int(group[int(i)]) for i in picked]

    for user, groups in eligible.items():
        # Positives: two different sessions of the same user.
        if len(groups) >= 2:
            for _ in range(positives_target):
                first, second = rng.choice(len(groups), size=2, replace=False)
                x1.append(draw(groups[int(first)], k))
                x2.append(draw(groups[int(second)], k_probe))
                labels.append(1.0)
                anchors.append(user)
        else:
            single_session += 1

        # Negatives: a different user, same dataset when required.
        partners = [
            other for other in eligible
            if other != user
            and (not within_dataset_negatives or user_dataset_ids[other] == user_dataset_ids[user])
        ]
        if partners:
            for _ in range(negatives_target):
                partner = int(rng.choice(partners))
                x1.append(draw(groups[int(rng.choice(len(groups)))], k))
                partner_groups = eligible[partner]
                x2.append(draw(partner_groups[int(rng.choice(len(partner_groups)))], k_probe))
                labels.append(0.0)
                anchors.append(user)

    if not labels:
        return {"x1_indices": torch.empty((0, k), dtype=torch.long),
                "x2_indices": torch.empty((0, k_probe), dtype=torch.long),
                "labels": torch.empty(0), "anchor_user_ids": torch.empty(0, dtype=torch.long),
                "skipped_users": skipped_users, "single_session_users": single_session}

    manifest = {
        "x1_indices": torch.tensor(x1, dtype=torch.long),
        "x2_indices": torch.tensor(x2, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "anchor_user_ids": torch.tensor(anchors, dtype=torch.long),
        "skipped_users": skipped_users,
        "single_session_users": single_session,
    }

    # Same requirement as the single-window path: accuracy is read at a fixed
    # threshold, so the set has to carry the balance that was asked for.
    positives = int((manifest["labels"] > 0.5).sum())
    negatives = int(manifest["labels"].numel() - positives)
    if positives and negatives:
        keep_positive = min(positives, int(negatives * match_ratio / (1.0 - match_ratio)))
        keep_negative = min(negatives, int(round(keep_positive * (1.0 - match_ratio) / match_ratio)))
        positive_idx = torch.nonzero(manifest["labels"] > 0.5, as_tuple=False).view(-1)
        negative_idx = torch.nonzero(manifest["labels"] <= 0.5, as_tuple=False).view(-1)
        chosen = torch.cat([
            positive_idx[torch.as_tensor(rng.permutation(positives)[:keep_positive], dtype=torch.long)],
            negative_idx[torch.as_tensor(rng.permutation(negatives)[:keep_negative], dtype=torch.long)],
        ])
        for key in ("x1_indices", "x2_indices", "labels", "anchor_user_ids"):
            manifest[key] = manifest[key][chosen]

    return manifest


@torch.no_grad()
def embed_all(model, sample_index, device, batch_size: int = 512) -> torch.Tensor:
    """Every window's embedding, once. This is the only forward pass a curve needs."""
    model.eval()
    chunks = []
    for start in range(0, sample_index.sample_count, batch_size):
        batch = sample_index.samples[start:start + batch_size].to(device)
        chunks.append(model.embed(batch).detach().cpu())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


@torch.no_grad()
def score_templates(model, embeddings: torch.Tensor, manifest, device) -> torch.Tensor:
    """
    Template scores for one manifest, from precomputed embeddings.

    Uses the model's own head so the scoring rule matches how it was trained: a
    cosine head stays a cosine comparison, a diff_linear head is applied to the
    averaged embeddings it was fitted against.
    """
    if manifest["labels"].numel() == 0:
        return torch.empty(0)

    normalise = getattr(model, "head", "diff_linear") == "cosine"

    def template(indices: torch.Tensor) -> torch.Tensor:
        vectors = embeddings[indices]                      # (pairs, k, dim)
        if normalise:
            vectors = F.normalize(vectors, dim=2)
        pooled = vectors.mean(dim=1)
        return F.normalize(pooled, dim=1) if normalise else pooled

    left = template(manifest["x1_indices"]).to(device)
    right = template(manifest["x2_indices"]).to(device)
    return model.score(left, right).view(-1).detach().cpu()




def _ranks_within(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Rank of the correct user per probe, ties rank-averaged.

    Same convention as roc_auc and for the same reason: an untrained model emits a
    near-constant score, and breaking those ties by sort order would report either rank
    1 or rank N for what is really no information at all. Averaged, a constant scorer
    lands at (N+1)/2.
    """
    correct = scores[torch.arange(scores.shape[0]), labels]
    greater = (scores > correct[:, None]).sum(dim=1)
    tied = (scores == correct[:, None]).sum(dim=1)          # includes the correct one
    return greater.float() + (tied.float() + 1.0) / 2.0


@torch.no_grad()
def cmc_curve(model, embeddings: torch.Tensor, sample_index, device,
              gallery_k: int = 8, probe_k: int = 1, probes_per_user: int = 16,
              seed: int | None = None, batch_size: int = 256,
              gallery_sizes: tuple[int, ...] = (), subsets: int = 20) -> dict:
    """
    Closed-set identification: rank every held-out user against a probe.

    This project reports *verification* - given two windows, same person or not, a
    two-class decision at chance 0.50. Most of the XR biometrics literature reports
    *identification* - given a probe, rank a gallery of N enrolled users and check
    whether the right one comes first, at chance 1/N. Those are different tasks, and a
    verification accuracy cannot be compared against a published rank-1 figure. This
    computes the second so the comparison can actually be made.

    Protocol matches the rest of the pipeline: the gallery template for each user comes
    from one session and the probes from a different one, so a correct match cannot be
    session matching. Users with a single session fall back to disjoint windows of the
    same session and are counted, exactly as cross-session pairing does.

    Chance is 1/N and N is the number of enrolled users, so rank-1 is only meaningful
    beside the gallery size - both are returned, and a rank-1 quoted without its N says
    nothing.

    `gallery_sizes` additionally reports rank-1 restricted to a random subset of N
    enrolled users, averaged over `subsets` draws. That is what makes an external
    result comparable: a paper reporting 78.5% rank-1 over 17 unseen users has not
    beaten a lower number measured over 419, because ranking against 17 candidates is a
    different problem. Scoring is done once and the subsets are column selections on
    the result, so this costs almost nothing.
    """
    rng = np.random.default_rng(seed)
    normalise = getattr(model, "head", "diff_linear") == "cosine"

    def template(indices: torch.Tensor) -> torch.Tensor:
        vectors = embeddings[indices]
        if normalise:
            vectors = F.normalize(vectors, dim=1)
        pooled = vectors.mean(dim=0)
        return F.normalize(pooled, dim=0) if normalise else pooled

    gallery_vectors, probe_vectors, probe_labels = [], [], []
    same_session_fallback = 0
    skipped = 0

    for user_index in range(sample_index.num_users):
        windows = sample_index.user_sample_indices[user_index]
        if windows.numel() < gallery_k + probe_k:
            skipped += 1
            continue

        groups = _sessions_for_user(sample_index, user_index, minimum=1)
        usable = [g for g in groups if g.numel() >= max(gallery_k, probe_k)]

        if len(usable) >= 2:
            first, second = rng.choice(len(usable), size=2, replace=False)
            gallery_pool, probe_pool = usable[int(first)], usable[int(second)]
        else:
            # One session: split its windows so gallery and probe stay disjoint. The
            # session shortcut is live for this user, which is why it is counted.
            same_session_fallback += 1
            shuffled = torch.as_tensor(rng.permutation(windows.numpy()))
            gallery_pool = shuffled[:gallery_k]
            probe_pool = shuffled[gallery_k:]
            if probe_pool.numel() < probe_k:
                skipped += 1
                same_session_fallback -= 1
                continue

        label = len(gallery_vectors)
        gallery_vectors.append(template(
            torch.as_tensor(rng.choice(gallery_pool.numpy(), size=gallery_k, replace=False))))

        for _ in range(probes_per_user):
            picked = rng.choice(probe_pool.numpy(), size=probe_k,
                                replace=probe_pool.numel() < probe_k)
            probe_vectors.append(template(torch.as_tensor(picked)))
            probe_labels.append(label)

    users = len(gallery_vectors)
    if users < 2 or not probe_vectors:
        return {"users": users, "probes": 0, "rank1": float("nan"), "chance": float("nan"),
                "same_session_fallback_users": same_session_fallback, "skipped_users": skipped,
                "cmc": []}

    gallery = torch.stack(gallery_vectors).to(device)
    probes = torch.stack(probe_vectors)
    labels = torch.tensor(probe_labels, dtype=torch.long)

    # Scored through the model's own head, so a cosine model is compared by cosine and
    # a diff_linear model by the layer it was actually trained with.
    chunks = []
    for start in range(0, probes.shape[0], batch_size):
        chunk = probes[start:start + batch_size].to(device)
        rows = chunk.shape[0]
        left = chunk.repeat_interleave(users, dim=0)
        right = gallery.repeat(rows, 1)
        chunks.append(model.score(left, right).view(rows, users).detach().cpu())
    all_scores = torch.cat(chunks)

    rank = _ranks_within(all_scores, labels)
    cmc = [float((rank <= k).float().mean()) for k in range(1, users + 1)]

    matched = {}
    for size in sorted({int(n) for n in gallery_sizes if 1 < int(n) <= users}):
        scores_at_size = []
        for draw in range(subsets):
            chosen = np.sort(rng.choice(users, size=size, replace=False))
            keep = torch.from_numpy(np.isin(labels.numpy(), chosen))
            if not keep.any():
                continue
            # Relabel into the subset's own index space before ranking.
            remap = {int(user): position for position, user in enumerate(chosen)}
            subset_labels = torch.tensor(
                [remap[int(v)] for v in labels[keep].tolist()], dtype=torch.long)
            subset_scores = all_scores[keep][:, torch.from_numpy(chosen)]
            subset_rank = _ranks_within(subset_scores, subset_labels)
            scores_at_size.append(float((subset_rank <= 1).float().mean()))
        if scores_at_size:
            matched[size] = {
                "rank1": float(np.mean(scores_at_size)),
                "sd": float(np.std(scores_at_size)),
                "chance": 1.0 / size,
            }

    return {
        "users": users,
        "probes": int(rank.numel()),
        "rank1_at_gallery_size": matched,
        "gallery_k": gallery_k,
        "probe_k": probe_k,
        "rank1": cmc[0],
        "rank5": cmc[min(4, users - 1)],
        "chance": 1.0 / users,
        "mean_rank": float(rank.float().mean()),
        "same_session_fallback_users": same_session_fallback,
        "skipped_users": skipped,
        "cmc": cmc,
    }


def format_cmc(result: dict) -> str:
    """ASCII only - Windows consoles are cp1252 and crash on box drawing when piped."""
    if not result.get("cmc"):
        return "Identification: not enough users with usable sessions."

    lines = [
        f"Closed-set identification: {result['users']} enrolled users, "
        f"{result['probes']} probes, gallery_k={result['gallery_k']}, probe_k={result['probe_k']}",
        f"  rank-1 : {result['rank1']:.4f}   (chance {result['chance']:.4f})",
        f"  rank-5 : {result['rank5']:.4f}",
        f"  mean rank {result['mean_rank']:.2f} of {result['users']}",
    ]
    matched = result.get("rank1_at_gallery_size") or {}
    if matched:
        lines.append("  rank-1 restricted to a random gallery of N users:")
        for size in sorted(matched):
            entry = matched[size]
            lines.append(f"    N={size:<4} {entry['rank1']:.4f} +/- {entry['sd']:.4f}"
                         f"   (chance {entry['chance']:.4f})")
    if result.get("same_session_fallback_users"):
        lines.append(f"  WARNING: {result['same_session_fallback_users']} users had one session; "
                     f"their gallery and probe share it")
    if result.get("skipped_users"):
        lines.append(f"  {result['skipped_users']} users skipped (too few windows)")
    return "\n".join(lines)

def variance_decomposition(embeddings: torch.Tensor, sample_index, normalise: bool = True) -> dict:
    """
    Split embedding variance into the three components that decide the k-curve.

    A template averages k windows from ONE session, so it can only reduce the
    within-session component. The between-session offset is shared by every window in
    a template and survives averaging untouched, which is why a flat curve is not by
    itself evidence of a broken sweep - and why the positive control, whose noise is
    independent per window, cannot settle that question either.

    Measuring the split turns the ambiguity into a prediction. Effective noise at k is
    roughly `between_session + within_session / k`, so the curve should improve until
    `within_session / k` falls below `between_session` and then flatten. Reporting
    `plateau_k` alongside the curve means a flat result can be checked against what
    the data says it should be, rather than argued about.

    Returns the three variances, the signal-to-shift ratio, and that plateau estimate.
    """
    sessions = getattr(sample_index, "window_session_ids", None)
    if embeddings.numel() == 0 or sessions is None or sessions.numel() == 0:
        return {}

    vectors = F.normalize(embeddings, dim=1) if normalise else embeddings

    session_means, user_means, within = [], [], []
    for user in range(sample_index.num_users):
        windows = sample_index.user_sample_indices[user]
        if windows.numel() == 0:
            continue
        per_user = []
        for session in torch.unique(sessions[windows]):
            group = windows[sessions[windows] == session]
            block = vectors[group]
            mean = block.mean(dim=0)
            per_user.append(mean)
            within.append(float((block - mean).pow(2).sum(dim=1).mean()))
        if per_user:
            stacked = torch.stack(per_user)
            session_means.append(stacked)
            user_means.append(stacked.mean(dim=0))

    if not user_means:
        return {}

    users = torch.stack(user_means)
    grand = users.mean(dim=0)
    between_user = float((users - grand).pow(2).sum(dim=1).mean())
    between_session = float(torch.cat([
        (block - block.mean(dim=0)).pow(2).sum(dim=1) for block in session_means
        if block.shape[0] > 1
    ]).mean()) if any(block.shape[0] > 1 for block in session_means) else 0.0
    within_session = float(np.mean(within)) if within else 0.0

    plateau = None
    if between_session > 0 and within_session > 0:
        plateau = max(1, int(round(within_session / between_session)))

    return {
        "between_user": between_user,
        "between_session": between_session,
        "within_session": within_session,
        "signal_to_shift": between_user / between_session if between_session > 0 else float("inf"),
        "plateau_k": plateau,
    }


def format_decomposition(parts: dict) -> str:
    if not parts:
        return "variance decomposition unavailable (no session provenance)"
    lines = [
        "Embedding variance, and what averaging can reach:",
        f"  between-user        {parts['between_user']:.4f}   the signal",
        f"  between-session     {parts['between_session']:.4f}   shared by a whole template, averaging cannot reduce it",
        f"  within-session      {parts['within_session']:.4f}   what averaging k windows actually reduces",
        f"  signal / shift      {parts['signal_to_shift']:.2f}",
    ]
    if parts.get("plateau_k"):
        lines.append(f"  predicted plateau   k ~ {parts['plateau_k']}   "
                     "(above this, within-session noise is below the between-session shift)")
    return "\n".join(lines)


def window_curve(model, sample_index, device, ks=(1, 2, 4, 8, 16),
                 pairs_per_user: int = 64, match_ratio: float = 0.5,
                 seed: int | None = 67, within_dataset_negatives: bool = True,
                 batch_size: int = 512) -> list[dict]:
    """
    Metrics as a function of how many windows are aggregated per side.

    k=1 reproduces the single-window operating point the project has reported so far,
    so the curve is an added dimension rather than a replacement metric.
    """
    embeddings = embed_all(model, sample_index, device, batch_size)
    if embeddings.numel() == 0:
        return []

    rows = []
    for entry in ks:
        # An entry is either a bare k (the symmetric diagonal, as before) or a
        # (reference, probe) pair. The two sides are worth very different amounts:
        # measured on a held-out corpus, 8 to 16 reference windows bought +0.13 rank-1
        # where 1 to 6 probe windows bought +0.11 from a lower base. Averaging k on
        # both sides therefore sweeps the wrong line through the space.
        k_ref, k_probe = (entry, entry) if isinstance(entry, (int, float)) else tuple(entry)
        k_ref, k_probe = int(k_ref), int(k_probe)

        manifest = generate_template_manifest(
            sample_index, pairs_per_user=pairs_per_user, k=k_ref, k_probe=k_probe,
            match_ratio=match_ratio, seed=seed,
            within_dataset_negatives=within_dataset_negatives,
        )
        if manifest["labels"].numel() == 0:
            rows.append({"k": k_ref, "k_probe": k_probe, "pairs": 0,
                         "note": "no session had k windows"})
            continue

        scores = score_templates(model, embeddings, manifest, device)
        labels = manifest["labels"]
        metrics = pair_metrics(scores, labels)
        accuracy = float(((scores > metrics["eer_threshold"]).float() == labels).float().mean())

        rows.append({
            "k": k_ref,
            "k_probe": k_probe,
            "pairs": int(labels.numel()),
            "positive_fraction": float(labels.mean()),
            "auc": metrics["auc"],
            "eer": metrics["eer"],
            "accuracy_at_eer": accuracy,
            "users_skipped": int(manifest["skipped_users"]),
            "single_session_users": int(manifest["single_session_users"]),
        })
    return rows


def format_curve(rows: list[dict]) -> str:
    lines = [f"{'k':>4} {'pairs':>7} {'pos':>6} {'AUC':>7} {'EER':>7} {'acc@EER':>8}  notes"]
    lines.append("-" * 62)
    for row in rows:
        if row.get("pairs", 0) == 0:
            lines.append(f"{row['k']:>4} {0:>7} {'':>6} {'':>7} {'':>7} {'':>8}  {row.get('note','')}")
            continue
        note = ""
        if row["single_session_users"]:
            note = f"{row['single_session_users']} single-session user(s)"
        if row["users_skipped"]:
            note += f"{'; ' if note else ''}{row['users_skipped']} user(s) lacked k windows"
        lines.append(f"{row['k']:>4} {row['pairs']:>7} {row['positive_fraction']:>6.3f} "
                     f"{row['auc']:>7.4f} {row['eer']:>7.4f} {row['accuracy_at_eer']:>8.4f}  {note}")
    return "\n".join(lines)
