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
    for user in range(sample_index.num_users):
        groups = _sessions_for_user(sample_index, user, minimum=k)
        if groups:
            eligible[user] = groups

    positives_target = int(round(pairs_per_user * match_ratio))
    negatives_target = pairs_per_user - positives_target

    x1, x2, labels, anchors = [], [], [], []
    skipped_users = sample_index.num_users - len(eligible)
    single_session = 0

    def draw(group: torch.Tensor) -> list[int]:
        picked = rng.choice(group.numel(), size=k, replace=False)
        return [int(group[int(i)]) for i in picked]

    for user, groups in eligible.items():
        # Positives: two different sessions of the same user.
        if len(groups) >= 2:
            for _ in range(positives_target):
                first, second = rng.choice(len(groups), size=2, replace=False)
                x1.append(draw(groups[int(first)]))
                x2.append(draw(groups[int(second)]))
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
                x1.append(draw(groups[int(rng.choice(len(groups)))]))
                partner_groups = eligible[partner]
                x2.append(draw(partner_groups[int(rng.choice(len(partner_groups)))]))
                labels.append(0.0)
                anchors.append(user)

    if not labels:
        empty = torch.empty((0, k), dtype=torch.long)
        return {"x1_indices": empty, "x2_indices": empty,
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
    for k in ks:
        manifest = generate_template_manifest(
            sample_index, pairs_per_user=pairs_per_user, k=int(k),
            match_ratio=match_ratio, seed=seed,
            within_dataset_negatives=within_dataset_negatives,
        )
        if manifest["labels"].numel() == 0:
            rows.append({"k": int(k), "pairs": 0, "note": "no session had k windows"})
            continue

        scores = score_templates(model, embeddings, manifest, device)
        labels = manifest["labels"]
        metrics = pair_metrics(scores, labels)
        accuracy = float(((scores > metrics["eer_threshold"]).float() == labels).float().mean())

        rows.append({
            "k": int(k),
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
