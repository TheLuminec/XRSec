"""
Identity-classification training with an angular margin (AM-Softmax).

Why this exists
---------------
The pairwise objective trains |e1 - e2| through a linear layer with BCE. Two things
about that hurt generalisation to unseen users, and both show up in the measurements:

1. **It learns per-dimension weights tied to the training identities.** Whatever the
   embedding dimensions come to mean is shaped by the 43-238 people in the training
   set, and the head weights them accordingly. Training accuracy reaches 0.93 while
   held-out users sit at 0.68.
2. **It wastes the data.** A pair is one training example built from two windows, so
   ~100k windows become ~100k pairs. Classifying identity uses every window as its
   own example, against a target with far more structure than one bit.

The verification field's answer is to train a classifier over training identities
with an additive angular margin, then throw the classifier away and compare
embeddings by cosine similarity. The margin forces each identity into a tight
angular cluster with space between clusters, which is a property that transfers to
identities the classifier never saw - the reason speaker and face verification are
trained this way rather than pairwise.

The classifier head is discarded after training. What is saved is the extractor plus
a cosine scoring head, so evaluation, boosting and `mode=test` see exactly the same
`forward(x1, x2) -> logit` interface as a pairwise-trained model, and the numbers
stay comparable.

Calibration
-----------
Cosine similarity ranks well but says nothing about where the accept/reject threshold
belongs, and accuracy is measured at `logit > 0`. After each epoch the two scalars of
the cosine head (scale and bias) are fitted on *training* pairs with the extractor
frozen. Without this, AUC would look fine while accuracy sat at chance for the wrong
reason. Nothing about the held-out set is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler


class WindowDataset(Dataset):
    """Every window as its own example, labelled with the user it came from."""

    def __init__(self, sample_index):
        self.samples = sample_index.samples
        labels = torch.zeros(sample_index.sample_count, dtype=torch.long)
        for user_index, window_indices in enumerate(sample_index.user_sample_indices):
            labels[window_indices] = user_index
        self.labels = labels
        self.num_classes = sample_index.num_users

    def __len__(self):
        return int(self.samples.shape[0])

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]



class CappedIdentitySampler(Sampler):
    """
    Draw at most `cap` windows from each identity per epoch, without replacement.

    The alternative - inverse-frequency weighting - equalises identities by resampling
    WITH replacement, so a 1260-window identity gets drawn far fewer times than it has
    windows: the effective identity count rises while the number of distinct windows
    the model sees falls. Capping equalises from the other direction. It discards the
    surplus of over-represented identities rather than re-drawing the scarce ones, so
    every window it yields is a distinct window, and the epoch gets cheaper rather than
    more expensive.

    Why this matters on the current corpus: the 419 pre-BOXRR identities are 17.2% of
    2439 and hold 40.8% of all windows, so uniform sampling gives them four times their
    share of every epoch's gradient and discounts the 2020 identities just acquired.

    A fresh subset is drawn each epoch, so the surplus is not permanently discarded -
    over many epochs a large identity contributes all of its windows, just never more
    than `cap` of them at once.
    """

    def __init__(self, labels: torch.Tensor, cap: int | None = None,
                 generator: torch.Generator | None = None):
        self.labels = labels
        self.generator = generator
        counts = torch.bincount(labels)
        present = counts[counts > 0]
        # Default to the median: identities above it are trimmed, those below are
        # untouched, so the epoch stays close to representative rather than collapsing
        # to the smallest identity.
        self.cap = int(cap) if cap else int(present.median().item())
        self.by_identity = [torch.nonzero(labels == i, as_tuple=True)[0]
                            for i in range(len(counts))]
        self.length = int(sum(min(self.cap, int(g.numel())) for g in self.by_identity))

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        picked = []
        for windows in self.by_identity:
            total = int(windows.numel())
            if total == 0:
                continue
            if total <= self.cap:
                picked.append(windows)
            else:
                order = torch.randperm(total, generator=self.generator)[:self.cap]
                picked.append(windows[order])
        pool = torch.cat(picked)
        return iter(pool[torch.randperm(pool.numel(), generator=self.generator)].tolist())


class AMSoftmaxHead(nn.Module):
    """
    Additive-margin softmax: cos(theta) - m for the true class, scaled by s.

    Subtracting a margin from the correct class's cosine forces that class to clear
    a gap before the loss is satisfied, which is what produces tight identity
    clusters rather than merely separable ones.
    """

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.35, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)
        self.margin = float(margin)
        self.scale = float(scale)
        self.num_classes = int(num_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.normalize(embeddings, dim=1) @ F.normalize(self.weight, dim=1).t()
        if labels is None:
            return cosine * self.scale
        margin = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), self.margin)
        return (cosine - margin) * self.scale


def identity_sample_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One over the number of windows that identity has, per window."""
    counts = torch.bincount(labels, minlength=num_classes).clamp(min=1).to(torch.float64)
    return (1.0 / counts)[labels]


def effective_identity_count(labels: torch.Tensor, num_classes: int) -> float:
    """
    How many evenly-represented identities this corpus is worth.

    Inverse participation ratio of the per-identity window counts: N identities each
    contributing equally gives N, and one identity dominating gives 1. Measured on the
    pooled 7-dataset corpus at sample_time=2 it is 193 against 312 real identities -
    so uniform sampling over windows discards about 38% of the identity diversity we
    have, on the one axis measured to bind.
    """
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float64)
    counts = counts[counts > 0]
    if counts.numel() == 0:
        return 0.0
    return float(counts.sum() ** 2 / (counts ** 2).sum())


def resolve_balance_mode(value) -> str:
    """`false`/`off` -> off, `true` -> weighted (back-compat), or a named mode."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "weighted"
    mode = str(value).strip().lower()
    if mode in ("", "false", "off", "none"):
        return "off"
    if mode in ("true", "weighted"):
        return "weighted"
    if mode == "cap":
        return "cap"
    raise ValueError(f"balance_identities must be off, weighted or cap, got {value!r}.")


def create_window_loader(sample_index, batch_size: int, device, seed: int, num_workers: int = 0,
                         balance_identities=False, balance_cap: int | None = None):
    """
    Windows as examples, optionally sampled so every identity contributes equally.

    Window counts per user span 77x on the pooled corpus (34 to 2639), so uniform
    sampling over windows lets the best-represented tenth of users supply a quarter of
    every epoch's gradient while the bottom half supply a fifth between them.
    AM-Softmax then separates frequent identities well and rare ones poorly, which is
    the wrong trade when the whole task is generalising to identities never seen.

    `balance_identities` draws each window with probability inversely proportional to
    how many its identity has, keeping the epoch the same size. Off by default so
    existing comparisons stay like-for-like.
    """
    pin_memory = device.type == "cuda" if device else False
    dataset = WindowDataset(sample_index)
    generator = torch.Generator().manual_seed(int(seed))

    mode = resolve_balance_mode(balance_identities)
    sampler = None
    if mode != "off":
        effective = effective_identity_count(dataset.labels, dataset.num_classes)
        if mode == "weighted":
            weights = identity_sample_weights(dataset.labels, dataset.num_classes)
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True, generator=generator)
            print(f"  balanced identity sampling (weighted, with replacement): "
                  f"{dataset.num_classes} identities, effective {effective:.0f} before")
        else:
            sampler = CappedIdentitySampler(dataset.labels, cap=balance_cap,
                                            generator=generator)
            print(f"  balanced identity sampling (capped at {sampler.cap} windows): "
                  f"{dataset.num_classes} identities, effective {effective:.0f} before, "
                  f"epoch {len(sampler)} of {len(dataset)} windows")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def train_identity_epoch(model, head, loader, optimizer, device):
    """
    One pass of identity classification. Returns (loss, classification accuracy).

    Metrics accumulate as GPU tensors and are read once at the end. Calling .item() per
    batch forces a device sync every step, so the GPU idles waiting for the CPU instead
    of queueing the next batch. Measured on a 3050 Ti at batch 256: 25.4 ms/step with
    per-batch .item(), 11.1 ms/step without - a 2.30x speedup for arithmetic that is
    mathematically identical.
    """
    model.train()
    head.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0

    for windows, labels in loader:
        windows = windows.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = head(model.embed(windows), labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * windows.size(0)
        correct += (logits.detach().argmax(dim=1) == labels).sum()
        total += int(labels.size(0))

    return float(total_loss) / max(total, 1), float(correct) / max(total, 1)


@torch.no_grad()
def _pair_cosines(model, sample_index, manifest, device, max_pairs: int, batch_size: int):
    """Cosine similarity and label for a sample of pairs, extractor frozen."""
    count = int(manifest["labels"].shape[0])
    if count == 0:
        return torch.empty(0), torch.empty(0)

    take = min(count, max_pairs)
    step = max(1, count // take)
    selected = torch.arange(0, count, step)[:take]

    x1 = manifest["x1_indices"][selected]
    x2 = manifest["x2_indices"][selected]
    labels = manifest["labels"][selected].view(-1)

    model.eval()
    cosines = []
    for start in range(0, selected.numel(), batch_size):
        a = sample_index.samples[x1[start:start + batch_size]].to(device)
        b = sample_index.samples[x2[start:start + batch_size]].to(device)
        cosines.append(F.cosine_similarity(model.embed(a), model.embed(b), dim=1, eps=1e-8).cpu())
    return torch.cat(cosines), labels


def calibrate_cosine_head(model, sample_index, manifest, device, batch_size: int,
                          max_pairs: int = 4096, steps: int = 300) -> None:
    """
    Fit the cosine head's scale and bias on training pairs, extractor frozen.

    Only two scalars are optimised and the embeddings are computed once, so this is
    cheap. It uses training pairs exclusively - the held-out set never informs the
    threshold.
    """
    if model.head != "cosine":
        return

    cosines, labels = _pair_cosines(model, sample_index, manifest, device, max_pairs, batch_size)
    if cosines.numel() == 0:
        return

    cosines = cosines.to(device)
    labels = labels.to(device)
    scale = model.cosine_scale.detach().clone().requires_grad_(True)
    bias = model.cosine_bias.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([scale, bias], lr=0.05)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimizer.zero_grad()
        criterion(scale * (cosines - bias), labels).backward()
        optimizer.step()

    with torch.no_grad():
        model.cosine_scale.copy_(scale.detach())
        model.cosine_bias.copy_(bias.detach())


def build_identity_trainer(model, sample_index, train_manifest, device, args):
    """
    Return (train_epoch_fn, optimizer) for identity training.

    The returned function matches the signature `run_training` expects, so the
    existing epoch loop, per-epoch evaluation, checkpointing and best-model
    selection are reused unchanged.
    """
    head = AMSoftmaxHead(
        embedding_dim=model.feature_extractor.embedding_dim,
        num_classes=sample_index.num_users,
        margin=float(getattr(args, "identity_margin", 0.35)),
        scale=float(getattr(args, "identity_scale", 30.0)),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.feature_extractor.parameters()) + list(head.parameters()),
        lr=float(args.lr),
        weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
    )

    window_loader = create_window_loader(
        sample_index,
        batch_size=int(args.batch_size),
        device=device,
        seed=int(getattr(args, "seed", 67)),
        num_workers=int(getattr(args, "num_workers", 0)),
        balance_identities=getattr(args, "balance_identities", False),
        balance_cap=getattr(args, "balance_cap", None),
    )

    print(f"Identity objective: {sample_index.num_users} classes, "
          f"{sample_index.sample_count} windows, margin={head.margin}, scale={head.scale}")

    def train_epoch_fn(_model, _loader, _criterion, _optimizer, _device):
        # run_training passes the pair loader and BCE criterion; identity training
        # uses neither, so they are ignored in favour of the window loader above.
        loss, accuracy = train_identity_epoch(model, head, window_loader, optimizer, device)
        calibrate_cosine_head(model, sample_index, train_manifest, device, int(args.batch_size))
        return loss, accuracy

    return train_epoch_fn, optimizer, head
