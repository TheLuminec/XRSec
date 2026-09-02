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
from torch.utils.data import DataLoader, Dataset


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


def create_window_loader(sample_index, batch_size: int, device, seed: int, num_workers: int = 0):
    pin_memory = device.type == "cuda" if device else False
    return DataLoader(
        WindowDataset(sample_index),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def train_identity_epoch(model, head, loader, optimizer, device):
    """One pass of identity classification. Returns (loss, classification accuracy)."""
    model.train()
    head.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for windows, labels in loader:
        windows = windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = head(model.embed(windows), labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * windows.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))

    return total_loss / max(total, 1), correct / max(total, 1)


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
