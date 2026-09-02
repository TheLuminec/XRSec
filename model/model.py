"""
Siamese head and model factory.

This module no longer contains an architecture. The backbone is chosen at runtime
from the feature extractor registry (see model/feature_extractor.py and
model/extractors/), and everything here is architecture-agnostic:

    create_model()  builds <extractor> + SiameseModel + criterion + optimizer
    SiameseModel    classifies |e1 - e2| through a learned linear layer

The published GNN + BiLSTM + attention architecture lives in
model/extractors/paper_gnn_bilstm.py and is the default extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EXTRACTOR = "paper_gnn_bilstm"


def create_model(
    embedding_dim=128,
    seq_len=10,
    lr=0.001,
    device=None,
    extractor=DEFAULT_EXTRACTOR,
    extractor_params=None,
    num_channels=7,
    weight_decay=0.0,
    head="diff_linear",
    head_scale=10.0,
):
    """
    Create the model.

    Args:
        embedding_dim: Dimension of the embedding space
        seq_len: Length of the input sequence
        lr: Learning rate
        device: Device to train on
        extractor: Registered feature extractor name (see model/extractors/)
        extractor_params: Hyperparameter overrides for that extractor
        num_channels: Number of input channels
        head: Pair scoring head - "diff_linear" or "cosine" (see SiameseModel)
        head_scale: Initial logit scale for the cosine head
        weight_decay: L2 penalty for Adam. The model overfits held-out users within a
            couple of epochs, and nothing previously regularised the weights.
    """
    # Imported here rather than at module scope: extractor modules import from this
    # module, so a top-level import would be circular.
    import feature_extractor as fe

    backbone = fe.create(
        extractor,
        seq_len=seq_len,
        num_channels=num_channels,
        embedding_dim=embedding_dim,
        hyperparams=extractor_params,
    ).to(device)

    model = SiameseModel(backbone, embedding_dim=embedding_dim,
                         head=head, head_scale=head_scale).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Feature extractor: {backbone.describe()}  head={head}")
    print(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, criterion, optimizer


class SiameseModel(nn.Module):
    """
    Siamese wrapper around a feature extractor: two windows in, one logit out.

    Two scoring heads are available.

    ``diff_linear`` (default, the original) learns a linear layer over
    |e1 - e2|. It is expressive, but every weight is tied to a particular
    embedding dimension, and those dimensions are shaped by the training
    identities - which is a route to memorising who is who. Training accuracy
    reaching 0.93 while held-out users sit at 0.68 is what that looks like.

    ``cosine`` scores by cosine similarity through a learnable scale and bias.
    It is a proper metric: magnitude is discarded, no per-dimension weights are
    learned, and the same comparison applies to identities never seen in
    training. This is what verification systems use, and it is the head the
    identity-classification objective (see model/identity_train.py) trains for.

    Args:
        head: "diff_linear" or "cosine".
        head_scale: Initial logit scale for the cosine head. Cosine lies in
            [-1, 1], so it needs scaling before BCE to produce usable gradients.
    """

    def __init__(self, feature_extractor, embedding_dim=128, head="diff_linear", head_scale=10.0):
        super().__init__()
        if head not in ("diff_linear", "cosine"):
            raise ValueError(f"head must be 'diff_linear' or 'cosine', got {head!r}.")

        self.feature_extractor = feature_extractor
        self.head = head

        if head == "diff_linear":
            # Key name preserved: checkpoints written before heads were selectable
            # still load.
            self.classifier = nn.Linear(embedding_dim, 1)
        else:
            self.cosine_scale = nn.Parameter(torch.tensor(float(head_scale)))
            self.cosine_bias = nn.Parameter(torch.tensor(0.0))

    def embed(self, x):
        return self.feature_extractor(x)

    def score(self, e1, e2):
        """Logit from a pair of embeddings, for callers that already have them."""
        if self.head == "diff_linear":
            return self.classifier(torch.abs(e1 - e2))
        cosine = F.cosine_similarity(e1, e2, dim=1, eps=1e-8).unsqueeze(1)
        return self.cosine_scale * (cosine - self.cosine_bias)

    def forward(self, x1, x2):
        return self.score(self.feature_extractor(x1), self.feature_extractor(x2))
