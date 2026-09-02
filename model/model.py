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

DEFAULT_EXTRACTOR = "paper_gnn_bilstm"


def create_model(
    embedding_dim=128,
    seq_len=10,
    lr=0.001,
    device=None,
    extractor=DEFAULT_EXTRACTOR,
    extractor_params=None,
    num_channels=7,
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

    model = SiameseModel(backbone, embedding_dim=embedding_dim).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Feature extractor: {backbone.describe()}")
    print(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


class SiameseModel(nn.Module):
    """
    Siamese wrapper around the Model feature extractor.
    Given two sequences, it computes their distance by learning
    a linear layer over the absolute difference of their embeddings.
    """

    def __init__(self, feature_extractor, embedding_dim=128):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x1, x2):
        # Extract features (embeddings)
        e1 = self.feature_extractor(x1)
        e2 = self.feature_extractor(x2)

        # Compute absolute difference
        diff = torch.abs(e1 - e2)

        # Return logit
        return self.classifier(diff)
