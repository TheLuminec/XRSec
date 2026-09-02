"""
Plain BiLSTM baseline: the paper architecture with the GNN aggregation removed.

Raw channels go straight into the recurrent stack, so comparing this against
``paper_gnn_bilstm`` at matched width isolates what the graph branches contribute.
It also serves as a second worked example of the extractor contract.
"""

import torch
import torch.nn as nn

from feature_extractor import FeatureExtractor, register


@register("bilstm")
class BiLstmExtractor(FeatureExtractor):
    """
    Args:
        lstm_hidden: Hidden width per direction.
        num_layers: Stacked LSTM layers.
        dropout: Inter-layer dropout (ignored when num_layers == 1).
        pooling: How the time dimension is collapsed - "mean", "max" or "last".
    """

    def __init__(
        self,
        seq_len: int,
        num_channels: int = 7,
        embedding_dim: int = 128,
        lstm_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        pooling: str = "mean",
    ):
        super().__init__(
            seq_len=seq_len,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            lstm_hidden=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
        )
        if pooling not in {"mean", "max", "last"}:
            raise ValueError(f"pooling must be 'mean', 'max' or 'last', got {pooling!r}.")
        self.pooling = pooling

        self.lstm = nn.LSTM(
            num_channels,
            lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            # torch only applies dropout between layers, and warns if set with one layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden * 2, embedding_dim)

    def forward(self, x):
        # (batch, channels, timesteps) -> (batch, timesteps, channels)
        out, _ = self.lstm(x.permute(0, 2, 1))

        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "max":
            pooled = out.max(dim=1).values
        else:
            pooled = out[:, -1, :]

        return self.fc(pooled)

    @classmethod
    def search_space(cls):
        return {
            "lstm_hidden": [32, 64, 128],
            "num_layers": [1, 2, 3],
            "dropout": [0.0, 0.2],
            "pooling": ["mean", "max", "last"],
        }
