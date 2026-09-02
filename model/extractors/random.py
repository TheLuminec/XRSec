"""
A testing feature extractor that does not use any info and output a random embedding.
Useful for debugging and sanity checks.
"""

import torch
import torch.nn as nn

from feature_extractor import FeatureExtractor, register


@register("random")
class RandomExtractor(FeatureExtractor):
    """
    A testing feature extractor that outputs a random embedding.
    Useful for debugging and sanity checks.

    Any configuration scoring meaningfully above chance against this baseline is
    learning something; anything level with it is not.
    """

    # forward() ignores its input, so outputs are not reproducible across calls.
    # This opts out of the contract tests' output-equality checks.
    deterministic = False

    def __init__(self, seq_len: int, num_channels: int = 7, embedding_dim: int = 128, bias: bool = False):
        super().__init__(seq_len=seq_len, num_channels=num_channels, embedding_dim=embedding_dim, bias=bias)

        if bias:
            self.bias = nn.Parameter(torch.ones(embedding_dim))
        else:
            self.bias = 0

    def forward(self, x):
        # Return a random embedding of the same shape as the expected output
        return torch.rand(x.size(0), self.embedding_dim, device=x.device) + self.bias

    @classmethod
    def search_space(cls):
        return {
            "bias": [True, False]
        }
