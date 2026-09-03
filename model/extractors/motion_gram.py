"""
Motion-Gram: cross-channel coupling and lead-lag timing as the identity descriptor.

Thesis
------
Every extractor here collapses the time axis with *marginal* statistics: the BiLSTM
stack mean-pools its hidden states, ``motion_tdnn`` pools per-channel mean and
standard deviation. Both summarise each channel in isolation, so anything that lives
in the *relationship between* channels is discarded before the embedding is formed.

That relationship is a plausible place for identity to hide. "How much does this
person move" is marginal and largely task-driven - it is what the scene asks of
them. "When this person yaws their head, do they also translate, and does the
rotation lead the translation or trail it" is a motor-coordination pattern, closer to
a personal constant than to the content being viewed. The second question is exactly
what a marginal pooling layer cannot ask.

So the descriptor here is the Gram matrix of the feature channels rather than their
marginals:

    1. Channel bank    the same closed-form kinematics as ``motion_tdnn`` (pose,
                       body-frame angular velocity, linear velocity, accelerations),
                       so the two are compared on equal input footing.
    2. Zero-lag block  the correlation matrix of the feature channels over the
                       window - which channels co-vary, and how strongly.
    3. Lagged block    the cross-correlation at a fixed offset. Its diagonal is each
                       channel's autocorrelation at that lag (how persistent, how
                       smooth this person's motion is); the antisymmetric part of its
                       off-diagonal is the *direction* of coupling, i.e. which channel
                       leads which. This is the piece with no analogue anywhere else
                       in the pipeline: correlation at lag zero is symmetric and
                       cannot represent precedence at all.
    4. Marginals       mean and log standard deviation, concatenated back in, so the
                       descriptor is a superset of statistics pooling rather than a
                       replacement for it.

Geometry
--------
Correlation matrices are not points in a flat space, and treating their raw entries
as a Euclidean vector is the usual reason covariance descriptors underperform. The
zero-lag block is therefore mapped through the log-Cholesky chart (Lin, 2019): shrink
toward the identity, factor C = L L^T, and emit the strict lower triangle of L
alongside log diag(L). That is a smooth bijection from the SPD cone to a flat space,
so the linear layer downstream operates on coordinates where addition is meaningful.

log-Cholesky is chosen over the more familiar log-Euclidean map (matrix logarithm via
eigendecomposition) for a concrete numerical reason: with ``hidden`` feature channels
and only ``seq_len`` timesteps, the correlation matrix is rank-deficient whenever
hidden > seq_len, which is the normal case at 5s/20Hz. Every surplus eigenvalue then
sits at the shrinkage floor, and the eigendecomposition backward pass divides by
differences between eigenvalues - so log-Euclidean produces NaN gradients precisely
in the configuration this extractor is meant to run in. Cholesky has no such failure
mode for a positive-definite input, which shrinkage guarantees.

``shrinkage`` is doing double duty and is a real hyperparameter, not a fudge factor:
it conditions the matrix, and it is Ledoit-Wolf shrinkage of a covariance estimated
from few samples, which at 100 timesteps and 32 channels is a genuinely
under-determined estimate.

Standing against this idea
--------------------------
``motion_tdnn`` records that hand-computed "channel correlations" scored 0.49-0.53 on
held-out users, i.e. chance. That is direct negative evidence, it lowers the prior
here, and it should be read before spending GPU time on this. What it measured was
the zero-lag correlation of the seven *raw* channels, scored as a feature on its own.
This differs in three ways that are individually testable from the search space:
correlations are taken over learned kinematic features rather than raw channels
(``layers``), the SPD geometry is respected rather than ignored (``matrix_map``), and
the lagged block adds precedence information that a zero-lag correlation provably
cannot carry (``lag``). If ``lag=0, layers=0, matrix_map=correlation`` reproduces
0.50, the earlier result is confirmed and the remaining axes say which addition, if
any, moves it.

Two configurations are worth running before any others, both diagnostic rather than
competitive:

- ``layers=0`` removes the feature-learning front end entirely, leaving a fixed
  statistic of the kinematic bank behind one linear layer. ``motion_tdnn`` established
  that capacity is not the bottleneck and that 43 identities get memorised at any
  size; this is the cheapest available probe of whether cross-channel signal exists at
  all, with almost nothing left to memorise with.
- ``use_marginals=False`` makes the descriptor blind to both position and scale
  (correlation is scale-free, and centring removes the mean). Since mean head
  position alone already scores ~0.69, that shortcut otherwise dominates any
  measurement. This configuration asks whether there is identity information in the
  motion at all, separately from posture and height.

Measured results - THIS DID NOT WORK
------------------------------------
5-fold cross-validation, 48 users of VR_User_Behavior partitioned into 5 disjoint
held-out groups, 5s @ 20Hz, emb 128, 512 pairs/user, per_dataset normalisation,
within-dataset negatives, 20 epochs (sweep e06289a224, 25 runs, 0 failures):

    rank  mean acc      sd  folds  extractor
       1    0.6580   0.025      5  motion_tdnn
       2    0.6562   0.037      5  bilstm
       3    0.6517   0.026      5  paper_gnn_bilstm
       4    0.6252   0.025      5  motion_gram      <- this one
       5    0.5173   0.003      5  random

Every extractor saw the same five folds, so the paired comparison is the one that
counts. Against ``bilstm``, fold by fold:

    motion_tdnn        +0.0018   t(4) = +0.16   won 3/5 folds
    paper_gnn_bilstm   -0.0045   t(4) = -0.69   won 2/5 folds
    motion_gram        -0.0310   t(4) = -3.67   won 0/5 folds

``motion_gram`` is worse than a plain BiLSTM, it lost on every fold, and the effect is
large relative to the fold-to-fold spread. This is not a "needs more tuning" result.
The cross-channel coupling hypothesis, as implemented here, is not competitive with
simply running the raw channels through a recurrent stack.

An earlier single-split run had this at 0.657 against 0.645 for ``bilstm`` and looked
like a marginal win. Cross-validation reversed the sign. Single-split numbers on this
data are worthless: ``bilstm`` alone ranges 0.620 to 0.727 across the five folds, an
11-point spread, because the effective sample size is held-out USERS, not pairs.

Two findings from the same sweep that outlive this extractor:

1. **The reported metric has a ~2-point floor above chance.** ``random`` - which
   ignores its input entirely - scores 0.5173 on best-epoch accuracy but 0.4973 at the
   final epoch. That +0.020 is pure max-over-20-epochs selection bias, and every
   extractor shows the same inflation (+0.016 to +0.027). So the honest chance floor
   for any "best test accuracy" in results/runs.csv is about 0.517, not 0.500.
2. **Nothing beats the BiLSTM baseline.** The top three are statistically
   indistinguishable, the published GNN architecture included. Whatever is limiting
   this problem, it is not the choice of feature extractor.

A later 50-run sweep (9ac1eb6cda) makes the verdict worse, and is the most damning
measurement here. It crossed every extractor with the training objective, scoring on
``selected_test_acc`` - the epoch chosen on a validation group of users disjoint from
both training and test, so the numbers are not selection-inflated:

    extractor          pair_bce   identity_softmax   paired diff   folds won
    bilstm               0.6177             0.6849       +0.0672         5/5
    paper_gnn_bilstm     0.6214             0.6867       +0.0653         5/5
    motion_tdnn          0.6237             0.6863       +0.0626         5/5
    motion_gram          0.6038             0.5980       -0.0058         1/5
    random               0.5014             0.4947       -0.0067         0/5

Swapping the pairwise head for AM-Softmax identity classification is worth about
+6.5 points to every other architecture, consistently, on every fold. It is worth
nothing to this one - ``motion_gram`` sits with the ``random`` control on that axis.

That is the strongest evidence against the design. When a better-structured training
signal lifts three different backbones by six points and leaves this descriptor
exactly where it was, the limitation is not the objective, the capacity, or the
tuning. It is that the Gram descriptor has already discarded whatever the improved
objective is teaching the others to encode - which is what collapsing a window to
channel-pair correlations does. Cross-channel coupling is apparently not where
identity lives in this signal, or not recoverable once the time axis is gone.

What was never isolated, and is the only part still worth testing if anyone returns to
this: whether the lagged block contributes anything at all (``lag=0`` vs 4 vs 8 on
matched folds). The lead-lag antisymmetric term is the one genuinely novel idea here
and it has only ever been measured bundled with everything else. Also untested under
cross-validation: ``use_marginals=False``, which would say whether anything outside
posture and height is person-specific at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from extractors._kinematics import derived_channels, split_pose
from feature_extractor import FeatureExtractor, register

#: Floor on per-window standard deviations, keeping a constant channel finite through
#: both the standardisation and the log.
_EPS = 1e-4

#: A lagged correlation needs enough overlapping frames to mean anything. Below this
#: the lagged block is dropped rather than estimated from a handful of samples.
_MIN_LAG_OVERLAP = 4


class _TemporalBlock(nn.Module):
    """Dilated 1D convolution -> BatchNorm -> GELU -> dropout, length preserving."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


@register("motion_gram")
class MotionGram(FeatureExtractor):
    """
    Args:
        hidden: Width of each temporal block, and the side length of the Gram matrix.
            Descriptor size is quadratic in this, so it is deliberately smaller than a
            pooling extractor's width would be.
        layers: Dilated temporal blocks before pooling (dilations 1, 2, 4, ...). 0
            takes the Gram matrix of the normalised kinematic bank directly, which is
            the near-parameter-free diagnostic configuration.
        kernel: Temporal kernel size of each block.
        dropout: Dropout after each block.
        lag: Offset in frames for the lagged block; 0 disables it. At 20Hz, 4 frames
            is 200ms, roughly the scale of a deliberate head movement. Silently
            disabled when the window is too short to leave enough overlap.
        shrinkage: Ledoit-Wolf shrinkage of the correlation matrix toward the
            identity, in [0, 1). Also what keeps the matrix positive definite.
        matrix_map: How the SPD block reaches flat space - "log_cholesky" for the
            log-Cholesky chart, "correlation" for the raw upper triangle (the
            geometry-ignoring ablation). Both emit the same number of features.
        use_marginals: Concatenate per-channel mean and log standard deviation.
            False makes the descriptor blind to position and scale.
        use_derivatives: Include the velocity/acceleration channels in the bank.
    """

    def __init__(
        self,
        seq_len: int,
        num_channels: int = 7,
        embedding_dim: int = 128,
        hidden: int = 32,
        layers: int = 2,
        kernel: int = 5,
        dropout: float = 0.1,
        lag: int = 4,
        shrinkage: float = 0.01,
        matrix_map: str = "log_cholesky",
        use_marginals: bool = True,
        use_derivatives: bool = True,
    ):
        super().__init__(
            seq_len=seq_len, num_channels=num_channels, embedding_dim=embedding_dim,
            hidden=hidden, layers=layers, kernel=kernel, dropout=dropout, lag=lag,
            shrinkage=shrinkage, matrix_map=matrix_map, use_marginals=use_marginals,
            use_derivatives=use_derivatives,
        )
        if matrix_map not in {"log_cholesky", "correlation"}:
            raise ValueError(f"matrix_map must be 'log_cholesky' or 'correlation', got {matrix_map!r}.")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError(f"shrinkage must be in [0, 1), got {shrinkage}.")
        if matrix_map == "log_cholesky" and shrinkage <= 0.0:
            raise ValueError("log_cholesky needs shrinkage > 0 to keep the matrix positive definite.")

        self.matrix_map = matrix_map
        self.shrinkage = float(shrinkage)
        self.use_marginals = bool(use_marginals)
        self.use_derivatives = bool(use_derivatives)

        # The requested lag is what gets recorded in hyperparams; this is what the
        # window length actually permits. Short windows (the contract tests use 10
        # timesteps) would otherwise estimate a correlation from two frames.
        self.lag_frames = int(lag) if lag > 0 and seq_len - lag >= _MIN_LAG_OVERLAP else 0

        # Derived from one dummy pass rather than hardcoded: the bank width depends on
        # whether the input carries orientation (7 + 17 with a quaternion, 3 + 10 for
        # position-only data), and every downstream dimension follows from it. Three
        # timesteps is the minimum the second-difference channels need.
        with torch.no_grad():
            bank_channels = self._channel_bank(
                torch.zeros(1, num_channels, max(int(seq_len), 3))
            ).shape[1]
        self.input_norm = nn.BatchNorm1d(bank_channels)

        blocks = []
        in_channels = bank_channels
        for index in range(layers):
            blocks.append(_TemporalBlock(in_channels, hidden, kernel, dilation=2 ** index, dropout=dropout))
            in_channels = hidden
        self.blocks = nn.Sequential(*blocks)

        width = in_channels
        self.register_buffer("_tril", torch.tril_indices(width, width, offset=-1), persistent=False)
        self.register_buffer("_triu", torch.triu_indices(width, width, offset=0), persistent=False)

        descriptor_dim = width * (width + 1) // 2                    # zero-lag block
        if self.use_marginals:
            descriptor_dim += 2 * width                              # mean, log std
        if self.lag_frames:
            descriptor_dim += width + width * (width - 1) // 2       # autocorr, lead-lag

        self.descriptor_norm = nn.BatchNorm1d(descriptor_dim)
        self.embed = nn.Linear(descriptor_dim, embedding_dim)

    def _channel_bank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw (batch, num_channels, T) -> stacked kinematic channels, all at length T.

        With ``channels=position`` there is no quaternion, so the orientation channels
        are absent and the bank is 3 + 10 wide rather than 7 + 17. The Gram matrix is
        sized from whatever this returns, so the descriptor follows automatically.
        """
        quaternion, position = split_pose(x, self.num_channels, owner=type(self).__name__)
        channels = []
        if quaternion is not None:
            channels.append(quaternion / (quaternion.norm(dim=1, keepdim=True) + 1e-8))
        channels.append(position)

        if self.use_derivatives:
            channels += derived_channels(quaternion, position)

        return torch.cat(channels, dim=1)

    def forward(self, x):
        features = self.blocks(self.input_norm(self._channel_bank(x)))
        _, width, timesteps = features.shape

        mean = features.mean(dim=2)
        std = features.std(dim=2, unbiased=False)
        # Standardised per window and per channel, so every Gram entry below is a
        # correlation: scale-free, and comparable across datasets whose coordinate
        # frames differ by 40x in range.
        z = (features - mean.unsqueeze(2)) / (std + _EPS).unsqueeze(2)

        parts = [mean, torch.log(std + _EPS)] if self.use_marginals else []

        correlation = z @ z.transpose(1, 2) / timesteps
        correlation = 0.5 * (correlation + correlation.transpose(1, 2))  # exact symmetry
        identity = torch.eye(width, device=features.device, dtype=features.dtype)
        correlation = (1.0 - self.shrinkage) * correlation + self.shrinkage * identity

        if self.matrix_map == "log_cholesky":
            factor = torch.linalg.cholesky(correlation)
            parts.append(factor[:, self._tril[0], self._tril[1]])
            parts.append(torch.log(torch.diagonal(factor, dim1=1, dim2=2)))
        else:
            parts.append(correlation[:, self._triu[0], self._triu[1]])

        if self.lag_frames:
            lag = self.lag_frames
            # cross[i, j] = corr(channel_i at t, channel_j at t + lag).
            cross = z[:, :, :-lag] @ z[:, :, lag:].transpose(1, 2) / (timesteps - lag)
            # Diagonal: how quickly each channel decorrelates from itself, i.e. the
            # smoothness of this person's motion. Antisymmetric part: which channel
            # leads which, the only place in the descriptor where order survives.
            parts.append(torch.diagonal(cross, dim1=1, dim2=2))
            skew = cross - cross.transpose(1, 2)
            parts.append(skew[:, self._tril[0], self._tril[1]])

        return self.embed(self.descriptor_norm(torch.cat(parts, dim=1)))

    @classmethod
    def search_space(cls):
        # Held to 216 combinations, matching motion_tdnn, because `grid: auto` trains
        # every one of them in a single process. kernel, dropout and use_derivatives
        # are deliberately absent: they are secondary to the question this extractor
        # asks, and are still settable through extractor_params or an explicit grid.
        return {
            "hidden": [16, 32, 64],
            "layers": [0, 1, 2],
            "lag": [0, 4, 8],
            "shrinkage": [0.01, 0.1],
            "matrix_map": ["correlation", "log_cholesky"],
            "use_marginals": [False, True],
        }
