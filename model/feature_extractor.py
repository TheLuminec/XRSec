"""
Slottable feature extractors.

A feature extractor is the only part of the pipeline that decides *how* a window of
headset motion becomes an embedding. Everything downstream — the Siamese head, pair
generation, boosting, evaluation — is unchanged by swapping one out, so extractors
can be written independently (including by a model) and compared on equal footing.

The contract is deliberately narrow:

    (batch, num_channels, seq_len)  ->  (batch, embedding_dim)

To add one, drop a module into ``model/extractors/`` that defines a FeatureExtractor
subclass decorated with ``@register("your_name")``. It is discovered automatically —
no imports to update, no registry to edit — and is immediately selectable with
``extractor=your_name`` and covered by the contract tests in
``tests/unit/test_feature_extractors.py``.

Declaring ``search_space()`` exposes the knobs worth sweeping. Keys must be real
constructor keyword arguments; that is enforced by the tests so a sweep can never be
launched against parameters an extractor does not accept.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module, ABC):
    """
    Base class for window -> embedding models.

    Subclasses declare their own hyperparameters as explicit keyword arguments with
    defaults, and pass them up to ``super().__init__`` so they can be recorded in
    checkpoints and rebuilt later.

    Args:
        seq_len: Number of timesteps per window (sample_time * sample_rate).
        num_channels: Input channels (7 = qx, qy, qz, qw, Hx, Hy, Hz).
        embedding_dim: Width of the output embedding.
        **hyperparams: Extractor-specific settings, recorded for checkpointing.
    """

    def __init__(self, seq_len: int, num_channels: int = 7, embedding_dim: int = 128, **hyperparams):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_channels = int(num_channels)
        self.embedding_dim = int(embedding_dim)
        self.hyperparams = dict(hyperparams)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (batch, num_channels, seq_len) to (batch, embedding_dim)."""

    @classmethod
    def search_space(cls) -> dict[str, list]:
        """
        Candidate values per hyperparameter, for sweeps.

        Keys must be keyword arguments accepted by ``__init__``. Default: no sweep.
        """
        return {}

    def describe(self) -> str:
        settings = ", ".join(f"{key}={value}" for key, value in sorted(self.hyperparams.items()))
        return f"{extractor_name(type(self))}(embedding_dim={self.embedding_dim}, seq_len={self.seq_len}{', ' + settings if settings else ''})"


_REGISTRY: dict[str, type[FeatureExtractor]] = {}
_discovered = False


def register(name: str):
    """Class decorator registering a FeatureExtractor under ``name``."""
    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, FeatureExtractor)):
            raise TypeError(f"@register('{name}') requires a FeatureExtractor subclass, got {cls!r}.")
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Extractor name '{name}' is already registered by {existing.__module__}.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def _discover() -> None:
    """
    Import every module in ``model.extractors`` so decorators run.

    Imported lazily: extractor modules import from ``model``, so eager discovery at
    module import time would create a cycle.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True  # set first, so a failing import cannot cause repeated retries
    import extractors  # noqa: F401  (its __init__ imports each sibling module)


def available() -> list[str]:
    """Names of all registered extractors, sorted."""
    _discover()
    return sorted(_REGISTRY)


def get(name: str) -> type[FeatureExtractor]:
    """Look up an extractor class by name."""
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown extractor '{name}'. Available: {', '.join(sorted(_REGISTRY)) or '(none)'}") from None


def extractor_name(cls: type) -> str:
    """Reverse lookup: registered name for a class, falling back to its class name."""
    for name, registered in _REGISTRY.items():
        if registered is cls:
            return name
    return cls.__name__


def search_space(name: str) -> dict[str, list]:
    return get(name).search_space()


def create(
    name: str,
    seq_len: int,
    num_channels: int = 7,
    embedding_dim: int = 128,
    hyperparams: dict | None = None,
) -> FeatureExtractor:
    """
    Instantiate a registered extractor.

    Unknown hyperparameters are rejected rather than ignored: a silently dropped
    setting would make a sweep report results for a configuration it never ran.
    """
    cls = get(name)
    hyperparams = dict(hyperparams or {})

    accepted = inspect.signature(cls.__init__).parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values())
    if not takes_kwargs:
        unknown = sorted(set(hyperparams) - set(accepted))
        if unknown:
            tunable = sorted(k for k in accepted if k not in {"self", "seq_len", "num_channels", "embedding_dim"})
            raise TypeError(
                f"Extractor '{name}' does not accept {unknown}. Accepted hyperparameters: {tunable or '(none)'}."
            )

    return cls(seq_len=seq_len, num_channels=num_channels, embedding_dim=embedding_dim, **hyperparams)


def check_output_contract(extractor: FeatureExtractor, batch_size: int = 2) -> torch.Tensor:
    """
    Run one dummy forward pass and verify the output contract.

    Cheap enough to call after building a new extractor, and used by the contract
    tests to validate every registered extractor including newly added ones.
    """
    device = next(extractor.parameters(), torch.zeros(1)).device
    dummy = torch.randn(batch_size, extractor.num_channels, extractor.seq_len, device=device)

    was_training = extractor.training
    extractor.eval()
    try:
        with torch.no_grad():
            output = extractor(dummy)
    finally:
        extractor.train(was_training)

    name = extractor_name(type(extractor))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Extractor '{name}' returned {type(output).__name__}, expected a Tensor.")
    expected = (batch_size, extractor.embedding_dim)
    if tuple(output.shape) != expected:
        raise ValueError(f"Extractor '{name}' returned shape {tuple(output.shape)}, expected {expected}.")
    if not torch.isfinite(output).all():
        raise ValueError(f"Extractor '{name}' produced non-finite values.")
    return output
