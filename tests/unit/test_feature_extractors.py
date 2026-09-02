"""
Contract tests applied to every registered feature extractor.

These are parametrized over the registry, so a new extractor dropped into
model/extractors/ is validated automatically with no edits here. An extractor that
passes these is safe to slot into training and sweeps.
"""

import inspect

import pytest
import torch

import feature_extractor as fe
from model import SiameseModel, create_model


ALL_EXTRACTORS = fe.available()


def test_registry_is_not_empty():
    assert ALL_EXTRACTORS, "No extractors were discovered under model/extractors/."


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_output_contract(name):
    """(batch, channels, seq_len) -> (batch, embedding_dim), finite."""
    extractor = fe.create(name, seq_len=10, embedding_dim=16)
    output = fe.check_output_contract(extractor, batch_size=3)
    assert output.shape == (3, 16)


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
@pytest.mark.parametrize("seq_len,embedding_dim", [(10, 8), (20, 32), (50, 64)])
def test_handles_varied_shapes(name, seq_len, embedding_dim):
    """Window length and embedding width are swept, so both must be honoured."""
    extractor = fe.create(name, seq_len=seq_len, embedding_dim=embedding_dim)
    assert extractor.seq_len == seq_len
    assert extractor.embedding_dim == embedding_dim
    fe.check_output_contract(extractor, batch_size=2)


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_search_space_keys_are_real_constructor_arguments(name):
    """
    A sweep key the constructor does not accept would silently produce duplicate
    runs of the default configuration, so mismatches must fail loudly here.
    """
    cls = fe.get(name)
    accepted = set(inspect.signature(cls.__init__).parameters)
    space = cls.search_space()

    assert isinstance(space, dict)
    for key, values in space.items():
        assert key in accepted, f"'{name}' sweeps '{key}', which __init__ does not accept."
        assert isinstance(values, (list, tuple)) and values, f"'{name}' sweep for '{key}' must be a non-empty list."


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_every_search_space_value_builds_and_runs(name):
    """Each candidate value must produce a working extractor, one axis at a time."""
    for key, values in fe.search_space(name).items():
        for value in values:
            extractor = fe.create(name, seq_len=10, embedding_dim=8, hyperparams={key: value})
            assert extractor.hyperparams[key] == value
            fe.check_output_contract(extractor, batch_size=2)


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_declared_determinism_matches_behaviour(name):
    """
    An extractor claiming determinism must return identical embeddings for identical
    input in eval mode. Catches dropout left active at eval, or unseeded noise, which
    would otherwise surface as unexplained variance between sweep runs.
    """
    extractor = fe.create(name, seq_len=10, embedding_dim=16)
    extractor.eval()
    x = torch.randn(3, 7, 10)
    with torch.no_grad():
        first, second = extractor(x), extractor(x)

    identical = torch.allclose(first, second, atol=1e-6)
    if fe.get(name).deterministic:
        assert identical, (
            f"'{name}' declares deterministic = True but produced different embeddings "
            "for the same input in eval mode. Fix the non-determinism, or set "
            "`deterministic = False` on the class if it is intentional."
        )
    else:
        assert not identical, (
            f"'{name}' declares deterministic = False but is in fact deterministic; "
            "remove the flag so it gets the stronger checks."
        )


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_records_hyperparameters_for_checkpointing(name):
    extractor = fe.create(name, seq_len=10, embedding_dim=8)
    assert isinstance(extractor.hyperparams, dict)
    assert fe.extractor_name(type(extractor)) == name
    assert name in extractor.describe()


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_slots_into_the_siamese_model(name):
    model, criterion, optimizer = create_model(
        embedding_dim=8, seq_len=10, lr=0.01, device=torch.device("cpu"), extractor=name
    )
    assert isinstance(model, SiameseModel)

    x1, x2 = torch.randn(4, 7, 10), torch.randn(4, 7, 10)
    logits = model(x1, x2)
    assert logits.shape == (4, 1)

    # One optimisation step must actually update something. Which parameters exist
    # depends on the extractor: a baseline like `random` can legitimately have none of
    # its own at default settings, in which case only the Siamese head learns.
    extractor_before = [p.detach().clone() for p in model.feature_extractor.parameters()]
    head_before = [p.detach().clone() for p in model.classifier.parameters()]

    loss = criterion(logits, torch.tensor([[1.0], [0.0], [1.0], [0.0]]))
    loss.backward()
    optimizer.step()

    if extractor_before:
        after = list(model.feature_extractor.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(extractor_before, after)), (
            f"'{name}' has parameters but none of them changed after an optimiser step; "
            "gradients are not reaching the extractor."
        )
    else:
        head_after = list(model.classifier.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(head_before, head_after)), (
            f"'{name}' exposes no parameters of its own, and the Siamese head did not "
            "train either, so this configuration cannot learn at all."
        )


def test_unknown_extractor_name_is_rejected():
    with pytest.raises(KeyError, match="Unknown extractor"):
        fe.create("no_such_extractor", seq_len=10)


def test_unknown_hyperparameter_is_rejected():
    """Silently ignoring a bad key would make a sweep report untrue configurations."""
    with pytest.raises(TypeError, match="does not accept"):
        fe.create(ALL_EXTRACTORS[0], seq_len=10, hyperparams={"definitely_not_a_param": 3})


def test_register_rejects_non_extractor():
    with pytest.raises(TypeError):
        fe.register("bad")(object)


@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_checkpoint_round_trip_rebuilds_the_same_extractor(name, tmp_path):
    """Saving and loading must restore the extractor identity and its hyperparameters."""
    from utils import load_checkpoint, save_checkpoint

    space = fe.search_space(name)
    # Pick a non-default value per axis, so defaults can't mask a lost setting.
    params = {key: values[-1] for key, values in space.items()}

    device = torch.device("cpu")
    model, _, optimizer = create_model(
        embedding_dim=8, seq_len=10, lr=0.01, device=device,
        extractor=name, extractor_params=params,
    )
    path = tmp_path / f"{name}.pth"
    save_checkpoint(path, model, optimizer, epoch=1)

    restored = load_checkpoint(str(path), device, seq_len=10)

    assert fe.extractor_name(type(restored.feature_extractor)) == name
    for key, value in params.items():
        assert restored.feature_extractor.hyperparams[key] == value

    # Weights are the ground truth for a round trip, and this holds for stochastic
    # extractors too.
    original_state = model.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], restored_state[key], atol=1e-6)

    # Identical weights only imply identical outputs when forward is a pure function.
    if fe.get(name).deterministic:
        x1, x2 = torch.randn(2, 7, 10), torch.randn(2, 7, 10)
        model.eval()
        restored.eval()
        with torch.no_grad():
            assert torch.allclose(model(x1, x2), restored(x1, x2), atol=1e-6)


# --- channel sets -------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_EXTRACTORS)
def test_non_default_channel_count_either_works_or_fails_clearly(name):
    """
    channels=position feeds extractors 3 channels instead of 7. An extractor that
    assumes the quaternion+position layout must say so by name, not die with an
    IndexError from somewhere inside its forward pass.
    """
    try:
        extractor = fe.create(name, seq_len=20, num_channels=3, embedding_dim=16)
    except ValueError as exc:
        message = str(exc)
        assert name in message
        assert "num_channels=3" in message
        assert "channels=full" in message, "the error must say what to do instead"
        return

    assert extractor.num_channels == 3
    output = fe.check_output_contract(extractor, batch_size=2)
    assert output.shape == (2, 16)


def test_a_channel_agnostic_extractor_actually_runs_on_three_channels():
    """At least one registered extractor must support position-only data."""
    extractor = fe.create("bilstm", seq_len=20, num_channels=3, embedding_dim=16)
    fe.check_output_contract(extractor, batch_size=2)


def test_seven_channel_creation_skips_the_probe(monkeypatch):
    """The validation forward pass must not be paid on ordinary 7-channel runs."""
    calls = []
    monkeypatch.setattr(fe, "check_output_contract", lambda *a, **k: calls.append(1))
    fe.create("bilstm", seq_len=20, num_channels=7, embedding_dim=8)
    assert calls == []
