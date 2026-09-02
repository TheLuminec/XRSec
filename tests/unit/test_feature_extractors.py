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

    # One optimisation step must actually update the extractor's parameters.
    before = [p.detach().clone() for p in model.feature_extractor.parameters()]
    loss = criterion(logits, torch.tensor([[1.0], [0.0], [1.0], [0.0]]))
    loss.backward()
    optimizer.step()
    after = list(model.feature_extractor.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


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

    x1, x2 = torch.randn(2, 7, 10), torch.randn(2, 7, 10)
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x1, x2), restored(x1, x2), atol=1e-6)
