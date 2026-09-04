"""Everything written into a checkpoint has to survive the weights-only unpickler."""
import pytest
import torch
from omegaconf import OmegaConf

from train import plain_container


pytestmark = pytest.mark.unit


def test_a_dictconfig_becomes_a_plain_dict_that_torch_can_reload(tmp_path):
    cfg = OmegaConf.create({"max_users": {"BOXRR-23_Dataset": 343}, "dirs": ["a", "b"]})
    payload = {"max_users": plain_container(cfg.max_users), "dirs": plain_container(cfg.dirs)}
    assert payload == {"max_users": {"BOXRR-23_Dataset": 343}, "dirs": ["a", "b"]}
    assert type(payload["max_users"]) is dict

    path = tmp_path / "ck.pth"
    torch.save(payload, path)
    assert torch.load(path, weights_only=True) == payload      # the strict loader accepts it


def test_scalars_and_none_pass_through():
    assert plain_container(None) is None
    assert plain_container(48) == 48
    assert plain_container("x") == "x"


def test_the_strict_loader_would_have_refused_the_raw_container(tmp_path):
    """The failure this guards against, pinned so it cannot quietly return."""
    cfg = OmegaConf.create({"max_users": {"BOXRR-23_Dataset": 343}})
    path = tmp_path / "bad.pth"
    torch.save({"max_users": cfg.max_users}, path)
    with pytest.raises(Exception, match="weights_only|Unsupported global"):
        torch.load(path, weights_only=True)
