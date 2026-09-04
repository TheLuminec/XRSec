"""An evaluation corpus the normaliser never saw follows a named policy, and says so."""
import pytest
import torch

from normalization import ChannelNormalizer


pytestmark = pytest.mark.unit


class _Index:
    """Minimal SampleIndex stand-in with the per-user/session provenance session needs."""

    def __init__(self, samples, dataset_names, users=None, sessions=None):
        self.samples = samples
        self.sample_count = int(samples.shape[0])
        self.dataset_names = list(dataset_names)
        self.window_dataset_ids = torch.zeros(self.sample_count, dtype=torch.long)
        users = users or [list(range(self.sample_count))]
        self.user_dataset_ids = [0] * len(users)
        self.user_sample_indices = [torch.as_tensor(w, dtype=torch.long) for w in users]
        self.num_users = len(users)
        self.window_session_ids = torch.as_tensor(sessions if sessions is not None
                                                  else [0] * self.sample_count, dtype=torch.long)


def _trained_then_unseen():
    torch.manual_seed(0)
    trained = _Index(torch.randn(50, 7, 12) * 2.0 + 5.0, ["Seen"])
    unseen = _Index(torch.randn(40, 7, 12) * 3.0 - 7.0, ["Unseen"],
                    users=[list(range(0, 20)), list(range(20, 40))],
                    sessions=[0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
    return ChannelNormalizer("per_dataset").fit(trained), unseen


def test_target_fit_standardises_on_the_evaluation_data_and_records_it():
    normalizer, unseen = _trained_then_unseen()
    normalizer.transform(unseen)
    assert unseen.samples.mean().abs().item() < 0.1
    assert unseen.samples.std().item() == pytest.approx(1.0, abs=0.1)
    assert normalizer.unseen_datasets == {"Unseen": "target_fit"}
    assert "Unseen=target_fit" in normalizer.describe()


def test_none_leaves_the_data_untouched_but_still_records_it():
    normalizer, unseen = _trained_then_unseen()
    normalizer.unseen = "none"
    before = unseen.samples.clone()
    normalizer.transform(unseen)
    assert torch.equal(unseen.samples, before)
    assert normalizer.unseen_datasets == {"Unseen": "none"}


def test_session_standardises_each_session_by_itself():
    """Every (user, session) block ends up centred: the absolute offset is gone."""
    normalizer, unseen = _trained_then_unseen()
    normalizer.unseen = "session"
    unseen.samples[:10] += 40.0                      # one session sits far away
    normalizer.transform(unseen)
    for start in (0, 10, 20, 30):
        block = unseen.samples[start:start + 10]
        assert block.mean().abs().item() < 0.15
        assert block.std().item() == pytest.approx(1.0, abs=0.15)
    assert normalizer.unseen_datasets == {"Unseen": "session"}


def test_a_seen_dataset_is_never_touched_by_the_unseen_policy():
    normalizer, _ = _trained_then_unseen()
    normalizer.unseen = "none"
    seen_again = _Index(torch.randn(20, 7, 12) * 2.0 + 5.0, ["Seen"])
    normalizer.transform(seen_again)
    assert seen_again.samples.mean().abs().item() < 0.3
    assert normalizer.unseen_datasets == {}


def test_the_policy_round_trips_through_the_checkpoint_and_can_be_overridden():
    normalizer, _ = _trained_then_unseen()
    normalizer.unseen = "session"
    state = normalizer.state_dict()
    assert ChannelNormalizer.from_state(state).unseen == "session"
    assert ChannelNormalizer.from_state(state, unseen="none").unseen == "none"
    with pytest.raises(ValueError, match="eval_normalize"):
        ChannelNormalizer("per_dataset", unseen="cohort")
