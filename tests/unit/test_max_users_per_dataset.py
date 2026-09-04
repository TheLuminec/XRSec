"""max_users as a per-dataset mapping, and the semantics tiers."""
import os

import pytest

from dataset import dataset_tier, select_user_subset


pytestmark = pytest.mark.unit


def _corpus(tmp_path, sizes):
    roots = []
    for name, count in sizes.items():
        root = tmp_path / name / "users"
        for user in range(count):
            (root / str(user)).mkdir(parents=True)
        roots.append(str(root))
    return roots


def _count(kept, name):
    return sum(1 for u in kept if os.path.basename(os.path.dirname(os.path.dirname(u))) == name)


def test_a_mapping_caps_only_the_named_dataset(tmp_path):
    """
    An identity-count curve over BOXRR needs alyx whole at every point: proportional
    apportionment of 419 across 2020 + 76 would leave 15 alyx users.
    """
    roots = _corpus(tmp_path, {"Big": 100, "Small": 20})
    kept = select_user_subset(roots, max_users={"Big": 30}, seed=7)
    assert _count(kept, "Big") == 30 and _count(kept, "Small") == 20


def test_a_mapping_is_a_no_op_when_nothing_is_actually_capped(tmp_path):
    roots = _corpus(tmp_path, {"Big": 10, "Small": 5})
    assert select_user_subset(roots, max_users={"Big": 10}, seed=1) is None
    assert select_user_subset(roots, max_users={}, seed=1) is None


def test_a_mapping_refuses_an_unknown_dataset_name(tmp_path):
    """A misspelt name would silently cap nothing and train on the full corpus."""
    roots = _corpus(tmp_path, {"Big": 10})
    with pytest.raises(ValueError, match="not in data_dirs"):
        select_user_subset(roots, max_users={"Bgi": 5}, seed=1)


def test_a_mapping_is_deterministic_and_seed_dependent(tmp_path):
    roots = _corpus(tmp_path, {"Big": 40, "Small": 4})
    assert select_user_subset(roots, {"Big": 8}, seed=3) == select_user_subset(roots, {"Big": 8}, seed=3)
    assert select_user_subset(roots, {"Big": 8}, seed=3) != select_user_subset(roots, {"Big": 8}, seed=4)


def test_an_integer_still_apportions_proportionally(tmp_path):
    """The original behaviour is untouched."""
    roots = _corpus(tmp_path, {"Big": 100, "Small": 20})
    kept = select_user_subset(roots, max_users=60, seed=7)
    assert _count(kept, "Big") == 50 and _count(kept, "Small") == 10


def test_dataset_tiers_cover_the_audited_corpus_and_admit_ignorance():
    assert dataset_tier("BOXRR-23_Dataset") == 1
    assert dataset_tier("360-degree_Saliency_Dataset_(PanoSaliency)") == 2
    assert dataset_tier("EyeNavGS_6-DoF_Navigation_Dataset") == 3
    assert dataset_tier("Nymeria_v1_2026") == 1, "prefix match for versioned converter output"
    assert dataset_tier("SomethingNobodyAudited") is None
