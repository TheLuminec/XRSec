"""Early stopping on the validation-selected metric."""
import inspect

import pytest

pytestmark = pytest.mark.unit


def _stops_at(scores, patience):
    """
    Mirror of the epoch loop's bookkeeping: reset the counter on improvement, stop once
    it exceeds patience. Kept as a unit so the rule is pinned without training anything.
    """
    best = float("-inf")
    without = 0
    for epoch, score in enumerate(scores, start=1):
        if score > best:
            best = score
            without = 0
        without += 1
        if patience and without > patience:
            return epoch
    return len(scores)


def test_a_run_that_peaks_early_stops_early():
    """Median validation-selected epoch across 304 recorded runs is 7 of 20."""
    scores = [0.5, 0.6, 0.7] + [0.65] * 17
    assert _stops_at(scores, patience=5) < 20


def test_a_run_still_improving_is_not_cut_off():
    """
    p90 of the selected epoch is 18 and 5% select epoch 19 or 20, so a fixed budget
    censors a real tail. Patience has to let those keep going - that is the half a
    plain truncation gets wrong.
    """
    scores = [0.5 + 0.01 * i for i in range(20)]
    assert _stops_at(scores, patience=5) == 20


def test_patience_zero_disables_it():
    """Default behaviour must be unchanged, or every recorded run stops comparing."""
    scores = [0.9] + [0.1] * 19
    assert _stops_at(scores, patience=0) == 20


def test_the_stop_comes_after_patience_epochs_without_improvement():
    scores = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Improvement at epoch 1, then flat; patience 3 stops at epoch 4.
    assert _stops_at(scores, patience=3) == 4


def test_a_late_peak_resets_the_counter():
    scores = [0.5, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.4]
    assert _stops_at(scores, patience=3) == 7


def test_run_training_exposes_the_parameter():
    from train import run_training

    assert "early_stopping_patience" in inspect.signature(run_training).parameters
