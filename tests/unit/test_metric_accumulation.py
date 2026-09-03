"""
Metrics accumulate on-device and are read once per epoch.

Calling .item() per batch forces a device sync every step, so the GPU idles waiting
for the CPU instead of queueing the next batch. Measured on a 3050 Ti at batch 256:
25.4 ms/step with per-batch .item(), 11.1 ms/step without - 2.30x, for arithmetic that
has to stay identical. These tests pin the "identical" half.
"""
import pytest
import torch

pytestmark = pytest.mark.unit


def _old_way(losses, sizes):
    """What the loops used to do: a Python float64 running total."""
    total = 0.0
    for loss, size in zip(losses, sizes):
        total += float(loss) * size
    return total


def _new_way(losses, sizes, device="cpu"):
    """What they do now: a device tensor read once at the end."""
    total = torch.zeros((), device=device)
    for loss, size in zip(losses, sizes):
        total += torch.tensor(loss, device=device) * size
    return float(total)


def test_loss_accumulation_matches_the_python_float_version():
    losses = [5.2837, 4.9912, 4.7715, 4.5003, 4.2210] * 40
    sizes = [256] * len(losses)
    assert _new_way(losses, sizes) == pytest.approx(_old_way(losses, sizes), rel=1e-5)


def test_accumulation_stays_accurate_over_a_long_epoch():
    """
    A float32 accumulator loses precision as the running total grows. At 2419
    identities an epoch is thousands of batches, so this is the case that matters.
    """
    losses = [3.14159] * 5000
    sizes = [1024] * 5000
    assert _new_way(losses, sizes) == pytest.approx(_old_way(losses, sizes), rel=1e-4)


def test_integer_counts_are_exact():
    """
    Correct-prediction counts run to ~10^6. float32 represents integers exactly only
    to 2^24, so this is the one that would break first if counts got large enough.
    """
    total = torch.zeros((), device="cpu")
    for _ in range(4000):
        total += torch.tensor(512)
    assert int(total) == 4000 * 512


def test_a_boolean_comparison_sums_the_same_either_way():
    torch.manual_seed(0)
    predicted = torch.randint(0, 2, (4096,)).float()
    labels = torch.randint(0, 2, (4096,)).float()

    old = int((predicted == labels).sum().item())
    new = torch.zeros((), device="cpu")
    for start in range(0, 4096, 256):
        chunk = slice(start, start + 256)
        new += (predicted[chunk] == labels[chunk]).sum()
    assert int(new) == old


def test_detach_keeps_the_accumulator_out_of_the_graph():
    """
    Accumulating a tensor that still carries grad_fn would retain the whole epoch's
    graph and run the GPU out of memory - the failure this change could plausibly have
    introduced.
    """
    weight = torch.ones(4, requires_grad=True)
    total = torch.zeros(())
    for _ in range(3):
        loss = (weight * 2).sum()
        total += loss.detach() * 8
    assert total.grad_fn is None
    assert not total.requires_grad
