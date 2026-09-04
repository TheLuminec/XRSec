"""The BOXRR converter's device selection and frame-layout guards."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prepare_boxrr import find_hmd_device, _device_axis_offsets

pytestmark = pytest.mark.unit

SIX_DOF = ["x", "y", "z", "i", "j", "k", "1"]


def _devices():
    return [
        {"name": "Oculus Quest2", "type": "HMD", "joint": "HEAD", "axes": SIX_DOF},
        {"name": "left", "type": "CONTROLLER", "joint": "LEFT_HAND", "axes": SIX_DOF},
        {"name": "right", "type": "CONTROLLER", "joint": "RIGHT_HAND", "axes": SIX_DOF},
    ]


def test_the_head_is_found_by_type_not_by_name():
    """
    20 real users produced 8 distinct HMD name strings, including both "Oculus Quest 2"
    and "Oculus Quest2", plus "Rift_S" and "Unknown". A name-based selector would have
    worked on whichever headset was tested first.
    """
    for name in ("Oculus Quest 2", "Oculus Quest2", "Rift_S", "Unknown", "", None):
        devices = _devices()
        devices[0]["name"] = name
        device, offset = find_hmd_device(devices)
        assert device is devices[0] and offset == 0


def test_the_head_is_found_when_it_is_not_first():
    devices = _devices()
    devices.append(devices.pop(0))            # head last
    device, offset = find_hmd_device(devices)
    assert device["type"] == "HMD"
    assert offset == 14, "offset must count the devices that precede it"


def test_a_recording_with_no_head_is_reported_not_guessed():
    """Tilt Brush recordings carry a single BRUSH device and no HMD."""
    device, offset = find_hmd_device(
        [{"name": "brush", "type": "OTHER", "joint": "BRUSH", "axes": SIX_DOF}])
    assert device is None and offset is None


def test_offsets_account_for_devices_with_different_axis_counts():
    devices = [
        {"type": "TRACKER", "axes": ["x", "y", "z"]},
        {"type": "HMD", "joint": "HEAD", "axes": SIX_DOF},
    ]
    device, offset = find_hmd_device(devices)
    assert offset == 3


def test_offsets_are_cumulative_over_the_device_list():
    offsets = [start for _, start in _device_axis_offsets(_devices())]
    assert offsets == [0, 7, 14]
