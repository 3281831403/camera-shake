"""Red tests for the camera-shake bugfix task.

A camera is ``{"pos": [x, y, z], "rot": [pitch, yaw, roll]}`` with angles in
degrees.  No test sleeps, calls the network, or depends on global randomness.
"""

from __future__ import annotations

import math

import pytest

from shake.trauma import (
    ShakeError,
    add_trauma,
    apply,
    decay,
    limit,
    noise_1d,
    offset,
    rotation,
    shake_batch,
)


def _camera() -> dict:
    return {"pos": [10.0, 20.0, 30.0], "rot": [0.0, 90.0, 0.0]}


def _config() -> dict:
    return {
        "half_life": 1.0,
        "seed": 20261119,
        "max_offset": 0.5,
        "max_deg": 10.0,
        "camera": _camera(),
        "space": "local",
    }


def test_add_trauma_clamps_to_unit_interval():
    assert add_trauma(0.2, 0.3) == pytest.approx(0.5, abs=1e-6)
    assert add_trauma(0.8, 0.5) == 1.0
    assert add_trauma(0.95, 1.0) == 1.0
    assert add_trauma(0.1, -0.5) == 0.0


def test_decay_uses_exponential_half_life():
    assert decay(0.8, 0.0, 1.0) == pytest.approx(0.8, abs=1e-6)
    assert decay(0.8, 1.0, 1.0) == pytest.approx(0.4, abs=1e-6)
    assert decay(0.8, 1.5, 0.5) == pytest.approx(0.1, abs=1e-6)
    assert decay(0.8, 0.5, 1.0) > decay(0.8, 1.0, 1.0)
    assert decay(0.8, 10.0, 0.5) >= 0.0


def test_noise_1d_is_deterministic_across_calls():
    seed = 20261119
    frequency = 2.0
    first = [noise_1d(seed, index * 0.11, frequency) for index in range(40)]
    second = [noise_1d(seed, index * 0.11, frequency) for index in range(40)]

    assert first == second
    assert all(-1.0 <= value <= 1.0 for value in first)
    assert len(set(first)) > 1
    assert noise_1d(seed, 0.37, frequency) == noise_1d(seed, 0.37, frequency)


def test_offset_maps_trauma_squared_to_amplitude():
    seed = 123456
    t = 0.83
    small = offset(0.5, seed, t, 2.0)
    full = offset(1.0, seed, t, 2.0)

    assert small == pytest.approx([value * 0.25 for value in full], abs=1e-6)


def test_offset_uses_independent_axes():
    samples = [offset(1.0, 777, index * 0.41, 1.0) for index in range(12)]

    assert all(len(sample) == 3 for sample in samples)
    assert any(
        abs(sample[0] - sample[1]) > 1e-9 or abs(sample[1] - sample[2]) > 1e-9
        for sample in samples
    )


def test_rotation_returns_degrees_not_radians():
    max_deg = 30.0
    magnitudes = [
        abs(component)
        for index in range(128)
        for component in rotation(1.0, 5150, index * 0.137, max_deg)
    ]

    assert max(magnitudes) <= max_deg + 1e-6
    assert max(magnitudes) > 2.0


def test_apply_rotates_world_offset_by_camera_orientation():
    camera = _camera()
    local = apply(camera, [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], "local")
    world = apply(camera, [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], "world")

    assert local["pos"] == pytest.approx([11.0, 20.0, 30.0], abs=1e-6)
    assert world["pos"] == pytest.approx([10.0, 20.0, 29.0], abs=1e-6)
    assert local["rot"] == pytest.approx([5.0, 90.0, 0.0], abs=1e-6)
    assert world["rot"] == pytest.approx([5.0, 90.0, 0.0], abs=1e-6)
    assert camera == _camera()


def test_limit_clamps_pitch_and_roll_but_not_yaw():
    camera = {"pos": [1.0, 2.0, 3.0], "rot": [150.0, 42.0, -150.0]}

    limited = limit(camera, 120.0, 30.0)

    assert limited["rot"] == pytest.approx([120.0, 42.0, -30.0], abs=1e-6)
    assert limited["pos"] == camera["pos"]
    assert camera["rot"] == [150.0, 42.0, -150.0]


def test_shake_batch_accumulates_trauma_across_events():
    config = _config()
    events = [{"amount": 0.4}, {"amount": 0.4}, {"amount": 0.4}]

    result = shake_batch(events, 0.0, config)

    assert result["trauma"] == pytest.approx([0.4, 0.8, 1.0], abs=1e-6)
    assert len(result["frames"]) == 3


def test_shake_batch_rolls_back_on_invalid_event():
    config = _config()
    events = [{"amount": 0.2}, {"amount": math.nan}, {"amount": 0.3}]

    with pytest.raises(ShakeError):
        shake_batch(events, 0.0, config)

    assert config["camera"] == _camera()


def test_zero_trauma_produces_zero_offset_and_rotation():
    assert offset(0.0, 7, 2.5, 5.0) == [0.0, 0.0, 0.0]
    assert rotation(0.0, 7, 2.5, 30.0) == [0.0, 0.0, 0.0]


def test_same_seed_replays_identical_sequences():
    first: list[float] = []
    second: list[float] = []
    for index in range(12):
        t = index * 0.23
        first.extend(offset(0.65, 4242, t, 1.5))
        first.extend(rotation(0.65, 4242, t, 20.0))
        second.extend(offset(0.65, 4242, t, 1.5))
        second.extend(rotation(0.65, 4242, t, 20.0))

    assert first == second
    assert any(abs(value) > 1e-9 for value in first)


def test_decay_rejects_non_positive_half_life():
    for half_life in (0.0, -2.5):
        with pytest.raises(ShakeError):
            decay(0.5, 0.1, half_life)


def test_decay_to_zero_stays_zero():
    fully_decayed = decay(1.0, 50.0, 1.0)

    assert fully_decayed == 0.0
    assert decay(fully_decayed, 5.0, 1.0) == 0.0
    assert decay(0.0, 0.0, 1.0) == 0.0


def test_shake_batch_preserves_event_order():
    config = _config()
    events = [{"amount": 0.2}, {"amount": 0.5}, {"amount": 0.1}]

    result = shake_batch(events, 0.0, config)

    assert result["trauma"] == pytest.approx([0.2, 0.7, 0.8], abs=1e-6)
    assert len(result["frames"]) == len(events)
    assert all("pos" in frame and "rot" in frame for frame in result["frames"])
