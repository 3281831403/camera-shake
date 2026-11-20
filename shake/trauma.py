"""Deterministic camera-shake primitives.

Trauma is a scalar in ``[0.0, 1.0]``.  Camera rotations are expressed in
degrees, and a camera is a mapping with::

    {"pos": [x, y, z], "rot": [pitch, yaw, roll]}

Floating-point assertions use an absolute tolerance of ``1e-6`` unless a
function's docstring says otherwise.  Nothing here sleeps, uses global random
state, or depends on the wall clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "ShakeError",
    "add_trauma",
    "decay",
    "noise_1d",
    "offset",
    "rotation",
    "apply",
    "limit",
    "shake_batch",
]


class ShakeError(ValueError):
    """Raised when camera-shake input is invalid or a batch is rejected."""


_MASK64 = (1 << 64) - 1
_FLUSH = 1e-12
_LOCAL = "local"
_WORLD = "world"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _number(value: object, name: str) -> float:
    """Return ``value`` as a finite float or raise :class:`ShakeError`."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShakeError(f"{name} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ShakeError(f"{name} must be finite, got {value!r}")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShakeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _triple(value: object, name: str) -> list[float]:
    """Return a three-element list of finite floats or raise ShakeError."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ShakeError(f"{name} must be a sequence of three numbers")
    if len(value) != 3:
        raise ShakeError(f"{name} must contain exactly three numbers")
    return [_number(component, f"{name}[{index}]") for index, component in enumerate(value)]


def _camera_parts(camera: object) -> tuple[list[float], list[float]]:
    if not isinstance(camera, Mapping):
        raise ShakeError("camera must be a mapping with 'pos' and 'rot'")
    if "pos" not in camera or "rot" not in camera:
        raise ShakeError("camera must contain both 'pos' and 'rot'")
    return _triple(camera["pos"], "camera['pos']"), _triple(camera["rot"], "camera['rot']")


def _clamp_trauma(trauma: float) -> float:
    return min(1.0, max(0.0, trauma))


# ---------------------------------------------------------------------------
# Deterministic hash-based value noise
# ---------------------------------------------------------------------------


def _mix64(value: int) -> int:
    """SplitMix64 finaliser; deterministic for any integer input."""

    value &= _MASK64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK64
    value ^= value >> 31
    return value


def _lattice(seed: int, index: int) -> float:
    """Hash-based lattice value in ``[-1.0, 1.0]`` for a seed and integer cell."""

    mixed = _mix64(seed * 0x9E3779B97F4A7C15 + index * 0xD1B54A32D192ED03)
    return (mixed / float(_MASK64)) * 2.0 - 1.0


def _derive_seed(seed: int, salt: int, axis: int) -> int:
    """Derive a per-axis seed so axes never share a noise stream."""

    return _mix64(
        seed * 0x9E3779B97F4A7C15
        + salt * 0xC2B2AE3D27D4EB4F
        + axis * 0x165667B19E3779F9
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_trauma(current: float, amount: float) -> float:
    """Add ``amount`` to ``current`` and clamp the result to ``[0.0, 1.0]``.

    Trauma is never allowed to overflow above ``1.0``; a negative ``amount``
    reduces trauma but the result never drops below ``0.0``.  Both arguments
    are finite numbers.  The return value is a float and is compared with a
    tolerance of ``1e-6``.
    """

    base = _number(current, "current")
    delta = _number(amount, "amount")
    return _clamp_trauma(base + delta)


def decay(trauma: float, dt: float, half_life: float) -> float:
    """Return trauma after ``dt`` seconds of exponential half-life decay.

    ``half_life`` is the time in seconds for trauma to halve, so the result is
    ``trauma * 0.5 ** (dt / half_life)``.  ``dt`` may be zero and is never
    negative.  The result never becomes negative; values that decay below
    ``1e-12`` are flushed to ``0.0`` so a fully decayed shake stays at zero.

    ``half_life`` must be positive; ``half_life <= 0`` raises
    :class:`ShakeError`.  Results are compared with a tolerance of ``1e-6``.
    """

    start = _number(trauma, "trauma")
    elapsed = _number(dt, "dt")
    if elapsed < 0.0:
        raise ShakeError(f"dt must be non-negative, got {dt!r}")
    span = _number(half_life, "half_life")
    if span <= 0.0:
        raise ShakeError(f"half_life must be positive, got {half_life!r}")

    result = start * (0.5 ** (elapsed / span))
    if result < 0.0:
        result = 0.0
    if result < _FLUSH:
        result = 0.0
    return result


def noise_1d(seed: int, t: float, frequency: float) -> float:
    """Return deterministic one-dimensional value noise in ``[-1.0, 1.0]``.

    The value depends only on ``seed``, ``t`` and ``frequency``: the same
    inputs always produce exactly the same result, both within one run and
    across replays.  No module-level or global random state is consulted.
    ``frequency`` is in Hz and must be positive.

    The curve interpolates hash-based lattice values, is continuous in ``t``,
    and varies with ``t`` rather than being constant.  Invalid ``frequency``
    raises :class:`ShakeError`.  Results are compared with a tolerance of
    ``1e-6``.
    """

    key = _integer(seed, "seed")
    time = _number(t, "t")
    rate = _number(frequency, "frequency")
    if rate <= 0.0:
        raise ShakeError(f"frequency must be positive, got {frequency!r}")

    position = time * rate
    cell = math.floor(position)
    fraction = position - cell

    left = _lattice(key, cell)
    right = _lattice(key, cell + 1)
    # Smoothstep interpolation keeps the curve continuous (C1 at the lattice).
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    return left + (right - left) * blend


def offset(
    trauma: float,
    seed: int,
    t: float,
    max_offset: float,
) -> list[float]:
    """Map trauma to a three-dimensional camera-local displacement.

    The scalar amplitude is ``clamp(trauma, 0.0, 1.0) ** 2 * max_offset``;
    squaring makes small trauma settle gently instead of shaking linearly.
    The returned vector is ``[x, y, z]`` where every component is the
    amplitude multiplied by an independent deterministic noise sample.  The
    three axes therefore do not share one sample and a shake is not forced to
    lie on the diagonal.

    Noise is obtained from :func:`noise_1d` with distinct derived seeds and a
    fixed base frequency of ``1.0``.  ``max_offset`` must be non-negative and
    invalid input raises :class:`ShakeError`.  Results are compared with a
    tolerance of ``1e-6``.
    """

    level = _clamp_trauma(_number(trauma, "trauma"))
    key = _integer(seed, "seed")
    time = _number(t, "t")
    amplitude = _number(max_offset, "max_offset")
    if amplitude < 0.0:
        raise ShakeError(f"max_offset must be non-negative, got {max_offset!r}")
    if level == 0.0 or amplitude == 0.0:
        return [0.0, 0.0, 0.0]

    scale = (level * level) * amplitude
    return [
        scale * noise_1d(_derive_seed(key, 1, axis), time, 1.0)
        for axis in range(3)
    ]


def rotation(
    trauma: float,
    seed: int,
    t: float,
    max_deg: float,
) -> list[float]:
    """Map trauma to ``[pitch, yaw, roll]`` shake expressed in degrees.

    The scalar amplitude is ``clamp(trauma, 0.0, 1.0) ** 2 * max_deg``.  Each
    of the three angles is the amplitude multiplied by an independent
    deterministic noise sample.  Values are degrees: no returned angle is
    converted to radians and no radian-to-degree factor is applied.

    Noise is obtained from :func:`noise_1d` with distinct derived seeds and a
    fixed base frequency of ``1.0``.  ``max_deg`` must be non-negative and
    invalid input raises :class:`ShakeError`.  Results are compared with a
    tolerance of ``1e-6``.
    """

    level = _clamp_trauma(_number(trauma, "trauma"))
    key = _integer(seed, "seed")
    time = _number(t, "t")
    amplitude = _number(max_deg, "max_deg")
    if amplitude < 0.0:
        raise ShakeError(f"max_deg must be non-negative, got {max_deg!r}")
    if level == 0.0 or amplitude == 0.0:
        return [0.0, 0.0, 0.0]

    scale = (level * level) * amplitude
    return [
        scale * noise_1d(_derive_seed(key, 2, axis), time, 1.0)
        for axis in range(3)
    ]


def _matmul3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            left[row][0] * right[0][col]
            + left[row][1] * right[1][col]
            + left[row][2] * right[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _vecmul3(matrix: list[list[float]], vector: Sequence[float]) -> list[float]:
    return [
        matrix[row][0] * vector[0]
        + matrix[row][1] * vector[1]
        + matrix[row][2] * vector[2]
        for row in range(3)
    ]


def _orientation_matrix(pitch: float, yaw: float, roll: float) -> list[list[float]]:
    """Return ``Rz(roll) @ Rx(pitch) @ Ry(yaw)`` with angles in degrees."""

    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)

    rx = [
        [1.0, 0.0, 0.0],
        [0.0, math.cos(p), -math.sin(p)],
        [0.0, math.sin(p), math.cos(p)],
    ]
    ry = [
        [math.cos(y), 0.0, math.sin(y)],
        [0.0, 1.0, 0.0],
        [-math.sin(y), 0.0, math.cos(y)],
    ]
    rz = [
        [math.cos(r), -math.sin(r), 0.0],
        [math.sin(r), math.cos(r), 0.0],
        [0.0, 0.0, 1.0],
    ]
    return _matmul3(rz, _matmul3(rx, ry))


def apply(
    camera: dict,
    offset: list[float],
    rotation: list[float],
    space: str,
) -> dict:
    """Apply a displacement and an angular shake to ``camera``.

    The input camera is never mutated.  The returned camera has ``pos`` equal
    to the source position plus the displacement and ``rot`` equal to the
    source rotation plus ``rotation`` component-wise.  All rotations are in
    degrees.

    ``space="local"`` adds the supplied displacement directly in camera-local
    axes.  ``space="world"`` interprets the supplied displacement as
    camera-local and rotates it into world axes with the camera orientation,
    using ``Rz(roll) @ Rx(pitch) @ Ry(yaw)``.  For a camera with
    ``rot=[0, 90, 0]``, a local displacement ``[1, 0, 0]`` therefore becomes
    world ``[0, 0, -1]``.  The angular shake is added in both cases.

    An unsupported ``space`` or a malformed camera raises
    :class:`ShakeError`.  Results are compared with a tolerance of ``1e-6``.
    """

    position, angles = _camera_parts(camera)
    translation = _triple(offset, "offset")
    delta_angles = _triple(rotation, "rotation")

    if space == _LOCAL:
        moved = [position[i] + translation[i] for i in range(3)]
    elif space == _WORLD:
        matrix = _orientation_matrix(angles[0], angles[1], angles[2])
        world = _vecmul3(matrix, translation)
        moved = [position[i] + world[i] for i in range(3)]
    else:
        raise ShakeError(f"unsupported space {space!r}; expected 'local' or 'world'")

    return {
        "pos": moved,
        "rot": [angles[i] + delta_angles[i] for i in range(3)],
    }


def limit(camera: dict, max_pitch: float, max_roll: float) -> dict:
    """Clamp pitch and roll to symmetric limits while preserving yaw.

    Pitch and roll are each clamped to ``[-limit, +limit]``; yaw is copied
    unchanged.  ``max_pitch`` and ``max_roll`` must be non-negative.  The input
    camera is not mutated and the position is preserved in the returned
    camera.  Invalid limits or a malformed camera raise :class:`ShakeError`.
    Results are compared with a tolerance of ``1e-6``.
    """

    position, angles = _camera_parts(camera)
    pitch_limit = _number(max_pitch, "max_pitch")
    roll_limit = _number(max_roll, "max_roll")
    if pitch_limit < 0.0:
        raise ShakeError(f"max_pitch must be non-negative, got {max_pitch!r}")
    if roll_limit < 0.0:
        raise ShakeError(f"max_roll must be non-negative, got {max_roll!r}")

    pitch = min(pitch_limit, max(-pitch_limit, angles[0]))
    roll = min(roll_limit, max(-roll_limit, angles[2]))
    return {
        "pos": list(position),
        "rot": [pitch, angles[1], roll],
    }


def _batch_config(config: object) -> dict:
    """Validate and normalise ``shake_batch`` configuration."""

    if not isinstance(config, Mapping):
        raise ShakeError("config must be a mapping")
    required = ("half_life", "seed", "max_offset", "max_deg", "camera", "space")
    missing = [name for name in required if name not in config]
    if missing:
        raise ShakeError(f"config is missing required keys: {', '.join(missing)}")

    half_life = _number(config["half_life"], "config['half_life']")
    if half_life <= 0.0:
        raise ShakeError(f"config['half_life'] must be positive, got {config['half_life']!r}")
    seed = _integer(config["seed"], "config['seed']")
    max_offset = _number(config["max_offset"], "config['max_offset']")
    if max_offset < 0.0:
        raise ShakeError(
            f"config['max_offset'] must be non-negative, got {config['max_offset']!r}"
        )
    max_deg = _number(config["max_deg"], "config['max_deg']")
    if max_deg < 0.0:
        raise ShakeError(f"config['max_deg'] must be non-negative, got {config['max_deg']!r}")
    camera = _camera_parts(config["camera"])
    space = config["space"]
    if space not in (_LOCAL, _WORLD):
        raise ShakeError(f"config['space'] unsupported: {space!r}")

    return {
        "half_life": half_life,
        "seed": seed,
        "max_offset": max_offset,
        "max_deg": max_deg,
        "camera": {"pos": camera[0], "rot": camera[1]},
        "space": space,
    }


def _batch_events(events: object) -> list[float]:
    """Validate every event up front so a bad event rejects the whole batch."""

    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise ShakeError("events must be a sequence of mappings")
    amounts: list[float] = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ShakeError(f"event {index} must be a mapping")
        if "amount" not in event:
            raise ShakeError(f"event {index} is missing 'amount'")
        amount = _number(event["amount"], f"events[{index}]['amount']")
        if not 0.0 <= amount <= 1.0:
            raise ShakeError(f"events[{index}]['amount'] must be within [0.0, 1.0]")
        amounts.append(amount)
    return amounts


def shake_batch(
    events: list[dict],
    dt: float,
    config: dict,
) -> dict:
    """Process explosion events in order with one accumulated trauma value.

    ``events`` is a sequence of mappings, each with a required finite
    ``"amount"`` in ``[0.0, 1.0]``.  ``dt`` is the non-negative time in
    seconds between consecutive events.  ``config`` supplies:

    * ``"half_life"``: positive decay half-life in seconds;
    * ``"seed"``: integer noise seed;
    * ``"max_offset"``: non-negative displacement amplitude;
    * ``"max_deg"``: non-negative rotational amplitude in degrees;
    * ``"camera"``: the base camera mapping;
    * ``"space"``: ``"local"`` or ``"world"``.

    The first event starts from zero trauma.  For each later event the previous
    trauma is decayed by ``dt`` and then the new amount is added, so earlier
    shakes are never discarded.  At event ``i`` the sample time is
    ``i * dt``; the function calls :func:`offset`, :func:`rotation` and
    :func:`apply` and records the result.

    The return value is::

        {"trauma": [<trauma after each event>],
         "frames": [<camera after each event>]}

    with exactly one frame per event in input order.  If any event or config
    value is invalid the whole call raises :class:`ShakeError` and returns no
    partial result.  Results are compared with a tolerance of ``1e-6``.
    """

    step = _number(dt, "dt")
    if step < 0.0:
        raise ShakeError(f"dt must be non-negative, got {dt!r}")

    settings = _batch_config(config)
    amounts = _batch_events(events)

    trauma = 0.0
    trauma_history: list[float] = []
    frames: list[dict] = []

    for index, amount in enumerate(amounts):
        if index > 0:
            trauma = decay(trauma, step, settings["half_life"])
        trauma = add_trauma(trauma, amount)

        sample_time = index * step
        displacement = offset(trauma, settings["seed"], sample_time, settings["max_offset"])
        angular = rotation(trauma, settings["seed"], sample_time, settings["max_deg"])
        frame = apply(settings["camera"], displacement, angular, settings["space"])

        trauma_history.append(trauma)
        frames.append(frame)

    return {"trauma": trauma_history, "frames": frames}
