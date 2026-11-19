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


def add_trauma(current: float, amount: float) -> float:
    """Add ``amount`` to ``current`` and clamp the result to ``[0.0, 1.0]``.

    Trauma is never allowed to overflow above ``1.0``; a negative ``amount``
    reduces trauma but the result never drops below ``0.0``.  Both arguments
    are finite numbers.  The return value is a float and is compared with a
    tolerance of ``1e-6``.
    """

    raise NotImplementedError


def decay(trauma: float, dt: float, half_life: float) -> float:
    """Return trauma after ``dt`` seconds of exponential half-life decay.

    ``half_life`` is the time in seconds for trauma to halve, so the result is
    ``trauma * 0.5 ** (dt / half_life)``.  ``dt`` may be zero and is never
    negative.  The result never becomes negative; values that decay below
    ``1e-12`` are flushed to ``0.0`` so a fully decayed shake stays at zero.

    ``half_life`` must be positive; ``half_life <= 0`` raises
    :class:`ShakeError`.  Results are compared with a tolerance of ``1e-6``.
    """

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


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

    raise NotImplementedError


def limit(camera: dict, max_pitch: float, max_roll: float) -> dict:
    """Clamp pitch and roll to symmetric limits while preserving yaw.

    Pitch and roll are each clamped to ``[-limit, +limit]``; yaw is copied
    unchanged.  ``max_pitch`` and ``max_roll`` must be non-negative.  The input
    camera is not mutated and the position is preserved in the returned
    camera.  Invalid limits or a malformed camera raise :class:`ShakeError`.
    Results are compared with a tolerance of ``1e-6``.
    """

    raise NotImplementedError


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

    raise NotImplementedError
