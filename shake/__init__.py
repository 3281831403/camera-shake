"""Camera-shake package."""

from .trauma import (
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

__all__ = [
    "ShakeError",
    "add_trauma",
    "apply",
    "decay",
    "limit",
    "noise_1d",
    "offset",
    "rotation",
    "shake_batch",
]
