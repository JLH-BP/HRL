"""校验 `3K` 维 Worker logits 的形状、数值和边界"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GROUP_WORKER_ACTIONS_PER_USER", "validate_worker_action"]

GROUP_WORKER_ACTIONS_PER_USER: int = 3


def validate_worker_action(action: ArrayLike, *, num_users: int, logit_bound: float) -> NDArray[np.float64]:
    """Validate one bounded fixed-slot Worker action with exactly ``3K`` logits."""
    if isinstance(num_users, bool) or not isinstance(num_users, int) or num_users < 1:
        raise ValueError("num_users must be a positive integer.")
    if not np.isfinite(logit_bound) or logit_bound <= 0.0:
        raise ValueError("logit_bound must be finite and positive.")
    values = np.asarray(action)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("action must contain real numeric values.")
    vector = np.asarray(action, dtype=np.float64)
    if vector.shape != (GROUP_WORKER_ACTIONS_PER_USER * num_users,) or not np.all(np.isfinite(vector)):
        raise ValueError("action must be finite with exactly 3 * num_users entries.")
    if np.any(np.abs(vector) > logit_bound):
        raise ValueError("action must be within the configured logit bounds.")
    return vector
