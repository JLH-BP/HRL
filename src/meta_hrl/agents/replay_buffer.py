"""容量受限且按 task ID 隔离的 transition 缓冲区"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ContextTransition", "TaskContextBuffer"]


@dataclass(frozen=True, slots=True)
class ContextTransition:
    """One immutable Worker transition associated with exactly one meta-task."""

    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    reward: float
    next_observation: NDArray[np.float32]
    terminated: bool
    task_id: int


class TaskContextBuffer:
    """Keep recent immutable context transitions independently for every task ID."""

    def __init__(self, capacity_per_task: int = 64) -> None:
        if isinstance(capacity_per_task, bool) or not isinstance(capacity_per_task, Integral) or capacity_per_task < 1:
            raise ValueError("capacity_per_task must be a positive integer.")
        self.capacity_per_task = int(capacity_per_task)
        self._transitions: dict[int, deque[ContextTransition]] = {}

    def add(
        self, *, task_id: int, observation: ArrayLike, action: ArrayLike, reward: float,
        next_observation: ArrayLike, terminated: bool,
    ) -> None:
        """Copy and append one finite transition without exposing mutable storage."""
        if isinstance(task_id, bool) or not isinstance(task_id, Integral):
            raise TypeError("task_id must be an integer.")
        if isinstance(terminated, (bool, np.bool_)) is False:
            raise TypeError("terminated must be a boolean.")
        current = self._finite_vector(observation, "observation")
        action_vector = self._finite_vector(action, "action")
        following = self._finite_vector(next_observation, "next_observation")
        if current.shape != following.shape:
            raise ValueError("observation and next_observation must have matching shapes.")
        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite.")
        transition = ContextTransition(current, action_vector, reward_value, following, bool(terminated), int(task_id))
        bucket = self._transitions.setdefault(int(task_id), deque(maxlen=self.capacity_per_task))
        bucket.append(transition)

    def recent(self, task_id: int, count: int | None = None) -> tuple[ContextTransition, ...]:
        """Return copied recent transitions for exactly one task, never another task."""
        if isinstance(task_id, bool) or not isinstance(task_id, Integral):
            raise TypeError("task_id must be an integer.")
        if count is not None and (isinstance(count, bool) or not isinstance(count, Integral) or count < 1):
            raise ValueError("count must be a positive integer or None.")
        values = tuple(self._transitions.get(int(task_id), ()))
        if count is not None:
            values = values[-int(count):]
        return tuple(
            ContextTransition(
                item.observation.copy(), item.action.copy(), item.reward,
                item.next_observation.copy(), item.terminated, item.task_id,
            )
            for item in values
        )

    @staticmethod
    def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float32]:
        raw = np.asarray(value)
        if np.iscomplexobj(raw) or raw.dtype.kind in {"b", "O", "U", "S"}:
            raise TypeError(f"{name} must contain real numeric values.")
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite nonempty one-dimensional vector.")
        return vector.copy()
