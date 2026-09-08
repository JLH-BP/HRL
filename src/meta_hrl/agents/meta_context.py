"""ppo兼容元rl条件反射的确定性上下文特征。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .replay_buffer import ContextTransition

__all__ = ["META_CONTEXT_SIZE", "MetaContextEncoder"]

META_CONTEXT_SIZE: int = 7


class MetaContextEncoder:
    """Summarize recent task transitions without learned parameters or side effects."""

    @property
    def context_size(self) -> int:
        """Return the fixed output feature dimension."""
        return META_CONTEXT_SIZE

    def encode(self, transitions: tuple[ContextTransition, ...]) -> NDArray[np.float32]:
        """Return reward/action/state-change statistics, or zeros for empty context."""
        if not isinstance(transitions, tuple):
            raise TypeError("transitions must be a tuple of ContextTransition instances.")
        if not transitions:
            return np.zeros(META_CONTEXT_SIZE, dtype=np.float32)
        if any(not isinstance(item, ContextTransition) for item in transitions):
            raise TypeError("transitions must contain ContextTransition instances.")
        rewards = np.asarray([item.reward for item in transitions], dtype=np.float64)
        action_norms = np.asarray([np.linalg.norm(item.action) for item in transitions], dtype=np.float64)
        changes = np.asarray(
            [np.linalg.norm(item.next_observation - item.observation) for item in transitions],
            dtype=np.float64,
        )
        terminations = np.asarray([item.terminated for item in transitions], dtype=np.float64)
        features = np.asarray(
            [
                np.mean(rewards), np.std(rewards), np.mean(action_norms), np.std(action_norms),
                np.mean(changes), np.mean(terminations), np.log1p(len(transitions)),
            ],
            dtype=np.float32,
        )
        return features
