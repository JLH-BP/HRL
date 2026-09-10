"""Worker-training wrapper for fixed, heuristic, and composite Managers."""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hrl.agents.high_level_policy import (
    ManagerDecision,
    ManagerSelection,
    RandomCompositeActionManager,
    coerce_manager_decision,
)
from hrl.grouping.candidate_groups import UserPartition

from .hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from .task_sampler import RSMAScenario

__all__ = ["PartitionManager", "WorkerTrainingEnv", "partition_membership_features"]


class PartitionManager(Protocol):
    """Select a partition and optional near/far/all service mode per interval."""

    def select(
        self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]
    ) -> ManagerSelection: ...


def partition_membership_features(partition: UserPartition, *, num_users: int) -> NDArray[np.float32]:
    """Return the symmetric user-by-user membership matrix, flattened."""
    features = np.zeros((num_users, num_users), dtype=np.float32)
    for group in partition:
        indices = np.asarray(group, dtype=np.int64)
        features[np.ix_(indices, indices)] = 1.0
    return features.reshape(-1)


class WorkerTrainingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Expose one Worker action per physical time slot.

    The underlying environment appends the current ``service_mask`` to its
    base observation. This wrapper appends membership features for the active
    (service-masked) partition, so the PPO Worker sees both the eligible users
    and the group structure for every resource-allocation decision.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
        manager: PartitionManager | None = None,
    ) -> None:
        self.hierarchical_env = HierarchicalRSMAEnv(config)
        # Stage-one Worker training must expose every legal service mask rather
        # than train only on legacy singleton/all-user decisions.
        self.manager: PartitionManager = RandomCompositeActionManager() if manager is None else manager
        if not (hasattr(self.manager, "select") or hasattr(self.manager, "select_action")):
            raise TypeError("manager must provide select(scenario, candidates) or select_action(...).")
        self.action_space = self.hierarchical_env.worker_action_space
        base_size = self.hierarchical_env.observation_space.shape[0]
        users = config.sampler.num_users
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_size + users * users,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        observation, info = self.hierarchical_env.reset(seed=seed, options=options)
        self._reset_manager(seed)
        observation, manager_info = self._set_next_manager_decision()
        info.update(manager_info)
        return self._augment(observation), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.hierarchical_env.step(action)
        if info["manager_action_required"]:
            observation, manager_info = self._set_next_manager_decision()
            info.update(manager_info)
        return self._augment(observation), reward, terminated, truncated, info

    def _reset_manager(self, seed: int | None) -> None:
        """Reset stateful schedule policies at an episode boundary when present."""
        reset_manager = getattr(self.manager, "reset", None)
        if not callable(reset_manager):
            return
        if seed is None:
            reset_manager()
            return
        try:
            reset_manager(seed=seed)
        except TypeError:
            # Third-party legacy Managers commonly expose a no-argument reset.
            reset_manager()

    def _select_manager_decision(self) -> ManagerDecision:
        selector = getattr(self.manager, "select_action", None)
        if not callable(selector):
            selector = self.manager.select
        selection = selector(self.hierarchical_env.scenario, self.hierarchical_env.candidates)
        return coerce_manager_decision(
            selection, self.hierarchical_env.scenario, self.hierarchical_env.candidates
        )

    def _set_next_manager_decision(self) -> tuple[NDArray[np.float32], dict[str, Any]]:
        decision = self._select_manager_decision()
        # A tuple is intentionally used here. Plain integers are the new
        # composite codec and can otherwise be confused with a legacy index.
        observation, manager_info = self.hierarchical_env.set_manager_action(
            (decision.candidate_index, decision.service_mode)
        )
        manager_info.update(
            {
                "manager_candidate_index": decision.candidate_index,
                "manager_service_mode": decision.service_mode,
                "manager_decision": decision,
                "manager_action": self._encode_manager_action(decision),
            }
        )
        return observation, manager_info

    def _encode_manager_action(self, decision: ManagerDecision) -> int:
        encoder = getattr(self.hierarchical_env, "encode_manager_action", None)
        if callable(encoder):
            return int(encoder(decision.candidate_index, decision.service_mode))
        return decision.candidate_index * 3 + ("all", "near", "far").index(decision.service_mode)

    def _active_partition(self) -> UserPartition:
        # The physical layer can turn unscheduled users into singleton groups.
        # Reflect that effective partition in the Worker feature vector.
        effective = getattr(self.hierarchical_env, "effective_partition", None)
        return self.hierarchical_env.partition if effective is None else effective

    def _augment(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.concatenate(
            (
                observation,
                partition_membership_features(
                    self._active_partition(),
                    num_users=self.hierarchical_env.config.sampler.num_users,
                ),
            )
        ).astype(np.float32, copy=False)
