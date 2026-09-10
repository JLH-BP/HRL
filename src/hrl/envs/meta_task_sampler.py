"""创建 train/validation/OOD 的慢变化无线任务"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from .task_sampler import ScenarioSamplerConfig

__all__ = ["MetaTaskSampler", "MetaTaskSpec"]


@dataclass(frozen=True, slots=True)
class MetaTaskSpec:
    """One slow-varying task distribution used across multiple fast episodes."""

    task_id: int
    split: str
    sampler_config: ScenarioSamplerConfig
    noise_variance: float


class MetaTaskSampler:
    """Sample distinct reproducible train, validation, and OOD task specifications."""

    def __init__(
        self,
        base_config: ScenarioSamplerConfig = ScenarioSamplerConfig(),
        ood_shift_scale: float = 1.0,
    ) -> None:
        if not isinstance(base_config, ScenarioSamplerConfig):
            raise TypeError("base_config must be a ScenarioSamplerConfig instance.")
        if not np.isfinite(ood_shift_scale) or ood_shift_scale < 0.0:
            raise ValueError("ood_shift_scale must be finite and nonnegative.")
        self.base_config = base_config
        self.ood_shift_scale = float(ood_shift_scale)

    def sample(self, *, split: str, task_id: int, seed: int) -> MetaTaskSpec:
        """Draw one task with ranges disjoint between in-distribution and OOD splits."""
        if split not in {"train", "validation", "ood"}:
            raise ValueError("split must be 'train', 'validation', or 'ood'.")
        if isinstance(task_id, bool) or not isinstance(task_id, Integral):
            raise TypeError("task_id must be an integer.")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer.")
        rng = np.random.default_rng(int(seed) + 1_000_003 * int(task_id))
        users = self.base_config.num_users
        if split == "ood":
            scale = self.ood_shift_scale
            near_bounds = (0, max(1, int(round(users * (1.0 - (2.0 / 3.0) * scale)))))
            k_factor = float(rng.uniform(3.0 + 22.0 * scale, 15.0 + 25.0 * scale))
            path_loss_exponent = float(rng.uniform(1.8 + scale, 2.5 + scale))
            qos = tuple(rng.uniform(0.25 + 0.75 * scale, 0.75 + 0.75 * scale, size=users))
            noise = float(rng.uniform(0.5 + scale, 1.0 + scale))
        else:
            near_bounds = (0, users)
            k_factor = float(rng.uniform(3.0, 15.0))
            path_loss_exponent = float(rng.uniform(1.8, 2.5))
            qos = tuple(rng.uniform(0.25, 0.75, size=users))
            noise = float(rng.uniform(0.5, 1.0))
        config = replace(
            self.base_config,
            near_field_user_count_range=near_bounds,
            k_factor_linear=k_factor,
            path_loss_exponent=path_loss_exponent,
            qos_rate_targets=qos,
        )
        return MetaTaskSpec(int(task_id), split, config, noise)
