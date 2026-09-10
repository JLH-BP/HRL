"""分别用 MLP 编码原始 Worker 状态和 context 的 SB3 特征提取器"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .meta_context import META_CONTEXT_SIZE

__all__ = ["ContextConditionedFeaturesExtractor"]


class ContextConditionedFeaturesExtractor(BaseFeaturesExtractor):
    """Encode Worker observations and trailing Meta-RL context with separate MLPs."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        context_size: int = META_CONTEXT_SIZE,
        state_feature_dim: int = 128,
        context_feature_dim: int = 32,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("observation_space must be a one-dimensional gymnasium Box.")
        observation_dim = observation_space.shape[0]
        if context_size < 1 or context_size >= observation_dim:
            raise ValueError("context_size must be positive and smaller than observation dimension.")
        if state_feature_dim < 1 or context_feature_dim < 1:
            raise ValueError("feature dimensions must be positive.")
        super().__init__(observation_space, features_dim=state_feature_dim + context_feature_dim)
        self.context_size = context_size
        self.state_encoder = nn.Sequential(
            nn.Linear(observation_dim - context_size, state_feature_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, context_feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Concatenate learned state and context representations for PPO heads."""
        state = observations[..., :-self.context_size]
        context = observations[..., -self.context_size:]
        return torch.cat((self.state_encoder(state), self.context_encoder(context)), dim=-1)
