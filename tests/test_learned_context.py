# Tests the learned state/context PPO feature extractor.
import gymnasium as gym
import pytest


def test_context_feature_extractor_separates_and_concatenates_branches() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from hrl.agents.learned_context import ContextConditionedFeaturesExtractor

    extractor = ContextConditionedFeaturesExtractor(
        gym.spaces.Box(-1.0, 1.0, shape=(10,), dtype=float),
        context_size=2,
        state_feature_dim=5,
        context_feature_dim=3,
    )
    features = extractor(torch.zeros((4, 10), dtype=torch.float32))

    assert extractor.features_dim == 8
    assert features.shape == (4, 8)


def test_context_feature_extractor_validates_dimensions() -> None:
    pytest.importorskip("stable_baselines3")
    from hrl.agents.learned_context import ContextConditionedFeaturesExtractor

    space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=float)
    with pytest.raises(ValueError, match="smaller"):
        ContextConditionedFeaturesExtractor(space, context_size=7)
    with pytest.raises(ValueError, match="positive"):
        ContextConditionedFeaturesExtractor(space, context_size=2, context_feature_dim=0)
