"""One-command, paired main-experiment workflow for mixed-field group RSMA.

The five paper baselines deliberately share one :class:`MainExperimentConfig`.
This module owns experiment orchestration and reporting only: physical-layer and
environment semantics remain in the environment/RSMA modules.  Keeping the
comparison harness here prevents configuration drift between baselines.
"""

from __future__ import annotations

import csv
import json
import platform
import random
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import combinations, product
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from hrl.channel.geometry import ULAConfig
from hrl.envs import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from hrl.envs.rsma_env import jain_fairness
from hrl.envs.task_sampler import ScenarioSamplerConfig, TaskSampler
from hrl.envs.worker_training_env import partition_membership_features
from hrl.grouping.candidate_groups import UserPartition, canonicalize_partition, enumerate_candidate_partitions
from hrl.rsma.baselines import fixed_common_rsma

__all__ = [
    "MAIN_EXPERIMENT_METHODS",
    "MAIN_METRICS",
    "MainEvaluationMetrics",
    "MainExperimentConfig",
    "MainExperimentResult",
    "MainMethodStrategy",
    "build_main_method_strategy",
    "equal_power_rzf_rsma_strategy",
    "evaluate_equal_power_rzf_rsma",
    "evaluate_main_policy",
    "fixed_group_ppo_strategy",
    "heuristic_group_ppo_strategy",
    "load_main_experiment_config",
    "near_first_ppo_strategy",
    "run_main_experiments",
    "hrl_joint_scheduler_strategy",
]


MAIN_EXPERIMENT_METHODS: tuple[str, ...] = (
    "equal_power_rzf_rsma",
    "fixed_group_ppo",
    "near_first_ppo",
    "heuristic_group_ppo",
    "hrl_joint_scheduler",
)

# These scalar metrics are recorded per training seed, receive bootstrap CIs,
# and are used by the paired sign-flip tests.  Per-user service fractions are
# appended dynamically once the user count is known.
MAIN_METRICS: tuple[str, ...] = (
    "mean_reward",
    "mean_sum_rate",
    "mean_min_user_rate",
    "mean_jain_fairness",
    "mean_total_qos_gap",
    "qos_satisfaction_rate",
    "mean_near_user_rate",
    "mean_far_user_rate",
    "mean_near_far_rate_difference",
    "mean_partition_switches",
    "mean_manager_decisions",
    "mean_manager_decision_frequency",
)

_SERVICE_MODES = ("all", "near", "far")


def _default_main_environment() -> HierarchicalRSMAEnvConfig:
    """Return the paper contract's genuinely mixed near/far task distribution."""
    base = HierarchicalRSMAEnvConfig()
    users = base.sampler.num_users
    return replace(
        base,
        sampler=replace(base.sampler, near_field_user_count_range=(1, users - 1)),
    )


class PredictivePolicy(Protocol):
    """Minimal SB3-compatible inference protocol."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[Any, Any]: ...


@dataclass(frozen=True, slots=True)
class MainExperimentConfig:
    """Single experiment contract shared by every main-experiment method.

    ``training_seeds`` index independent training runs.  ``test_seeds`` index
    held-out channel scenarios and are deliberately independent of the former;
    every trained model is evaluated over exactly this same ordered set.
    """

    experiment_name: str = "main_mixed_near_far"
    methods: tuple[str, ...] = MAIN_EXPERIMENT_METHODS
    training_seeds: tuple[int, ...] = (11, 29, 47, 61, 73, 89, 101, 127)
    test_seeds: tuple[int, ...] = (
        10_011,
        10_029,
        10_047,
        10_061,
        10_073,
        10_089,
        10_101,
        10_127,
        10_149,
        10_163,
        10_181,
        10_199,
        10_211,
        10_229,
        10_241,
        10_263,
    )
    environment: HierarchicalRSMAEnvConfig = field(default_factory=_default_main_environment)
    fixed_partition: UserPartition = ((0, 1), (2, 3), (4, 5))
    common_power_fraction: float = 0.2
    worker_total_timesteps: int = 100_000
    manager_total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    policy_net_arch: tuple[int, ...] = (256, 256)
    confidence_level: float = 0.95
    bootstrap_samples: int = 10_000
    monte_carlo_permutations: int = 100_000
    analysis_seed: int = 2026
    output_directory: Path = Path("outputs/main_experiment")
    source_config: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a nonempty string.")
        if not isinstance(self.environment, HierarchicalRSMAEnvConfig):
            raise TypeError("environment must be a HierarchicalRSMAEnvConfig instance.")
        methods = tuple(str(method) for method in self.methods)
        if not methods:
            raise ValueError("methods must contain at least one main-experiment method.")
        unknown = set(methods).difference(MAIN_EXPERIMENT_METHODS)
        if unknown:
            raise ValueError(f"Unknown main-experiment method(s): {sorted(unknown)}")
        if len(set(methods)) != len(methods):
            raise ValueError("methods must not contain duplicates.")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "training_seeds", _validated_seeds(self.training_seeds, "training_seeds"))
        object.__setattr__(self, "test_seeds", _validated_seeds(self.test_seeds, "test_seeds"))
        if not np.isfinite(self.common_power_fraction) or not 0.0 <= float(self.common_power_fraction) <= 1.0:
            raise ValueError("common_power_fraction must be within [0, 1].")
        users = self.environment.sampler.num_users
        near_min, near_max = self.environment.sampler.near_field_user_count_range
        if users < 2 or near_min < 1 or near_max >= users:
            raise ValueError(
                "MainExperimentConfig requires a mixed near/far task distribution: "
                "near_field_user_count_range must be within [1, num_users - 1]."
            )
        partition = canonicalize_partition(self.fixed_partition, num_users=users)
        candidates = enumerate_candidate_partitions(num_users=users)
        if partition not in candidates:
            raise ValueError("fixed_partition is not available in the candidate partition set.")
        object.__setattr__(self, "fixed_partition", partition)
        for name in ("worker_total_timesteps", "manager_total_timesteps", "n_steps", "batch_size", "n_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        if not isinstance(self.learning_rate, Real) or not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        architecture = tuple(int(width) for width in self.policy_net_arch)
        if not architecture or any(width < 1 for width in architecture):
            raise ValueError("policy_net_arch must contain positive layer widths.")
        object.__setattr__(self, "policy_net_arch", architecture)
        if not isinstance(self.confidence_level, Real) or not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be within (0, 1).")
        for name in ("bootstrap_samples", "monte_carlo_permutations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(self.analysis_seed, bool) or not isinstance(self.analysis_seed, Integral):
            raise TypeError("analysis_seed must be an integer.")
        object.__setattr__(self, "output_directory", Path(self.output_directory))
        if self.source_config is not None:
            object.__setattr__(self, "source_config", Path(self.source_config))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MainExperimentConfig":
        """Load one authoritative YAML configuration file."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Main experiment YAML was not found: {source}")
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - package dependency protects normal installs.
            raise ImportError("PyYAML is required to load main experiment configurations.") from error
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        except Exception as error:  # yaml's parser errors have varying public types.
            raise ValueError(f"Could not parse main experiment YAML: {source}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("Main experiment YAML must contain a mapping at its root.")
        return cls.from_mapping(payload, source_config=source)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, source_config: str | Path | None = None
    ) -> "MainExperimentConfig":
        """Build the contract from a YAML-compatible mapping.

        The parser accepts compact top-level ``training_seeds``/``test_seeds``
        fields and a ``seeds: {training, test}`` form so CLI users do not need
        Python dataclass syntax in YAML.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        seeds = _mapping(payload.get("seeds", {}), "seeds")
        environment = _environment_from_mapping(_mapping(payload.get("environment", {}), "environment"))
        training = _mapping(payload.get("training", {}), "training")
        worker = _mapping(training.get("worker", training), "training.worker")
        manager = _mapping(training.get("manager", {}), "training.manager")
        analysis = _mapping(payload.get("analysis", {}), "analysis")
        baseline = _mapping(payload.get("baseline", {}), "baseline")
        fixed_group = _mapping(payload.get("fixed_group", {}), "fixed_group")
        methods_value = payload.get("methods", MAIN_EXPERIMENT_METHODS)
        if isinstance(methods_value, str):
            methods = tuple(item.strip() for item in methods_value.split(",") if item.strip())
        else:
            methods = tuple(methods_value)
        fixed_partition = fixed_group.get("partition", payload.get("fixed_partition", ((0, 1), (2, 3), (4, 5))))
        defaults = cls()
        return cls(
            experiment_name=str(payload.get("experiment_name", "main_mixed_near_far")),
            methods=methods,
            training_seeds=tuple(payload.get("training_seeds", seeds.get("training", defaults.training_seeds))),
            test_seeds=tuple(payload.get("test_seeds", seeds.get("test", defaults.test_seeds))),
            environment=environment,
            fixed_partition=tuple(tuple(group) for group in fixed_partition),
            common_power_fraction=float(baseline.get("common_power_fraction", payload.get("common_power_fraction", 0.2))),
            worker_total_timesteps=int(worker.get("total_timesteps", payload.get("worker_total_timesteps", 100_000))),
            manager_total_timesteps=int(manager.get("total_timesteps", payload.get("manager_total_timesteps", worker.get("total_timesteps", 100_000)))),
            learning_rate=float(worker.get("learning_rate", training.get("learning_rate", 3e-4))),
            n_steps=int(worker.get("n_steps", training.get("n_steps", 256))),
            batch_size=int(worker.get("batch_size", training.get("batch_size", 64))),
            n_epochs=int(worker.get("n_epochs", training.get("n_epochs", 10))),
            policy_net_arch=tuple(worker.get("policy_net_arch", training.get("policy_net_arch", (256, 256)))),
            confidence_level=float(analysis.get("confidence_level", 0.95)),
            bootstrap_samples=int(analysis.get("bootstrap_samples", 10_000)),
            monte_carlo_permutations=int(analysis.get("monte_carlo_permutations", 100_000)),
            analysis_seed=int(analysis.get("seed", 2026)),
            output_directory=Path(payload.get("output_directory", "outputs/main_experiment")),
            source_config=None if source_config is None else Path(source_config),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation stored in each manifest."""
        sampler = self.environment.sampler
        ula = sampler.ula_config
        return {
            "experiment_name": self.experiment_name,
            "output_directory": str(self.output_directory),
            "methods": list(self.methods),
            "training_seeds": list(self.training_seeds),
            "test_seeds": list(self.test_seeds),
            "environment": {
                "num_users": sampler.num_users,
                "num_antennas": ula.num_antennas,
                "carrier_frequency_hz": ula.carrier_frequency_hz,
                "element_spacing_m": ula.element_spacing_m,
                "angle_range_rad": list(sampler.angle_range_rad),
                "near_range_m": list(sampler.near_range_m),
                "far_range_m": list(sampler.far_range_m),
                "near_field_user_count_range": list(sampler.near_field_user_count_range),
                "k_factor_linear": sampler.k_factor_linear,
                "path_loss_exponent": sampler.path_loss_exponent,
                "path_loss_reference_distance_m": sampler.path_loss_reference_distance_m,
                "qos_rate_targets": list(sampler.qos_rate_targets),
                "total_power": self.environment.total_power,
                "noise_variance": self.environment.noise_variance,
                "private_scheme": self.environment.private_scheme,
                "high_level_interval": self.environment.high_level_interval,
                "episode_length": self.environment.episode_length,
                "partition_switch_penalty": self.environment.partition_switch_penalty,
                "sum_rate_weight": self.environment.sum_rate_weight,
                "fairness_weight": self.environment.fairness_weight,
                "qos_gap_weight": self.environment.qos_gap_weight,
                "action_logit_bound": self.environment.action_logit_bound,
            },
            "fixed_group": {"partition": [list(group) for group in self.fixed_partition]},
            "baseline": {"common_power_fraction": self.common_power_fraction},
            "training": {
                "worker": {
                    "total_timesteps": self.worker_total_timesteps,
                    "learning_rate": self.learning_rate,
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "policy_net_arch": list(self.policy_net_arch),
                },
                "manager": {"total_timesteps": self.manager_total_timesteps},
            },
            "analysis": {
                "confidence_level": self.confidence_level,
                "bootstrap_samples": self.bootstrap_samples,
                "monte_carlo_permutations": self.monte_carlo_permutations,
                "seed": self.analysis_seed,
            },
        }


@dataclass(frozen=True, slots=True)
class MainMethodStrategy:
    """Description of one method's high-level policy replacement."""

    name: str
    worker_manager: Any | None
    trains_worker: bool
    trains_manager: bool
    description: str


@dataclass(frozen=True, slots=True)
class MainEvaluationMetrics:
    """Paired held-out metrics produced from complete episodes."""

    mean_reward: float
    mean_sum_rate: float
    mean_min_user_rate: float
    mean_jain_fairness: float
    mean_total_qos_gap: float
    qos_satisfaction_rate: float
    mean_near_user_rate: float
    mean_far_user_rate: float
    mean_near_far_rate_difference: float
    mean_user_service_fractions: tuple[float, ...]
    mean_partition_switches: float
    mean_manager_decisions: float
    mean_manager_decision_frequency: float
    episodes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_reward": self.mean_reward,
            "mean_sum_rate": self.mean_sum_rate,
            "mean_min_user_rate": self.mean_min_user_rate,
            "mean_jain_fairness": self.mean_jain_fairness,
            "mean_total_qos_gap": self.mean_total_qos_gap,
            "qos_satisfaction_rate": self.qos_satisfaction_rate,
            "mean_near_user_rate": self.mean_near_user_rate,
            "mean_far_user_rate": self.mean_far_user_rate,
            "mean_near_far_rate_difference": self.mean_near_far_rate_difference,
            "mean_user_service_fractions": list(self.mean_user_service_fractions),
            "mean_partition_switches": self.mean_partition_switches,
            "mean_manager_decisions": self.mean_manager_decisions,
            "mean_manager_decision_frequency": self.mean_manager_decision_frequency,
            "episodes": self.episodes,
        }

    def flat_dict(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            key: value for key, value in self.to_dict().items() if key != "mean_user_service_fractions"
        }
        for user, fraction in enumerate(self.mean_user_service_fractions):
            values[f"mean_user_service_fraction_{user}"] = fraction
        return values


@dataclass(frozen=True, slots=True)
class MainExperimentResult:
    """Locations of the reproducible main-experiment artifact set."""

    output_directory: Path
    manifest_path: Path
    results_json_path: Path
    results_csv_path: Path
    significance_path: Path
    plot_paths: tuple[Path, ...]


def _validated_seeds(values: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        seeds = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of integers.") from error
    if not seeds:
        raise ValueError(f"{name} must not be empty.")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise TypeError(f"{name} must contain integers.")
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _environment_from_mapping(payload: Mapping[str, Any]) -> HierarchicalRSMAEnvConfig:
    """Convert concise YAML environment settings into validated dataclasses."""
    default = _default_main_environment()
    sampler_data = dict(_mapping(payload.get("sampler", {}), "environment.sampler"))
    # Flat YAML keys are intentionally accepted for a readable experiment file.
    for key in (
        "num_users",
        "near_range_m",
        "far_range_m",
        "near_field_user_count_range",
        "k_factor_linear",
        "path_loss_exponent",
        "path_loss_reference_distance_m",
        "qos_rate_targets",
        "angle_range_rad",
    ):
        if key in payload:
            sampler_data[key] = payload[key]
    if "angle_range_degrees" in payload:
        sampler_data["angle_range_rad"] = tuple(np.deg2rad(np.asarray(payload["angle_range_degrees"], dtype=np.float64)))
    ula_data = dict(_mapping(sampler_data.pop("ula", {}), "environment.sampler.ula"))
    for key in ("num_antennas", "carrier_frequency_hz", "element_spacing_m"):
        if key in payload:
            ula_data[key] = payload[key]
    ula = ULAConfig(
        num_antennas=int(ula_data.get("num_antennas", default.sampler.ula_config.num_antennas)),
        carrier_frequency_hz=float(ula_data.get("carrier_frequency_hz", default.sampler.ula_config.carrier_frequency_hz)),
        element_spacing_m=ula_data.get("element_spacing_m", default.sampler.ula_config.element_spacing_m),
    )
    sampler = ScenarioSamplerConfig(
        num_users=int(sampler_data.get("num_users", default.sampler.num_users)),
        ula_config=ula,
        angle_range_rad=tuple(sampler_data.get("angle_range_rad", default.sampler.angle_range_rad)),
        near_range_m=tuple(sampler_data.get("near_range_m", default.sampler.near_range_m)),
        far_range_m=tuple(sampler_data.get("far_range_m", default.sampler.far_range_m)),
        near_field_user_count_range=tuple(sampler_data.get("near_field_user_count_range", default.sampler.near_field_user_count_range)),
        k_factor_linear=float(sampler_data.get("k_factor_linear", default.sampler.k_factor_linear)),
        path_loss_exponent=float(sampler_data.get("path_loss_exponent", default.sampler.path_loss_exponent)),
        path_loss_reference_distance_m=float(
            sampler_data.get("path_loss_reference_distance_m", default.sampler.path_loss_reference_distance_m)
        ),
        qos_rate_targets=tuple(sampler_data.get("qos_rate_targets", default.sampler.qos_rate_targets)),
    )
    env_values: dict[str, Any] = {"sampler": sampler}
    for name in (
        "total_power",
        "noise_variance",
        "private_scheme",
        "high_level_interval",
        "episode_length",
        "partition_switch_penalty",
        "sum_rate_weight",
        "fairness_weight",
        "qos_gap_weight",
        "action_logit_bound",
    ):
        env_values[name] = payload.get(name, getattr(default, name))
    return HierarchicalRSMAEnvConfig(**env_values)


def load_main_experiment_config(path: str | Path) -> MainExperimentConfig:
    """Convenience loader used by the CLI and external scripts."""
    return MainExperimentConfig.from_yaml(path)


def _candidate_index_for_partition(config: MainExperimentConfig) -> int:
    candidates = enumerate_candidate_partitions(num_users=config.environment.sampler.num_users)
    return candidates.index(config.fixed_partition)


def _fixed_manager(config: MainExperimentConfig) -> Any:
    """Construct an explicit multi-user fixed-group policy, never index zero."""
    from hrl.agents.high_level_policy import FixedPartitionManager

    try:
        from hrl.agents.high_level_policy import ExplicitFixedPartitionManager
    except ImportError:
        ExplicitFixedPartitionManager = None  # type: ignore[assignment,misc]
    if ExplicitFixedPartitionManager is not None:
        return ExplicitFixedPartitionManager(partition=config.fixed_partition, service_mode="all")
    # Compatibility for historical checkpoints.  The index is derived from the
    # requested non-singleton partition and is never hard-coded to zero.
    return FixedPartitionManager(candidate_index=_candidate_index_for_partition(config))


def equal_power_rzf_rsma_strategy(config: MainExperimentConfig, *, seed: int) -> MainMethodStrategy:
    del config, seed
    return MainMethodStrategy(
        name="equal_power_rzf_rsma",
        worker_manager=None,
        trains_worker=False,
        trains_manager=False,
        description="All users; fixed common-power fraction and equal-rate RZF-RSMA.",
    )


def fixed_group_ppo_strategy(config: MainExperimentConfig, *, seed: int) -> MainMethodStrategy:
    del seed
    return MainMethodStrategy(
        name="fixed_group_ppo",
        worker_manager=_fixed_manager(config),
        trains_worker=True,
        trains_manager=False,
        description="Explicit fixed multi-user partition with all-user service.",
    )


def near_first_ppo_strategy(config: MainExperimentConfig, *, seed: int) -> MainMethodStrategy:
    from hrl.agents.high_level_policy import NearFieldFirstSequentialManager

    try:
        manager = NearFieldFirstSequentialManager(seed=seed)
    except TypeError:  # legacy implementation has no constructor arguments.
        manager = NearFieldFirstSequentialManager()
    return MainMethodStrategy(
        name="near_first_ppo",
        worker_manager=manager,
        trains_worker=True,
        trains_manager=False,
        description="Fixed near -> far -> near service order with a PPO Worker.",
    )


def heuristic_group_ppo_strategy(config: MainExperimentConfig, *, seed: int) -> MainMethodStrategy:
    del config, seed
    from hrl.agents.high_level_policy import HeuristicPartitionManager

    return MainMethodStrategy(
        name="heuristic_group_ppo",
        worker_manager=HeuristicPartitionManager(),
        trains_worker=True,
        trains_manager=False,
        description="Traditional CSI/geometry grouping with all-user service.",
    )


def hrl_joint_scheduler_strategy(config: MainExperimentConfig, *, seed: int) -> MainMethodStrategy:
    del config
    try:
        from hrl.agents.high_level_policy import RandomCompositeActionManager
    except ImportError as error:
        raise ImportError(
            "hrl_joint_scheduler requires RandomCompositeActionManager for staged Worker pretraining."
        ) from error
    try:
        worker_manager = RandomCompositeActionManager(seed=seed)
    except TypeError:
        worker_manager = RandomCompositeActionManager()
    return MainMethodStrategy(
        name="hrl_joint_scheduler",
        worker_manager=worker_manager,
        trains_worker=True,
        trains_manager=True,
        description="Staged PPO: random legal composite actions for Worker pretraining, then PPO Manager.",
    )


def build_main_method_strategy(
    method: str, config: MainExperimentConfig, *, seed: int
) -> MainMethodStrategy:
    """Return one of the five explicit strategy factories."""
    factories = {
        "equal_power_rzf_rsma": equal_power_rzf_rsma_strategy,
        "fixed_group_ppo": fixed_group_ppo_strategy,
        "near_first_ppo": near_first_ppo_strategy,
        "heuristic_group_ppo": heuristic_group_ppo_strategy,
        "hrl_joint_scheduler": hrl_joint_scheduler_strategy,
    }
    try:
        return factories[method](config, seed=seed)
    except KeyError as error:
        raise ValueError(f"Unknown main-experiment method: {method}") from error


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch before every independent PPO run."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _as_manager_tuple(selection: Any, environment: HierarchicalRSMAEnv) -> tuple[int, str]:
    """Normalize legacy ints, tuples and ManagerDecision objects to (index, mode)."""
    if hasattr(selection, "candidate_index"):
        index = int(selection.candidate_index)
        mode = str(getattr(selection, "service_mode", "all"))
    elif isinstance(selection, tuple) and len(selection) == 2:
        index, mode = int(selection[0]), str(selection[1])
    elif isinstance(selection, (Integral, np.integer)) and not isinstance(selection, (bool, np.bool_)):
        # A legacy fixed manager returns a partition index.  Modern static
        # managers return ManagerDecision/tuple, so this fallback is unambiguous.
        index, mode = int(selection), "all"
    else:
        raise TypeError("A static Manager must return an int, (partition_index, service_mode), or ManagerDecision.")
    if mode not in _SERVICE_MODES:
        raise ValueError(f"Unknown service mode: {mode}")
    if not 0 <= index < len(environment.candidates):
        raise ValueError("Manager selected an invalid candidate partition index.")
    return index, mode


def _set_manager_action(
    environment: HierarchicalRSMAEnv, index: int, mode: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the composite action API while retaining a narrow legacy fallback."""
    try:
        return environment.set_manager_action((index, mode))
    except (TypeError, ValueError):
        if mode != "all":
            raise
        return environment.set_manager_action(index)


def _decode_learned_manager_action(environment: HierarchicalRSMAEnv, action: Any) -> tuple[int, str]:
    value = np.asarray(action)
    if value.size != 1:
        raise ValueError("PPO Manager must predict one discrete action.")
    encoded = int(value.reshape(-1)[0])
    decoder = getattr(environment, "decode_manager_action", None)
    if callable(decoder):
        partition_index, mode = decoder(encoded)
        return int(partition_index), str(mode)
    if not 0 <= encoded < len(environment.candidates):
        raise ValueError("PPO Manager predicted an invalid candidate partition index.")
    return encoded, "all"


def _active_partition(environment: HierarchicalRSMAEnv) -> UserPartition:
    partition = getattr(environment, "effective_partition", None)
    if partition is None:
        partition = environment.partition
    return partition


def _worker_observation(environment: HierarchicalRSMAEnv, observation: np.ndarray) -> np.ndarray:
    """Mirror WorkerTrainingEnv's non-learning observation augmentation."""
    users = environment.config.sampler.num_users
    membership = partition_membership_features(_active_partition(environment), num_users=users)
    return np.concatenate((observation, membership)).astype(np.float32, copy=False)


def _manager_observation(
    environment: HierarchicalRSMAEnv, observation: np.ndarray, previous_partition_index: int | None
) -> np.ndarray:
    """Mirror ManagerTrainingEnv's state augmentation without a second environment."""
    users = environment.config.sampler.num_users
    if previous_partition_index is None:
        membership = np.zeros(users * users, dtype=np.float32)
    else:
        # ManagerTrainingEnv uses the effective (service-masked) partition.
        # Reusing it here keeps frozen-Worker/Manager inference observationally
        # identical to the PPO Manager's training rollouts.
        membership = partition_membership_features(
            _active_partition(environment), num_users=users
        )
    return np.concatenate((observation, membership)).astype(np.float32, copy=False)


def _episode_record(
    *,
    seed: int,
    scenario: Any,
    cumulative_rates: np.ndarray,
    cumulative_qos_gaps: np.ndarray,
    service_fractions: np.ndarray,
    reward: float,
    steps: int,
    partition_switches: int,
    manager_decisions: int,
) -> dict[str, Any]:
    average_rates = cumulative_rates / max(steps, 1)
    near_mask = np.asarray(scenario.near_field_mask, dtype=bool)
    near_rate = float(np.mean(average_rates[near_mask])) if np.any(near_mask) else 0.0
    far_rate = float(np.mean(average_rates[~near_mask])) if np.any(~near_mask) else 0.0
    return {
        "scenario_seed": int(seed),
        "episode_reward": float(reward),
        "sum_rate": float(np.sum(average_rates)),
        "min_user_rate": float(np.min(average_rates)),
        "jain_fairness": float(jain_fairness(cumulative_rates)),
        "total_qos_gap": float(np.sum(cumulative_qos_gaps)),
        # This is a user-level fraction, matching HierarchicalRSMAEnv's
        # `qos_satisfaction_rate`; retaining the stricter all-user indicator
        # makes complete-episode success auditable without overloading it.
        "qos_satisfaction_rate": float(np.mean(cumulative_qos_gaps <= 1e-10)),
        "all_users_qos_satisfied": bool(np.all(cumulative_qos_gaps <= 1e-10)),
        "near_user_rate": near_rate,
        "far_user_rate": far_rate,
        "near_far_rate_difference": near_rate - far_rate,
        "user_service_fractions": [float(value) for value in service_fractions],
        "partition_switches": int(partition_switches),
        "manager_decisions": int(manager_decisions),
        "manager_decision_frequency": float(manager_decisions / max(steps, 1)),
        "near_field_mask": [bool(value) for value in near_mask],
    }


def _metrics_from_episode_records(records: Sequence[Mapping[str, Any]]) -> MainEvaluationMetrics:
    if not records:
        raise ValueError("At least one held-out episode is required.")
    user_count = len(records[0]["user_service_fractions"])
    return MainEvaluationMetrics(
        mean_reward=_mean(records, "episode_reward"),
        mean_sum_rate=_mean(records, "sum_rate"),
        mean_min_user_rate=_mean(records, "min_user_rate"),
        mean_jain_fairness=_mean(records, "jain_fairness"),
        mean_total_qos_gap=_mean(records, "total_qos_gap"),
        qos_satisfaction_rate=_mean(records, "qos_satisfaction_rate"),
        mean_near_user_rate=_mean(records, "near_user_rate"),
        mean_far_user_rate=_mean(records, "far_user_rate"),
        mean_near_far_rate_difference=_mean(records, "near_far_rate_difference"),
        mean_user_service_fractions=tuple(
            float(np.mean([float(record["user_service_fractions"][user]) for record in records]))
            for user in range(user_count)
        ),
        mean_partition_switches=_mean(records, "partition_switches"),
        mean_manager_decisions=_mean(records, "manager_decisions"),
        mean_manager_decision_frequency=_mean(records, "manager_decision_frequency"),
        episodes=len(records),
    )


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records]))


def evaluate_main_policy(
    worker: PredictivePolicy,
    *,
    manager: Any,
    environment: HierarchicalRSMAEnvConfig,
    test_seeds: Sequence[int],
    learned_manager: bool = False,
) -> tuple[MainEvaluationMetrics, list[dict[str, Any]]]:
    """Evaluate a fixed/heuristic or PPO Manager plus PPO Worker on shared seeds.

    The rollout is intentionally implemented against ``HierarchicalRSMAEnv``
    rather than a training wrapper.  It therefore records every time slot's
    service mask, accumulated user rate and Manager decision, including the
    data needed to check near-first and joint scheduling claims.
    """
    if not hasattr(worker, "predict"):
        raise TypeError("worker must provide predict(observation, deterministic=True).")
    if not hasattr(manager, "predict" if learned_manager else "select"):
        expected = "predict" if learned_manager else "select"
        raise TypeError(f"manager must provide {expected}(...).")
    seeds = _validated_seeds(test_seeds, "test_seeds")
    env = HierarchicalRSMAEnv(environment)
    records: list[dict[str, Any]] = []
    try:
        for scenario_seed in seeds:
            observation, _ = env.reset(seed=scenario_seed)
            scenario = env.scenario
            if not learned_manager:
                reset_manager = getattr(manager, "reset", None)
                if callable(reset_manager):
                    try:
                        reset_manager(seed=scenario_seed)
                    except TypeError:
                        reset_manager()
            users = environment.sampler.num_users
            cumulative_rates = np.zeros(users, dtype=np.float64)
            service_counts = np.zeros(users, dtype=np.float64)
            partition_switches = 0
            manager_decisions = 0
            episode_reward = 0.0
            steps = 0
            terminated = False
            truncated = False
            manager_required = True
            previous_partition_index: int | None = None
            last_info: dict[str, Any] = {}
            while not (terminated or truncated):
                if manager_required:
                    if learned_manager:
                        manager_observation = _manager_observation(env, observation, previous_partition_index)
                        raw_action, _ = manager.predict(manager_observation, deterministic=True)
                        index, mode = _decode_learned_manager_action(env, raw_action)
                    else:
                        index, mode = _as_manager_tuple(manager.select(scenario, env.candidates), env)
                    observation, _ = _set_manager_action(env, index, mode)
                    previous_partition_index = index
                    manager_decisions += 1
                worker_action, _ = worker.predict(_worker_observation(env, observation), deterministic=True)
                observation, reward, terminated, truncated, info = env.step(worker_action)
                last_info = info
                user_rates = np.asarray(info["user_rates"], dtype=np.float64)
                if user_rates.shape != (users,):
                    raise RuntimeError("Hierarchical environment info must provide one user rate per user.")
                cumulative_rates += user_rates
                mask = np.asarray(info.get("service_mask", getattr(env, "service_mask", np.ones(users))), dtype=np.float64)
                if mask.shape != (users,):
                    raise RuntimeError("Hierarchical environment must expose a user-length service_mask.")
                service_counts += mask
                partition_switches += int(bool(info.get("partition_switched", False)))
                episode_reward += float(reward)
                steps += 1
                manager_required = bool(info.get("manager_action_required", False)) and not (terminated or truncated)
            # The environment's terminal fields are authoritative when present;
            # fallbacks keep the evaluator useful for legacy checkpoints.
            reported_rates = np.asarray(last_info.get("cumulative_user_rates", cumulative_rates), dtype=np.float64)
            if reported_rates.shape != (users,):
                reported_rates = cumulative_rates
            reported_gaps = np.asarray(
                last_info.get(
                    "cumulative_qos_gaps",
                    np.maximum(scenario.qos_rate_targets * steps - reported_rates, 0.0),
                ),
                dtype=np.float64,
            )
            if reported_gaps.shape != (users,):
                reported_gaps = np.maximum(scenario.qos_rate_targets * steps - reported_rates, 0.0)
            service_fractions = np.asarray(last_info.get("service_fractions", service_counts / max(steps, 1)), dtype=np.float64)
            if service_fractions.shape != (users,):
                service_fractions = service_counts / max(steps, 1)
            records.append(
                _episode_record(
                    seed=scenario_seed,
                    scenario=scenario,
                    cumulative_rates=reported_rates,
                    cumulative_qos_gaps=reported_gaps,
                    service_fractions=service_fractions,
                    reward=float(last_info.get("episode_reward", episode_reward)),
                    steps=steps,
                    partition_switches=partition_switches,
                    manager_decisions=manager_decisions,
                )
            )
    finally:
        env.close()
    return _metrics_from_episode_records(records), records


def evaluate_equal_power_rzf_rsma(
    config: MainExperimentConfig,
) -> tuple[MainEvaluationMetrics, list[dict[str, Any]]]:
    """Evaluate the all-user fixed-alpha Equal-Power + RZF-RSMA baseline.

    A held-out channel is static inside an episode, exactly like the learning
    environment, so the physical baseline is applied at every configured time
    slot and receives the same cumulative QoS accounting.
    """
    records: list[dict[str, Any]] = []
    sampler = TaskSampler(config.environment.sampler)
    steps = config.environment.episode_length
    for scenario_seed in config.test_seeds:
        sampler.reset(scenario_seed)
        scenario = sampler.sample()
        result = fixed_common_rsma(
            scenario.channels,
            total_power=config.environment.total_power,
            noise_variance=config.environment.noise_variance,
            common_power_fraction=config.common_power_fraction,
            private_scheme="rzf",
        )
        cumulative_rates = np.asarray(result.rates.user_rates, dtype=np.float64) * steps
        cumulative_gaps = np.maximum(scenario.qos_rate_targets * steps - cumulative_rates, 0.0)
        fairness = jain_fairness(cumulative_rates)
        reward = (
            config.environment.sum_rate_weight * float(np.sum(cumulative_rates))
            + config.environment.fairness_weight * fairness
            - config.environment.qos_gap_weight * float(np.sum(cumulative_gaps))
        )
        records.append(
            _episode_record(
                seed=scenario_seed,
                scenario=scenario,
                cumulative_rates=cumulative_rates,
                cumulative_qos_gaps=cumulative_gaps,
                service_fractions=np.ones(config.environment.sampler.num_users, dtype=np.float64),
                reward=reward,
                steps=steps,
                partition_switches=0,
                manager_decisions=0,
            )
        )
    return _metrics_from_episode_records(records), records


def _save_named_model(model: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination.with_suffix("")))
    return destination.with_suffix(".zip")


def _train_learning_method(
    *,
    strategy: MainMethodStrategy,
    config: MainExperimentConfig,
    seed: int,
    seed_directory: Path,
) -> tuple[Any, Any | None, dict[str, str]]:
    """Run the staged Worker/Manager pipeline for one training seed."""
    from hrl.training.train_hrl import (
        ManagerPPOConfig,
        StagedHRLConfig,
        train_manager_ppo,
        train_staged_hrl_worker,
    )

    checkpoints = seed_directory / "checkpoints"
    logs = seed_directory / "logs"
    _seed_everything(seed)
    worker_result = train_staged_hrl_worker(
        StagedHRLConfig(
            env=config.environment,
            manager=strategy.worker_manager,
            seed=seed,
            total_timesteps=config.worker_total_timesteps,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            policy_net_arch=config.policy_net_arch,
            checkpoint_directory=checkpoints,
            log_directory=logs / "worker",
        )
    )
    checkpoint_paths = {"worker": str(_save_named_model(worker_result.model, checkpoints / "worker"))}
    manager_model: Any | None = None
    if strategy.trains_manager:
        _seed_everything(seed)
        manager_result = train_manager_ppo(
            ManagerPPOConfig(
                worker=worker_result.model,
                env=config.environment,
                seed=seed,
                total_timesteps=config.manager_total_timesteps,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                policy_net_arch=config.policy_net_arch,
                checkpoint_directory=checkpoints,
                log_directory=logs / "manager",
            )
        )
        manager_model = manager_result.model
        checkpoint_paths["manager"] = str(_save_named_model(manager_model, checkpoints / "manager"))
    return worker_result.model, manager_model, checkpoint_paths


def _scenario_manifest(config: MainExperimentConfig) -> list[dict[str, Any]]:
    """Record held-out scene geometry so paired comparisons are auditable."""
    sampler = TaskSampler(config.environment.sampler)
    records: list[dict[str, Any]] = []
    for seed in config.test_seeds:
        sampler.reset(seed)
        scenario = sampler.sample()
        records.append(
            {
                "scenario_seed": seed,
                "ranges_m": scenario.ranges_m.tolist(),
                "angles_rad": scenario.angles_rad.tolist(),
                "near_field_mask": scenario.near_field_mask.astype(bool).tolist(),
                "qos_rate_targets": scenario.qos_rate_targets.tolist(),
            }
        )
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_ready(value.to_dict())
    return value


def _seed_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record["metrics"]
        row: dict[str, Any] = {
            "method": record["method"],
            "training_seed": record["training_seed"],
            "test_scenarios": metrics["episodes"],
        }
        row.update({key: value for key, value in metrics.items() if key != "mean_user_service_fractions"})
        for user, value in enumerate(metrics["mean_user_service_fractions"]):
            row[f"mean_user_service_fraction_{user}"] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result CSV.")
    headers = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in headers:
                headers.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_interval(
    values: np.ndarray, *, confidence: float, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _metric_columns(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    users = len(records[0]["metrics"]["mean_user_service_fractions"])
    return MAIN_METRICS + tuple(f"mean_user_service_fraction_{user}" for user in range(users))


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float:
    if metric.startswith("mean_user_service_fraction_"):
        index = int(metric.rsplit("_", 1)[1])
        return float(metrics["mean_user_service_fractions"][index])
    return float(metrics[metric])


def _summarize_records(
    records: Sequence[Mapping[str, Any]], config: MainExperimentConfig
) -> dict[str, Any]:
    rng = np.random.default_rng(config.analysis_seed)
    metric_columns = _metric_columns(records)
    summary: dict[str, Any] = {}
    for method in config.methods:
        method_records = [record for record in records if record["method"] == method]
        if not method_records:
            continue
        metrics: dict[str, Any] = {}
        for metric in metric_columns:
            values = np.asarray([_metric_value(record["metrics"], metric) for record in method_records], dtype=np.float64)
            lower, upper = _bootstrap_interval(
                values, confidence=config.confidence_level, samples=config.bootstrap_samples, rng=rng
            )
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "confidence_level": config.confidence_level,
                "confidence_interval": [lower, upper],
                "seeds": int(values.size),
            }
        summary[method] = {"metrics": metrics, "training_seeds": [record["training_seed"] for record in method_records]}
    return summary


def _sign_flip_p_value(
    differences: np.ndarray, *, permutations: int, rng: np.random.Generator
) -> tuple[float, str]:
    observed = abs(float(np.mean(differences)))
    if differences.size <= 16:
        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=differences.size)), dtype=np.float64)
        null = np.abs((signs * differences).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12)), "exact_sign_flip"
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(permutations, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (permutations + 1)), "monte_carlo_sign_flip"


def _paired_significance(
    records: Sequence[Mapping[str, Any]], config: MainExperimentConfig
) -> dict[str, Any]:
    rng = np.random.default_rng(config.analysis_seed + 1)
    metric_columns = _metric_columns(records)
    by_method_seed: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        by_method_seed.setdefault(str(record["method"]), {})[int(record["training_seed"])] = record["metrics"]
    comparisons: list[dict[str, Any]] = []
    for left, right in combinations(config.methods, 2):
        shared = tuple(sorted(set(by_method_seed.get(left, {})).intersection(by_method_seed.get(right, {}))))
        if not shared:
            continue
        metrics: dict[str, Any] = {}
        for metric in metric_columns:
            differences = np.asarray(
                [
                    _metric_value(by_method_seed[right][seed], metric)
                    - _metric_value(by_method_seed[left][seed], metric)
                    for seed in shared
                ],
                dtype=np.float64,
            )
            lower, upper = _bootstrap_interval(
                differences,
                confidence=config.confidence_level,
                samples=config.bootstrap_samples,
                rng=rng,
            )
            p_value, procedure = _sign_flip_p_value(
                differences, permutations=config.monte_carlo_permutations, rng=rng
            )
            metrics[metric] = {
                "direction": f"{right} - {left}",
                "paired_seeds": list(shared),
                "mean_difference": float(np.mean(differences)),
                "confidence_level": config.confidence_level,
                "confidence_interval": [lower, upper],
                "two_sided_p_value": p_value,
                "procedure": procedure,
            }
        comparisons.append({"left_method": left, "right_method": right, "metrics": metrics})
    return {"comparison_count": len(comparisons), "comparisons": comparisons}


def _plot_summary(summary: Mapping[str, Any], config: MainExperimentConfig, output_directory: Path) -> tuple[Path, ...]:
    """Generate compact mean-with-CI comparison plots in PNG and SVG form."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - project dependency protects normal installs.
        raise ImportError("matplotlib is required to generate main-experiment plots.") from error
    output_directory.mkdir(parents=True, exist_ok=True)
    plotted = (
        "mean_sum_rate",
        "mean_min_user_rate",
        "mean_jain_fairness",
        "mean_total_qos_gap",
        "qos_satisfaction_rate",
        "mean_near_user_rate",
        "mean_far_user_rate",
        "mean_user_service_fraction_0",
    )
    paths: list[Path] = []
    labels = [method for method in config.methods if method in summary]
    for metric in plotted:
        if not labels or any(metric not in summary[method]["metrics"] for method in labels):
            continue
        values = [float(summary[method]["metrics"][metric]["mean"]) for method in labels]
        lower = [float(summary[method]["metrics"][metric]["confidence_interval"][0]) for method in labels]
        upper = [float(summary[method]["metrics"][metric]["confidence_interval"][1]) for method in labels]
        error = np.vstack((np.asarray(values) - np.asarray(lower), np.asarray(upper) - np.asarray(values)))
        figure, axis = plt.subplots(figsize=(max(7.0, len(labels) * 1.75), 4.8))
        positions = np.arange(len(labels))
        axis.bar(positions, values, yerr=error, capsize=4, color="#287a8c", edgecolor="#1a4450")
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_ylabel(metric)
        axis.set_title(f"{config.experiment_name}: {metric}")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        stem = output_directory / metric
        for suffix in (".png", ".svg"):
            path = stem.with_suffix(suffix)
            figure.savefig(path, dpi=180)
            paths.append(path)
        plt.close(figure)
    return tuple(paths)


def _write_config_snapshot(config: MainExperimentConfig, destination: Path) -> None:
    """Write YAML when available, with JSON as a deterministic fallback."""
    payload = config.to_dict()
    try:
        import yaml

        destination.write_text(yaml.safe_dump(_json_ready(payload), sort_keys=False), encoding="utf-8")
    except ImportError:  # pragma: no cover
        destination.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _method_seed_payload(
    *,
    method: str,
    training_seed: int,
    metrics: MainEvaluationMetrics,
    episodes: list[dict[str, Any]],
    checkpoint_paths: Mapping[str, str],
    strategy: MainMethodStrategy,
) -> dict[str, Any]:
    return {
        "method": method,
        "training_seed": training_seed,
        "strategy": {
            "description": strategy.description,
            "trains_worker": strategy.trains_worker,
            "trains_manager": strategy.trains_manager,
        },
        "checkpoint_paths": dict(checkpoint_paths),
        "metrics": metrics.to_dict(),
        "episodes": episodes,
    }


def run_main_experiments(
    config: MainExperimentConfig | str | Path,
    *,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    output_directory: str | Path | None = None,
) -> MainExperimentResult:
    """Train/evaluate the requested main methods under one paired contract.

    The implementation is intentionally serial: PPO jobs are independent but
    serial execution keeps CPU usage, deterministic seed ordering and artifact
    ownership predictable for a one-command paper reproduction run.
    """
    loaded = load_main_experiment_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(loaded, MainExperimentConfig):
        raise TypeError("config must be MainExperimentConfig or a YAML path.")
    selected_methods = loaded.methods if methods is None else tuple(methods)
    selected_seeds = loaded.training_seeds if seeds is None else _validated_seeds(seeds, "seeds")
    unknown = set(selected_methods).difference(MAIN_EXPERIMENT_METHODS)
    if unknown:
        raise ValueError(f"Unknown main-experiment method(s): {sorted(unknown)}")
    if not selected_methods:
        raise ValueError("methods must not be empty.")
    effective = replace(
        loaded,
        methods=tuple(selected_methods),
        training_seeds=tuple(selected_seeds),
        output_directory=loaded.output_directory if output_directory is None else Path(output_directory),
    )
    root = effective.output_directory
    root.mkdir(parents=True, exist_ok=True)
    summary_directory = root / "summary"
    manifest_path = root / "manifest.json"
    snapshot_path = root / "config_snapshot.yaml"
    _write_config_snapshot(effective, snapshot_path)
    manifest: dict[str, Any] = {
        "experiment_name": effective.experiment_name,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_config": None if effective.source_config is None else str(effective.source_config),
        "configuration": effective.to_dict(),
        "config_snapshot": str(snapshot_path),
        "held_out_scenarios": _scenario_manifest(effective),
        "runs": [],
    }
    _write_json(manifest_path, manifest)
    run_records: list[dict[str, Any]] = []
    for method in effective.methods:
        for seed in effective.training_seeds:
            strategy = build_main_method_strategy(method, effective, seed=seed)
            seed_directory = root / method / f"seed_{seed}"
            seed_directory.mkdir(parents=True, exist_ok=True)
            _write_json(
                seed_directory / "training_config.json",
                {"method": method, "training_seed": seed, "configuration": effective.to_dict()},
            )
            checkpoint_paths: dict[str, str] = {}
            if strategy.trains_worker:
                worker, learned_manager, checkpoint_paths = _train_learning_method(
                    strategy=strategy,
                    config=effective,
                    seed=seed,
                    seed_directory=seed_directory,
                )
                if strategy.trains_manager:
                    assert learned_manager is not None
                    metrics, episodes = evaluate_main_policy(
                        worker,
                        manager=learned_manager,
                        environment=effective.environment,
                        test_seeds=effective.test_seeds,
                        learned_manager=True,
                    )
                else:
                    metrics, episodes = evaluate_main_policy(
                        worker,
                        manager=strategy.worker_manager,
                        environment=effective.environment,
                        test_seeds=effective.test_seeds,
                    )
            else:
                metrics, episodes = evaluate_equal_power_rzf_rsma(effective)
            payload = _method_seed_payload(
                method=method,
                training_seed=seed,
                metrics=metrics,
                episodes=episodes,
                checkpoint_paths=checkpoint_paths,
                strategy=strategy,
            )
            evaluation_path = seed_directory / "evaluation.json"
            _write_json(evaluation_path, payload)
            run_record = {
                "method": method,
                "training_seed": seed,
                "evaluation_path": str(evaluation_path),
                "checkpoint_paths": checkpoint_paths,
                "metrics": metrics.to_dict(),
            }
            run_records.append(run_record)
            manifest["runs"].append(run_record)
            _write_json(manifest_path, manifest)
    summary = _summarize_records(run_records, effective)
    significance = _paired_significance(run_records, effective)
    seed_rows = _seed_rows(run_records)
    results_csv_path = summary_directory / "results.csv"
    _write_csv(results_csv_path, seed_rows)
    results_payload = {
        "experiment_name": effective.experiment_name,
        "configuration": effective.to_dict(),
        "records": run_records,
        "summary": summary,
    }
    results_json_path = root / "results.json"
    _write_json(results_json_path, results_payload)
    _write_json(summary_directory / "results.json", results_payload)
    significance_path = summary_directory / "significance.json"
    _write_json(significance_path, significance)
    plot_paths = _plot_summary(summary, effective, summary_directory / "plots")
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["results_json"] = str(results_json_path)
    manifest["results_csv"] = str(results_csv_path)
    manifest["significance"] = str(significance_path)
    manifest["plots"] = [str(path) for path in plot_paths]
    _write_json(manifest_path, manifest)
    return MainExperimentResult(
        output_directory=root,
        manifest_path=manifest_path,
        results_json_path=results_json_path,
        results_csv_path=results_csv_path,
        significance_path=significance_path,
        plot_paths=plot_paths,
    )
