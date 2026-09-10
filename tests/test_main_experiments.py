"""Smoke and contract tests for the paired five-method main experiment runner."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hrl.agents.high_level_policy import ExplicitFixedPartitionManager, NearFieldFirstSequentialManager
from hrl.cli import main
from hrl.envs import HierarchicalRSMAEnvConfig
from hrl.training.main_experiments import (
    MainExperimentConfig,
    build_main_method_strategy,
    evaluate_equal_power_rzf_rsma,
    evaluate_main_policy,
    load_main_experiment_config,
    run_main_experiments,
)


class ZeroWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.zeros(18, dtype=np.float32), None


def _small_environment(*, episode_length: int = 2, interval: int = 1) -> HierarchicalRSMAEnvConfig:
    base = HierarchicalRSMAEnvConfig()
    sampler = replace(base.sampler, near_field_user_count_range=(3, 3))
    return replace(base, sampler=sampler, episode_length=episode_length, high_level_interval=interval)


def _physical_config(output_directory: Path) -> MainExperimentConfig:
    return MainExperimentConfig(
        methods=("equal_power_rzf_rsma",),
        training_seeds=(11, 29),
        test_seeds=(10011, 10029),
        environment=_small_environment(),
        output_directory=output_directory,
        bootstrap_samples=20,
        monte_carlo_permutations=20,
    )


def test_main_yaml_declares_the_explicit_mixed_field_contract() -> None:
    config = load_main_experiment_config("configs/experiments/main_mixed_near_far.yaml")

    assert config.environment.sampler.near_field_user_count_range == (1, 5)
    assert config.fixed_partition == ((0, 1), (2, 3), (4, 5))
    assert config.methods == (
        "equal_power_rzf_rsma",
        "fixed_group_ppo",
        "near_first_ppo",
        "heuristic_group_ppo",
        "hrl_joint_scheduler",
    )


def test_same_held_out_seed_set_produces_identical_physical_evaluation() -> None:
    config = _physical_config(Path("unused"))
    first_metrics, first_episodes = evaluate_equal_power_rzf_rsma(config)
    second_metrics, second_episodes = evaluate_equal_power_rzf_rsma(config)

    assert first_episodes == second_episodes
    assert first_metrics == second_metrics
    assert [episode["scenario_seed"] for episode in first_episodes] == list(config.test_seeds)


def test_fixed_and_near_first_use_same_scenarios_but_real_service_masks() -> None:
    environment = _small_environment(episode_length=4, interval=1)
    fixed_metrics, fixed_episodes = evaluate_main_policy(
        ZeroWorker(),
        manager=ExplicitFixedPartitionManager(partition=((0, 1), (2, 3), (4, 5))),
        environment=environment,
        test_seeds=(71, 73),
    )
    near_metrics, near_episodes = evaluate_main_policy(
        ZeroWorker(),
        manager=NearFieldFirstSequentialManager(),
        environment=environment,
        test_seeds=(71, 73),
    )

    assert [episode["scenario_seed"] for episode in fixed_episodes] == [71, 73]
    assert [episode["scenario_seed"] for episode in near_episodes] == [71, 73]
    np.testing.assert_allclose(fixed_metrics.mean_user_service_fractions, np.ones(6))
    # Exactly alternating near/far intervals makes every user eligible for half
    # of the four physical slots, proving this is not a partition-only baseline.
    np.testing.assert_allclose(near_metrics.mean_user_service_fractions, np.full(6, 0.5))
    assert near_metrics.mean_manager_decisions == 4.0


def test_main_runner_writes_manifest_seed_results_summary_plots_and_significance(tmp_path: Path) -> None:
    result = run_main_experiments(_physical_config(tmp_path / "main"))

    assert result.manifest_path.is_file()
    assert result.results_json_path.is_file()
    assert result.results_csv_path.is_file()
    assert result.significance_path.is_file()
    assert result.plot_paths
    assert all(path.is_file() for path in result.plot_paths)
    payload = json.loads(result.results_json_path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 2
    for record in payload["records"]:
        assert record["method"] == "equal_power_rzf_rsma"
        assert [episode["scenario_seed"] for episode in json.loads(Path(record["evaluation_path"]).read_text(encoding="utf-8"))["episodes"]] == [10011, 10029]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert len(manifest["held_out_scenarios"]) == 2
    assert (result.output_directory / "config_snapshot.yaml").is_file()


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
def test_same_training_seed_reproduces_fixed_group_worker_evaluation(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    common = dict(
        methods=("fixed_group_ppo",),
        training_seeds=(11,),
        test_seeds=(10011,),
        environment=_small_environment(),
        worker_total_timesteps=8,
        manager_total_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16,),
        bootstrap_samples=10,
        monte_carlo_permutations=10,
    )
    first = run_main_experiments(MainExperimentConfig(**common, output_directory=tmp_path / "first"))
    second = run_main_experiments(MainExperimentConfig(**common, output_directory=tmp_path / "second"))
    first_payload = json.loads((first.output_directory / "fixed_group_ppo" / "seed_11" / "evaluation.json").read_text(encoding="utf-8"))
    second_payload = json.loads((second.output_directory / "fixed_group_ppo" / "seed_11" / "evaluation.json").read_text(encoding="utf-8"))

    assert first_payload["metrics"] == second_payload["metrics"]
    assert first_payload["episodes"] == second_payload["episodes"]


def test_cli_dispatches_main_experiment_runner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("yaml")
    import yaml

    config = _physical_config(tmp_path / "cli_output")
    payload = config.to_dict()
    payload["output_directory"] = str(config.output_directory)
    config_path = tmp_path / "main.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert main(["run-main-experiments", "--config", str(config_path)]) == 0
    output_paths = [Path(line) for line in capsys.readouterr().out.splitlines() if line]
    assert len(output_paths) == 4
    assert all(path.is_file() for path in output_paths)


def test_strategy_factories_keep_fixed_group_non_singleton_and_hrl_staged() -> None:
    config = _physical_config(Path("unused"))
    fixed = build_main_method_strategy("fixed_group_ppo", config, seed=11)
    hrl = build_main_method_strategy("hrl_joint_scheduler", config, seed=11)

    fixed_decision = fixed.worker_manager.select(
        # The concrete scenario is unnecessary to verify the chosen explicit
        # partition, but the manager deliberately validates one, so sample it.
        _scenario_for_config(config),
        _candidates_for_config(config),
    )
    assert fixed_decision.partition == ((0, 1), (2, 3), (4, 5))
    assert fixed_decision.candidate_index != 0
    assert hrl.trains_worker and hrl.trains_manager


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
@pytest.mark.parametrize(
    ("method", "expects_manager_checkpoint"),
    [
        ("fixed_group_ppo", False),
        ("near_first_ppo", False),
        ("heuristic_group_ppo", False),
        ("hrl_joint_scheduler", True),
    ],
)
def test_each_learning_main_method_smoke_trains_and_writes_required_checkpoints(
    tmp_path: Path, method: str, expects_manager_checkpoint: bool
) -> None:
    pytest.importorskip("stable_baselines3")
    config = MainExperimentConfig(
        methods=(method,),
        training_seeds=(11,),
        test_seeds=(10011,),
        environment=_small_environment(),
        worker_total_timesteps=8,
        manager_total_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16,),
        bootstrap_samples=10,
        monte_carlo_permutations=10,
        output_directory=tmp_path / method,
    )

    result = run_main_experiments(config)
    evaluation = json.loads(
        (result.output_directory / method / "seed_11" / "evaluation.json").read_text(encoding="utf-8")
    )
    checkpoints = evaluation["checkpoint_paths"]
    assert Path(checkpoints["worker"]).is_file()
    assert ("manager" in checkpoints) is expects_manager_checkpoint
    if expects_manager_checkpoint:
        assert Path(checkpoints["manager"]).is_file()
    assert evaluation["episodes"][0]["scenario_seed"] == 10011


def _scenario_for_config(config: MainExperimentConfig):
    from hrl.envs.task_sampler import TaskSampler

    sampler = TaskSampler(config.environment.sampler)
    sampler.reset(17)
    return sampler.sample()


def _candidates_for_config(config: MainExperimentConfig):
    from hrl.grouping.candidate_groups import enumerate_candidate_partitions

    return enumerate_candidate_partitions(num_users=config.environment.sampler.num_users)
