"""负责 Meta-PPO 的训练、基准实验、显著性检验和敏感性分析；。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hrl.training import (
    FlatPPOConfig,
    MetaPPOBenchmarkConfig,
    MetaPPOConfig,
    MetaPPOSensitivityConfig,
    analyze_meta_ppo_benchmark,
    analyze_meta_ppo_sensitivity,
    analyze_meta_ppo_significance,
    run_meta_ppo_benchmark,
    run_meta_ppo_sensitivity,
    train_flat_ppo,
    train_meta_ppo,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a comma-separated float list") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _add_common_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=_csv_ints, default=(11, 29, 47))
    parser.add_argument("--task-ids", type=_csv_ints, default=tuple(range(16)))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--adaptation-episodes", type=_csv_ints, default=(0, 1, 5, 10))
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--task-seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without executing a training or analysis workflow."""
    parser = argparse.ArgumentParser(prog="meta-hrl", description="META_HRL experiment workflows")
    commands = parser.add_subparsers(dest="command", required=True)

    flat_train = commands.add_parser("train-flat", help="train the Flat PPO baseline and export its evaluation curve")
    flat_train.add_argument("--output-directory", type=Path, default=Path("outputs/flat_ppo_train"))
    flat_train.add_argument("--seed", type=int, default=42)
    flat_train.add_argument("--timesteps", type=int, default=100_000)
    flat_train.add_argument("--n-steps", type=int, default=256)
    flat_train.add_argument("--batch-size", type=int, default=64)
    flat_train.add_argument("--n-epochs", type=int, default=10)
    flat_train.add_argument("--evaluation-interval", type=int, default=5_000)
    flat_train.add_argument("--evaluation-episodes", type=int, default=100)
    flat_train.add_argument("--evaluation-seed", type=int, default=10_000)

    train = commands.add_parser("train-meta", help="train one context-conditioned Meta-PPO Worker")
    train.add_argument("--output-directory", type=Path, default=Path("outputs/meta_ppo_train"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--task-ids", type=_csv_ints, default=tuple(range(16)))
    train.add_argument("--timesteps", type=int, default=100_000)
    train.add_argument("--n-steps", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--n-epochs", type=int, default=10)
    train.add_argument("--raw-context", action="store_true", help="disable the learned context feature branch")

    benchmark = commands.add_parser("benchmark", help="run learned-versus-raw paired Meta-PPO benchmark")
    _add_common_run_options(benchmark)
    benchmark.add_argument("--output-directory", type=Path, default=Path("outputs/meta_ppo_benchmark"))

    sensitivity = commands.add_parser("sensitivity", help="run context/OOD sensitivity grid")
    _add_common_run_options(sensitivity)
    sensitivity.add_argument("--context-capacities", type=_csv_ints, default=(16, 64, 128))
    sensitivity.add_argument("--context-feature-dims", type=_csv_ints, default=(16, 32, 64))
    sensitivity.add_argument("--ood-shift-scales", type=_csv_floats, default=(0.5, 1.0, 1.5))
    sensitivity.add_argument("--output-directory", type=Path, default=Path("outputs/meta_ppo_sensitivity"))

    analysis = commands.add_parser("analyze-benchmark", help="export benchmark CSV/SVG/difference artifacts")
    analysis.add_argument("benchmark_path", type=Path)
    analysis.add_argument("--output-directory", type=Path)
    analysis.add_argument("--metrics", type=lambda value: tuple(item.strip() for item in value.split(",") if item.strip()), default=("mean_reward", "mean_sum_rate"))

    significance = commands.add_parser("significance", help="run paired bootstrap and sign-flip inference")
    significance.add_argument("benchmark_path", type=Path)
    significance.add_argument("--output-path", type=Path)
    significance.add_argument("--bootstrap-samples", type=int, default=10_000)
    significance.add_argument("--permutations", type=int, default=100_000)
    significance.add_argument("--seed", type=int, default=42)

    sensitivity_analysis = commands.add_parser("analyze-sensitivity", help="export sensitivity CSV/SVG/bootstrap artifacts")
    sensitivity_analysis.add_argument("sensitivity_path", type=Path)
    sensitivity_analysis.add_argument("--output-directory", type=Path)
    sensitivity_analysis.add_argument("--metric", default="mean_reward")
    sensitivity_analysis.add_argument("--bootstrap-samples", type=int, default=10_000)
    sensitivity_analysis.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected reproducible Meta-PPO experiment command."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "train-flat":
        result = train_flat_ppo(FlatPPOConfig(
            seed=arguments.seed,
            total_timesteps=arguments.timesteps,
            n_steps=arguments.n_steps,
            batch_size=arguments.batch_size,
            n_epochs=arguments.n_epochs,
            checkpoint_directory=arguments.output_directory / "checkpoints",
            log_directory=arguments.output_directory / "logs",
            evaluation_directory=arguments.output_directory / "evaluations",
            evaluation_interval=arguments.evaluation_interval,
            evaluation_episodes=arguments.evaluation_episodes,
            evaluation_seed=arguments.evaluation_seed,
        ))
        print(result.model_path)
        print(result.evaluation_csv_path)
        print(result.evaluation_svg_path)
    elif arguments.command == "train-meta":
        result = train_meta_ppo(MetaPPOConfig(
            seed=arguments.seed, task_ids=arguments.task_ids, total_timesteps=arguments.timesteps,
            n_steps=arguments.n_steps, batch_size=arguments.batch_size, n_epochs=arguments.n_epochs,
            use_learned_context_encoder=not arguments.raw_context,
            checkpoint_directory=arguments.output_directory / "checkpoints",
            log_directory=arguments.output_directory / "logs",
        ))
        print(result.model_path)
    elif arguments.command == "benchmark":
        result = run_meta_ppo_benchmark(MetaPPOBenchmarkConfig(
            seeds=arguments.seeds, training_task_ids=arguments.task_ids, evaluation_task_ids=arguments.task_ids,
            total_timesteps=arguments.timesteps, n_steps=arguments.n_steps, batch_size=arguments.batch_size,
            n_epochs=arguments.n_epochs, adaptation_episodes=arguments.adaptation_episodes,
            evaluation_episodes=arguments.evaluation_episodes, task_seed=arguments.task_seed,
            output_directory=arguments.output_directory,
        ))
        print(result.result_path)
    elif arguments.command == "sensitivity":
        result = run_meta_ppo_sensitivity(MetaPPOSensitivityConfig(
            seeds=arguments.seeds, training_task_ids=arguments.task_ids, evaluation_task_ids=arguments.task_ids,
            context_capacities=arguments.context_capacities, context_feature_dims=arguments.context_feature_dims,
            ood_shift_scales=arguments.ood_shift_scales, total_timesteps=arguments.timesteps,
            n_steps=arguments.n_steps, batch_size=arguments.batch_size, n_epochs=arguments.n_epochs,
            adaptation_episodes=arguments.adaptation_episodes, evaluation_episodes=arguments.evaluation_episodes,
            task_seed=arguments.task_seed, output_directory=arguments.output_directory,
        ))
        print(result.result_path)
    elif arguments.command == "analyze-benchmark":
        result = analyze_meta_ppo_benchmark(arguments.benchmark_path, output_directory=arguments.output_directory, metrics=arguments.metrics)
        print(result.csv_path)
    elif arguments.command == "significance":
        result = analyze_meta_ppo_significance(
            arguments.benchmark_path, output_path=arguments.output_path,
            bootstrap_samples=arguments.bootstrap_samples, monte_carlo_permutations=arguments.permutations,
            seed=arguments.seed,
        )
        print(result.result_path)
    else:
        result = analyze_meta_ppo_sensitivity(
            arguments.sensitivity_path, output_directory=arguments.output_directory, metric=arguments.metric,
            bootstrap_samples=arguments.bootstrap_samples, seed=arguments.seed,
        )
        print(result.summary_path)
    return 0
