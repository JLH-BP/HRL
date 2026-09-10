"""为1-HRL公开Flat RL训练和评估入口点。"""

from .analyze_meta_hrl import MetaPPOAnalysisResult, analyze_meta_ppo_benchmark
from .analyze_sensitivity_meta_hrl import MetaPPOSensitivityAnalysisResult, analyze_meta_ppo_sensitivity
from .benchmark_meta_hrl import (
    MetaPPOBenchmarkConfig,
    MetaPPOBenchmarkResult,
    run_meta_ppo_benchmark,
)
from .evaluate import EvaluationMetrics, evaluate_flat_policy, evaluate_physical_baselines
from .evaluate_meta_hrl import (
    MetaAdaptationCurve,
    MetaAdaptationPoint,
    evaluate_meta_adaptation,
)
from .evaluate_hrl import StagedHRLEvaluationMetrics, evaluate_staged_hrl_worker
from .sensitivity_meta_hrl import MetaPPOSensitivityConfig, MetaPPOSensitivityResult, run_meta_ppo_sensitivity
from .significance_meta_hrl import MetaPPOSignificanceResult, analyze_meta_ppo_significance
from .train_flat_rl import FlatPPOConfig, FlatPPOTrainingResult, train_flat_ppo
from .train_meta_hrl import (
    MetaPPOConfig,
    MetaPPOTrainingResult,
    TaskCyclingContextualWorkerEnv,
    train_meta_ppo,
)
from .train_hrl import (
    ManagerPPOConfig,
    ManagerPPOTrainingResult,
    StagedHRLConfig,
    StagedHRLTrainingResult,
    train_manager_ppo,
    train_staged_hrl_worker,
)
from .main_experiments import (
    MAIN_EXPERIMENT_METHODS,
    MAIN_METRICS,
    MainEvaluationMetrics,
    MainExperimentConfig,
    MainExperimentResult,
    MainMethodStrategy,
    build_main_method_strategy,
    evaluate_equal_power_rzf_rsma,
    evaluate_main_policy,
    load_main_experiment_config,
    run_main_experiments,
)

__all__ = [
    "EvaluationMetrics",
    "MAIN_EXPERIMENT_METHODS",
    "MAIN_METRICS",
    "FlatPPOConfig",
    "MetaPPOAnalysisResult",
    "MetaPPOBenchmarkConfig",
    "MetaPPOBenchmarkResult",
    "MetaPPOSignificanceResult",
    "MetaPPOSensitivityAnalysisResult",
    "MetaPPOSensitivityConfig",
    "MetaPPOSensitivityResult",
    "ManagerPPOConfig",
    "ManagerPPOTrainingResult",
    "MainEvaluationMetrics",
    "MainExperimentConfig",
    "MainExperimentResult",
    "MainMethodStrategy",
    "MetaAdaptationCurve",
    "MetaAdaptationPoint",
    "MetaPPOConfig",
    "MetaPPOTrainingResult",
    "StagedHRLConfig",
    "StagedHRLEvaluationMetrics",
    "StagedHRLTrainingResult",
    "TaskCyclingContextualWorkerEnv",
    "analyze_meta_ppo_benchmark",
    "analyze_meta_ppo_significance",
    "analyze_meta_ppo_sensitivity",
    "build_main_method_strategy",
    "evaluate_equal_power_rzf_rsma",
    "evaluate_flat_policy",
    "evaluate_main_policy",
    "evaluate_meta_adaptation",
    "evaluate_physical_baselines",
    "evaluate_staged_hrl_worker",
    "run_meta_ppo_benchmark",
    "run_meta_ppo_sensitivity",
    "run_main_experiments",
    "load_main_experiment_config",
    "train_flat_ppo",
    "train_manager_ppo",
    "train_meta_ppo",
    "train_staged_hrl_worker",
]
