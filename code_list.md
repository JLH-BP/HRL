<!-- 维护 META_HRL 当前源码、接口与测试的索引。 -->
# META_HRL 代码清单

本文档记录仓库当前的模块职责、主要公共接口和测试覆盖范围。它反映当前已实现状态；
历史阶段计划请阅读 `docs/meta_hrl_task_framework.md`，首次阅读代码请阅读 `代码说明.md`。

## 1. 全局约定

| 项目 | 约定 |
| --- | --- |
| 默认系统 | 128 阵元 ULA 下行链路，`K=6` 个单天线用户 |
| 信道矩阵 | `channels` 形状 `(K, M)`，每行对应一个用户信道 |
| 速率与功率 | 速率单位 bit/s/Hz；功率和噪声均为线性归一化值 |
| 单公共流动作 | `2K+1=13` 个 logits：6 私有功率、1 公共功率、6 公共率 |
| 组 RSMA Worker 动作 | `3K=18` 个 logits：6 私有功率、6 组公共功率、6 组公共率 |
| 分区表示 | 已排序用户元组的规范分区，6 用户可枚举 203 个候选 |
| Gymnasium 返回 | `reset() -> (observation, info)`；`step() -> (observation, reward, terminated, truncated, info)` |

## 2. 项目入口与配置

| 文件 | 状态 | 内容 |
| --- | --- | --- |
| `pyproject.toml` | 已实现 | 包元数据、Python >=3.10、Gymnasium/NumPy/SB3 依赖及 `meta-hrl` CLI 入口 |
| `README.md` | 已更新 | 项目概览、安装、示例、实验命令与已运行实验结论 |
| `代码说明.md` | 已更新 | 面向首次读者的代码导读与数据流说明 |
| `具体工作.md` | 已更新 | 面向 SCI 二区投稿的研究、实验和写作工作清单 |
| `src/meta_hrl/cli.py` | 已实现 | `train-meta`、`benchmark`、`sensitivity`、分析和显著性命令 |
| `configs/channel/rician_near_far.yaml` | 已定义 | 28 GHz、阵列、用户区域、近远场和莱斯信道设定 |
| `configs/rsma/default.yaml` | 已定义 | 一层 RSMA 功率、噪声、预编码与动作约定 |
| `configs/environment/default.yaml` | 已定义 | 单步环境观测、动作、场景和奖励默认值 |
| `configs/training/default.yaml` | 已定义 | Flat PPO 和两阶段 HRL 的训练超参数 |
| `configs/experiments/meta_hrl_mixed_near_far.yaml` | 已定义 | 元任务维度及训练/验证/OOD 划分说明 |

YAML 文件当前是可审阅的实验规范；训练 API 主要通过 Python dataclass 和 CLI 参数配置，
并非自动加载全部 YAML。

## 3. 信道模块：`src/meta_hrl/channel/`

| 文件 | 核心接口 | 作用 |
| --- | --- | --- |
| `geometry.py` | `ULAConfig` | ULA 波长、阵元坐标、孔径、Rayleigh 距离等物理量 |
| `geometry.py` | `polar_to_cartesian()`、`cartesian_to_polar()` | broadside 参考系下的二维坐标转换 |
| `geometry.py` | `element_to_user_distances_m()`、`classify_near_field()` | 阵元-用户距离和近远场分类 |
| `far_field.py` | `ula_los_steering_vector()` | 单位范数远场平面波 LoS 导向矢量 |
| `near_field.py` | `ula_near_field_los_steering_vector()` | 单位范数近场球面波相位导向矢量 |
| `rician.py` | `sample_rician_channel()` | 近/远场 LoS 与 NLoS 项混合的莱斯信道采样 |
| `validation.py` | `steering_vector_norms()`、`compare_near_and_far_field()` | 阵列响应与近远场差异诊断 |
| `validation.py` | `user_channel_correlation_matrix()`、`summarize_channel_diagnostics()` | 用户信道相关性和统计摘要 |

## 4. RSMA 与分组模块

### `src/meta_hrl/rsma/`

| 文件 | 核心接口 | 作用 |
| --- | --- | --- |
| `constraints.py` | `ResourceAllocation`、`ResourceFeasibility` | 单公共流资源分配及其可行性数据结构 |
| `constraints.py` | `project_action_to_resource_allocation()` | 将 13 维 logits 投影为满足总功率和公共率预算的分配 |
| `constraints.py` | `project_nonnegative_simplex()` | 非负单纯形投影工具 |
| `precoding.py` | `mrt_private_directions()`、`rzf_private_directions()` | MRT/RZF 私有流方向 |
| `precoding.py` | `one_layer_rsma_precoders()` | 构造带功率权重的一层 RSMA 波束矩阵 |
| `signal_model.py` | `one_layer_rsma_signal_terms()` | 计算公共流和 SIC 后私有流的信号/干扰项 |
| `rate.py` | `one_layer_rsma_rates()` | 由 SINR 计算公共率、私有率、用户率和总速率 |
| `baselines.py` | `equal_power_mrt()`、`equal_power_rzf()` | 等功率 SDMA 基线 |
| `baselines.py` | `fixed_common_rsma()` | 固定公共功率的一层 RSMA 基线 |
| `group_rsma.py` | `GroupRSMAAllocation`、`project_group_action()` | 将 18 维 Worker 动作映射为组 RSMA 的可行资源 |
| `group_rsma.py` | `active_group_slots()` | 用组内最小用户编号固定组公共流槽位 |
| `group_rsma.py` | `group_rsma_precoders()`、`group_rsma_rates()` | 组公共流预编码与速率计算 |

### `src/meta_hrl/grouping/`

| 文件 | 核心接口 | 作用 |
| --- | --- | --- |
| `candidate_groups.py` | `UserGroup`、`UserPartition` | 分组和分区类型别名 |
| `candidate_groups.py` | `canonicalize_partition()` | 验证并规范化分区，消除组标签置换 |
| `candidate_groups.py` | `enumerate_candidate_partitions()` | 枚举满足约束的无标签候选用户分区 |
| `heuristic.py` | `GroupingWeights`、`pairwise_grouping_affinity()` | 组合几何、近远场标签和信道相关性的亲和度 |
| `heuristic.py` | `score_partition()`、`select_heuristic_partition()` | 候选分区打分与确定性启发式选择 |
| `metrics.py` | `pending()` | 预留的分组质量指标模块 |

## 5. 场景与环境模块：`src/meta_hrl/envs/`

| 文件 | 核心接口 | 作用 |
| --- | --- | --- |
| `task_sampler.py` | `ScenarioSamplerConfig` | 场景采样范围与系统参数 |
| `task_sampler.py` | `RSMAScenario` | 一个静态场景：CSI、几何、路径损耗、K 因子和 QoS |
| `task_sampler.py` | `TaskSampler` | 以独立 RNG 生成可复现混合近远场场景 |
| `observation_encoder.py` | `observation_size()`、`encode_observation()` | 将复 CSI 和元数据编码为 `float32` 观察向量 |
| `rsma_env.py` | `RSMAEnvConfig`、`OneStepRSMAEnv` | 单步、一层 RSMA Gymnasium 环境，供 Flat PPO 使用 |
| `rsma_env.py` | `jain_fairness()` | Jain 公平性计算；全零速率时返回 0 |
| `hierarchical_env.py` | `HierarchicalRSMAEnvConfig`、`HierarchicalRSMAEnv` | 多步组 RSMA 环境，显式分离 Manager 和 Worker 接口 |
| `worker_training_env.py` | `WorkerTrainingEnv` | 在固定或启发式 Manager 下训练 Worker 的包装环境 |
| `worker_training_env.py` | `partition_membership_features()` | 将分区转为同组关系观察特征 |
| `manager_training_env.py` | `ManagerTrainingEnv` | 冻结 Worker 后训练离散分区 Manager 的环境 |
| `meta_task_sampler.py` | `MetaTaskSpec`、`MetaTaskSampler` | 创建 train/validation/OOD 的慢变化无线任务 |
| `contextual_worker_env.py` | `ContextualWorkerTrainingEnv` | 将按 task ID 隔离的 transition context 追加给 Worker 观察 |
| `envs/__init__.py` | 公共重导出 | 对外导出稳定环境与采样 API |

### 环境调用关系

```text
TaskSampler -> RSMAScenario -> encode_observation
                              |
OneStepRSMAEnv: action -> 单公共流 RSMA -> reward

HierarchicalRSMAEnv:
  Manager: set_manager_action(partition_index)
  Worker:  step(worker_action) -> 组 RSMA -> reward

ContextualWorkerTrainingEnv:
  MetaTaskSampler + WorkerTrainingEnv + TaskContextBuffer
  -> 原始 Worker 观察 + 7 维 context
```

`OneStepRSMAEnv.step()` 先根据功率 logits 计算物理公共率上限，再对公共率 logits 做投影。
因此公共率分配不会超过同一动作功率决定的可解码公共率。

## 6. 智能体模块：`src/meta_hrl/agents/`

| 文件 | 核心接口 | 作用 |
| --- | --- | --- |
| `high_level_policy.py` | `FixedPartitionManager` | 固定输出指定候选分区 |
| `high_level_policy.py` | `HeuristicPartitionManager` | 根据场景调用启发式分组，输出候选索引 |
| `high_level_policy.py` | `select_partition_from_scores()` | 由候选分数选择 Manager 决策 |
| `low_level_policy.py` | `validate_worker_action()` | 校验 `3K` 维 Worker logits 的形状、数值和边界 |
| `replay_buffer.py` | `ContextTransition`、`TaskContextBuffer` | 容量受限且按 task ID 隔离的 transition 缓冲区 |
| `meta_context.py` | `META_CONTEXT_SIZE`、`MetaContextEncoder` | 从最近 transition 得到 7 维确定性任务 context |
| `learned_context.py` | `ContextConditionedFeaturesExtractor` | 分别用 MLP 编码原始 Worker 状态和 context 的 SB3 特征提取器 |

## 7. 训练、评估与分析：`src/meta_hrl/training/`

| 文件 | 核心入口 | 作用 |
| --- | --- | --- |
| `train_flat_rl.py` | `FlatPPOConfig`、`train_flat_ppo()` | 训练单步环境上的连续动作 PPO |
| `evaluate.py` | `evaluate_flat_policy()` | 汇总 Flat PPO 的 reward、速率、公平性和 QoS 指标 |
| `evaluate.py` | `evaluate_physical_baselines()` | 在匹配场景上评估 RZF SDMA 与固定公共 RSMA |
| `train_hrl.py` | `StagedHRLConfig`、`train_staged_hrl_worker()` | 固定/启发式 Manager 下训练 PPO Worker |
| `train_hrl.py` | `ManagerPPOConfig`、`train_manager_ppo()` | 冻结 Worker 后训练离散 PPO Manager |
| `evaluate_hrl.py` | `evaluate_staged_hrl_worker()` | 评估 Worker/Manager 组合与切换信息 |
| `train_meta_hrl.py` | `MetaPPOConfig`、`train_meta_ppo()` | 在 task ID 之间按 episode 轮换训练 context-conditioned PPO Worker |
| `train_meta_hrl.py` | `TaskCyclingContextualWorkerEnv` | 为 SB3 提供任务轮换的 Gymnasium 环境 |
| `evaluate_meta_hrl.py` | `evaluate_meta_adaptation()` | 计算 validation/OOD 的 support-episode 适应曲线 |
| `benchmark_meta_hrl.py` | `run_meta_ppo_benchmark()` | learned/raw context 的配对多 seed 基准 |
| `significance_meta_hrl.py` | `analyze_meta_ppo_significance()` | bootstrap CI 和双侧 sign-flip 检验 |
| `sensitivity_meta_hrl.py` | `run_meta_ppo_sensitivity()` | context 容量、特征维度和 OOD shift 三因素网格 |
| `analyze_meta_hrl.py` | `analyze_meta_ppo_benchmark()` | 导出基准 CSV、SVG 曲线和配对差异 JSON |
| `analyze_sensitivity_meta_hrl.py` | `analyze_meta_ppo_sensitivity()` | 导出敏感性 CSV、SVG 因子趋势和摘要 JSON |
| `training/__init__.py` | 公共重导出 | 集中暴露训练、评估和分析 API |

## 8. 工具、文档和结果

| 路径 | 状态 | 作用 |
| --- | --- | --- |
| `src/meta_hrl/utils/seed.py` | 骨架 | 预留跨库随机种子工具 |
| `src/meta_hrl/utils/logging.py` | 骨架 | 预留统一日志工具 |
| `src/meta_hrl/utils/normalization.py` | 骨架 | 预留归一化工具 |
| `src/meta_hrl/utils/checkpoint.py` | 骨架 | 预留通用 checkpoint 工具 |
| `docs/meta_hrl_task_framework.md` | 设计文档 | 研究问题、系统模型、阶段计划和实验建议 |
| `docs/meta_hrl_task_framework.docx` | 文档产物 | Markdown 框架文档的 Word 版本 |
| `scripts/generate_task_framework_docx.py` | 已实现 | 生成 DOCX 文档 |
| `outputs/` | 实验产物 | checkpoint、TensorBoard 日志、JSON/CSV/SVG 结果 |

## 9. 测试：`tests/`

| 测试范围 | 对应测试文件 |
| --- | --- |
| 包与 CLI | `test_package.py`、`test_cli.py` |
| 信道与几何 | `test_geometry.py`、`test_far_field.py`、`test_near_field.py`、`test_rician.py`、`test_validation.py` |
| 单公共流 RSMA | `test_rsma_constraints.py`、`test_rsma_precoding.py`、`test_rsma_signal_model.py`、`test_rsma_rate.py`、`test_rsma_baselines.py` |
| 场景、观测和单步环境 | `test_task_sampler.py`、`test_observation_encoder.py`、`test_rsma_env.py` |
| 分组与组 RSMA | `test_candidate_groups.py`、`test_grouping_heuristic.py`、`test_group_rsma.py` |
| HRL 环境与训练 | `test_hrl_policy_interfaces.py`、`test_staged_hrl.py`、`test_manager_training_env.py`、`test_train_hrl.py`、`test_train_manager.py`、`test_evaluate_hrl.py` |
| Meta-PPO | `test_meta_context.py`、`test_learned_context.py`、`test_train_meta_hrl.py`、`test_evaluate_meta_hrl.py` |
| Benchmark 与分析 | `test_benchmark_meta_hrl.py`、`test_significance_meta_hrl.py`、`test_sensitivity_meta_hrl.py`、`test_analyze_meta_hrl.py`、`test_analyze_sensitivity_meta_hrl.py` |

运行全部测试：

```bash
python -m pytest
```

修改信道、动作投影、观察维度或 Gymnasium 调用顺序时，应先运行对应的局部测试，再运行全量测试。

## 10. 当前实现边界

- Flat RL 使用静态单步 contextual bandit，不包含跨时隙信道演化；
- HRL 是两阶段训练：先 Worker，后冻结 Worker 训练 Manager，尚未实现端到端联合微调；
- Meta-PPO 的任务 context 本身是 7 维确定性 transition 统计，学习型分支仅学习后续特征表示；
- `grouping/metrics.py` 和 `utils/` 中部分模块仍为骨架；
- 默认物理模型假设理想 CSI，CSI 误差、移动性和更大规模分区处理属于后续扩展。
