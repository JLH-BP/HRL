<!-- Serves as the version-controlled source for the META-HRL project task framework. -->
# META-HRL 项目任务框架

**项目阶段：** Phase 4 Flat RL PPO 基线已完成  
**系统：** 128 阵元 ULA 下行大规模 MIMO，6 个单天线用户，混合近场/远场莱斯信道。  
**当前范围：** 已实现并测试信道、单公共流一层 RSMA、候选分组/启发式分组和单步 Gymnasium 资源分配环境；完整 HRL、Meta-RL 与训练流程尚未实现。

## 1. 研究目标

利用元分层强化学习（Meta-HRL）为混合近远场 RSMA 系统联合决策：

1. 六用户分组；
2. 用户私有流功率分配；
3. 公共流或组公共流功率分配；
4. 公共组内用户的公共流速率分配；
5. 后续可扩展的预编码、波束码本选择、SIC 顺序、QoS 和能效控制。

研究核心问题是：近场球面波同时依赖距离与角度，远场平面波主要依赖角度；固定分组与固定资源策略难以适应用户几何、莱斯因子、SNR 和 QoS 的变化。HRL 将慢变化的分组与快变化的资源分配分离，Meta-RL 用任务上下文提升未见场景的少样本适应能力。

## 2. 系统与信道模型

- 基站天线数：`M = 128`；阵列：ULA；建议阵元间距 `d = λ/2`。
- 用户数：`K = 6`；用户在二维平面随机生成，后续可扩展三维与移动性。
- 近远场边界使用 Rayleigh 距离：`r_R = 2D²/λ`，其中 `D=(M-1)d`。
- 远场 LoS：归一化平面波导向矢量 `a_FF(θ)`。
- 近场 LoS：按每阵元到用户距离计算的归一化球面波导向矢量 `a_NF(r, θ)`；初版仅要求相位曲率，幅度变化可作为扩展。
- 莱斯信道：`h_k = sqrt(β_k)[sqrt(κ_k/(1+κ_k)) a_k^LoS + sqrt(1/(1+κ_k)) g_k]`，其中 `g_k` 为复高斯 NLoS 项。
- 大尺度衰落需支持路径损耗、阴影衰落与距离相关 K 因子；初版建议完美 CSI，后续增加估计误差与延迟 CSI。

在训练前必须验证：导向矢量归一化、近场距离-角度聚焦、远场角度选择性、近场大距离极限、同角不同距离用户可分性与莱斯统计特征。

## 3. RSMA 与资源分配

MVP 采用单公共流的一层 RSMA：

`x = p_c s_c + Σ_k p_k s_k`。

用户先解码公共流，再经 SIC 解码私有流；公共流率由目标用户中的最低可解码速率约束。公共率分配 `C_k` 需满足 `C_k ≥ 0` 且 `Σ_k C_k ≤ R_c`，用户总速率为 `R_k = C_k + R_p,k`。

第一阶段固定 MRT 或 RZF 预编码，让 RL 仅优化公共/私有功率和公共率分配。第二阶段扩展为组公共流：高层将用户划分为多个组，每组一个公共流。所有动作必须满足总功率、非负功率、公共率预算、唯一分组和组规模约束。

## 4. 分层强化学习设计

### 高层 Manager

高层每 `T_H` 个信道块更新，基于距离、角度、近远场标签、路径损耗、K 因子、历史速率、QoS 和用户相关性选择：

- 候选用户分组；
- RSMA 公共流模式；
- 各组资源预算；
- 用户优先级或低层候选波束集合。

第一版建议使用有限候选分组集合，而不是直接输出所有集合分区，以避免组合动作爆炸与组标签置换。

### 低层 Worker

低层每个或每几个信道块更新，输入瞬时 CSI 的结构化特征、高层目标、SINR、有效增益与 QoS 缺口，输出公共/私有流功率比例及公共率分配。softmax 或 simplex 投影确保功率预算和速率预算可行。

### 奖励与约束

初版奖励采用加权和速率，加上 QoS outage、公平性和分组切换惩罚。Jain 公平性、最小速率、outage、能效与组切换应分别记录。硬约束由环境投影实现，软 QoS 约束可采用惩罚或拉格朗日约束强化学习。

## 5. Meta-RL 设计

一个 meta-task 表示一类无线环境分布：近远场比例、用户位置和角度簇、莱斯 K 因子、SNR、QoS、路径损耗、移动性与 CSI 误差。Meta-training、validation 和 OOD meta-testing 必须使用分离的随机种子与分布。

推荐采用上下文编码方案：从 `(state, action, reward, next_state)` transition 中推断任务潜变量，并作为高层和低层策略条件。必须对比域随机化、Flat RL、无 Meta 的 HRL、无层次 Meta-RL 及完整 Meta-HRL，并评估新任务少样本适应曲线。

## 6. 实验协议、基线与指标

核心基线包括 Equal Power + MRT/RZF、SDMA、NOMA、无分组 RSMA、启发式分组 RSMA、固定分组、经典优化参考、Flat PPO/SAC、HRL without Meta-RL 与完整 Meta-HRL。

核心指标包括和速率、平均/最小/5% 分位用户速率、Jain 公平性、QoS 满足率、outage、能效、公共/私有功率比例、公共率、SIC 复杂度、分组切换、训练稳定性、推理延迟和 OOD 适应速度。所有结果需多随机种子平均和置信区间。

必要消融包括：移除 Meta-RL、移除 HRL、全部远场建模、移除公共流、固定/随机/启发式分组、仅优化功率、完美/不完美 CSI、静态/移动用户以及不含 QoS 或切换惩罚的奖励。

## 7. 实施阶段

1. **Phase 0：假设冻结**——载波频率、坐标系、路径损耗、K 因子、RSMA 架构、CSI、目标函数和任务分布。
2. **Phase 1：信道模块（已完成）**——ULA 几何、近远场导向矢量、莱斯信道、数值检查与诊断。
3. **Phase 2：RSMA 与分组模块（已完成）**——信号模型、SINR、速率、MRT/RZF、约束、传统基线、候选分区和启发式分组。
4. **Phase 3：单步环境 MVP（已完成）**——场景采样、CSI+几何状态、13 维 logits 动作、功率/公共率约束投影、速率/公平性/QoS 奖励和 Gymnasium API。
5. **Phase 4：Flat RL（已完成 MVP）**——基于 Stable-Baselines3 PPO 的连续动作 MLP 策略、模型保存、独立种子评估，以及等功率 SDMA/固定公共 RSMA 对比。
6. **Phase 5：HRL（两阶段 PPO MVP 已完成）**——组公共流物理层、203 候选分区、18 维 Worker 资源动作、多步高层时钟与分组切换代价均已实现。可先在固定/启发式 Manager 下训练 PPO Worker，再冻结 Worker 训练离散 Manager PPO；联合微调与更强 HRL 算法仍待实现。
7. **Phase 6：Meta-RL（上下文基础层已完成）**——已实现可复现 train/validation/OOD 元任务划分、按 task ID 隔离的 transition context buffer、7 维确定性 context encoder 与 context-conditioned Worker 环境；学习型上下文后验、少样本元训练和 OOD 适应曲线仍待实现。
8. **Phase 7：完整实验**——参数扫描、消融、复杂度统计和论文级图表。

## 8. 推荐项目目录

```text
META_HRL/
├── configs/
│   ├── channel/rician_near_far.yaml
│   ├── rsma/default.yaml
│   ├── environment/default.yaml
│   ├── training/default.yaml
│   └── experiments/meta_hrl_mixed_near_far.yaml
├── docs/meta_hrl_task_framework.{md,docx}
├── scripts/generate_task_framework_docx.py
├── outputs/
├── src/meta_hrl/
│   ├── channel/{geometry,near_field,far_field,rician,validation}.py
│   ├── rsma/{signal_model,precoding,rate,constraints,baselines}.py
│   ├── grouping/{heuristic,candidate_groups,metrics}.py
│   ├── envs/{rsma_env,hierarchical_env,task_sampler,observation_encoder}.py
│   ├── agents/{high_level_policy,low_level_policy,meta_context,replay_buffer}.py
│   ├── training/{train_flat_rl,train_hrl,train_meta_hrl,evaluate}.py
│   └── utils/{seed,logging,normalization,checkpoint}.py
└── tests/
```

每个模块遵循职责清晰、可组合的接口边界。环境实现顺序必须遵循“信道正确性优先于 RSMA，RSMA 正确性优先于环境，环境正确性优先于 RL 训练”。当前单步环境固定分组和单公共流，用于 Flat RL 资源分配基线；双时间尺度分层环境将在 HRL 阶段基于该 API 扩展。

### 3.1 当前单步环境接口

`meta_hrl.envs.OneStepRSMAEnv` 遵循 Gymnasium API。`reset(seed=...)` 采样一个静态混合近远场莱斯场景，返回长度为 `1572` 的 `float32` 观测；每位用户依次编码归一化 CSI 实部/虚部、`log1p` 距离、归一化角度、近场标记、路径增益、K 因子及 QoS 目标。`step()` 接收 13 维实数 logits：前 7 维通过 softmax 在 6 个私有流和 1 个公共流间分配总功率，环境据此计算物理公共率瓶颈；后 6 维再通过 softmax 分配该公共率。每回合仅一步，返回 `terminated=True`、`truncated=False`。奖励为加权和速率与 Jain 公平性减去 QoS 缺口，详细诊断通过 `info` 返回。

## 9. 验证

生成 Word 文档：

```bash
python -m pip install -e ".[docs]"
python scripts/generate_task_framework_docx.py
```

运行项目：

```bash
python -m pip install -e .
meta-hrl
```

`pytest` 是可选的开发验证依赖，不是项目运行前提。
