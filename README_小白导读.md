# SimpleVLA-RL 项目导读（小白版）

> 本文件面向刚接触 VLA / 强化学习的同学，帮你快速理清 **SimpleVLA-RL** 的代码结构与运行流程，读完即可跑通代码。

---

## 一、这个项目是干什么的？（一句话版）

**SimpleVLA-RL** 是一个用 **强化学习（RL）** 训练 **视觉-语言-动作模型（VLA, Vision-Language-Action）** 的框架。

- 它让机器人通过"看图像 + 读语言指令"来学习如何做动作（抓取、放置等）。
- 相比传统的"监督微调（SFT）"，用 RL 训练可以在数据很少的情况下显著提升机器人在长程任务上的成功率。
- 项目基于 [veRL](https://github.com/volcengine/verl) 构建，并针对 VLA 做了大量适配（多环境并行渲染、动作 token 处理、0/1 奖励等）。

### 它解决什么问题？

机器人想学会"把杯子放到盘子上"这种任务，需要大量成功演示数据。RL 的好处是：
- 奖励设计极简：只看任务最后**成功 (1) 还是失败 (0)**，不需要复杂的奖励函数。
- 通过探索策略（动态采样、自适应裁剪、温度调参）让模型自己尝试不同动作。

### 支持的模型和环境

| 类别 | 支持内容 |
|------|----------|
| VLA 模型 | OpenVLA、OpenVLA-OFT |
| 仿真环境 | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)、[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) |
| 并行策略 | FSDP（数据并行）、vLLM（快速推理） |

---

## 二、核心概念快速扫盲

| 概念 | 大白话解释 |
|------|-----------|
| **VLA** | 视觉-语言-动作模型：输入是"机器人看到的图像 + 语言指令"，输出是"机械臂下一步要执行的动作" |
| **SFT** | 监督微调：用人类/专家演示数据教模型模仿动作（RL 训练前需要一个 SFT 模型作为起点） |
| **RL** | 强化学习：让模型自己在环境里试，成功给奖励 1，失败给 0，逐步学好 |
| **Rollout（采样/rollout）** | 模型在仿真环境里实际"跑一遍"，产生一条轨迹（图像→动作→奖励）的过程 |
| **Reward（奖励）** | 任务成功得 1，失败得 0（本项目用最简单的二元奖励） |
| **Actor** | 策略网络，即 VLA 模型本身，负责生成动作 |
| **Ref Policy** | 参考策略（就是初始 SFT 模型），用于计算 KL 散度防止策略跑偏 |
| **PPO / GRPO** | 强化学习算法。本项目用 **GRPO**（Group Relative Policy Optimization），它不需要 Critic 网络 |
| **轨迹 (trajectory)** | 一次完整任务执行过程：从开始到成功/失败的一连串（图像, 动作）对 |
| **action chunk** | 一次预测输出的连续多个动作步（如 8 步、25 步），避免每步都重新预测 |

---

## 三、整体代码结构

```
SimpleVLA-RL/
├── README.md                     # 官方原版 README（论文、结果、引用等）
├── SETUP.md                      # 环境安装详细指南（必读）
├── align.json                    # Ray 运行时环境变量配置（含 WANDB_API_KEY）
│
├── examples/                     # 【启动脚本】运行 RL 训练的入口
│   ├── run_openvla_oft_rl_libero.sh     # LIBERO 环境训练脚本
│   ├── run_openvla_oft_rl_twin2.sh      # RoboTwin2.0 环境训练脚本
│   ├── overwrite_vla_ckpt_utils.sh      # 把 VLA 模型代码覆盖到 checkpoint 目录
│   └── robotwin2_tasks_info.txt         # RoboTwin2.0 支持的任务列表
│
├── modified_codes/robotwin2/     # 对官方 RoboTwin2.0 的修改版代码
│   └── ...（任务定义、机器人控制、规划器等，用于覆盖官方代码）
│
├── verl/                         # 【核心代码】基于 veRL 改造的训练框架
│   ├── trainer/                  # 训练主流程
│   │   ├── main_ppo.py           # ★ 程序入口：初始化 Ray + 启动训练
│   │   └── ppo/
│   │       └── ray_trainer.py    # ★ RL 训练主循环（数据加载→采样→更新→评估→存盘）
│   │
│   ├── workers/                  # 各种"工人"模块
│   │   ├── fsdp_workers.py       # ★ Actor/Rollout/Ref 的核心实现
│   │   ├── actor/
│   │   │   └── dp_rob.py         # ★ 策略更新、loss 计算、log_prob/entropy
│   │   ├── rollout/
│   │   │   ├── rob_rollout.py    # ★ VLA 采样：建环境、多环境并行、生成动作、交互、收奖励
│   │   │   └── pre_collect_twin2_seed.py  # 预收集 RoboTwin 可行种子
│   │   └── critic/               # Critic 网络（GRPO 不用）
│   │
│   ├── utils/
│   │   ├── dataset/
│   │   │   └── rob_dataset.py    # ★ 训练/测试数据集构建
│   │   ├── vla_utils/
│   │   │   ├── openvla/          # OpenVLA 模型实现
│   │   │   └── openvla_oft/      # ★ OpenVLA-OFT 模型实现（含 constants.py）
│   │   ├── libero_utils.py       # LIBERO 环境工具函数
│   │   ├── openvla_utils.py      # OpenVLA 辅助工具
│   │   ├── fsdp_utils.py         # FSDP 分布式工具
│   │   └── envs/robotwin2/       # RoboTwin2.0 环境副本（由 copy 脚本生成）
│   │
│   ├── models/                   # 模型注册与加载
│   ├── single_controller/        # Ray 分布式控制器
│   └── protocol.py               # 数据协议定义
│
├── copy_overwrite_robotwin2.sh   # 把官方 RoboTwin2.0 复制并覆盖修改版代码
└── pre_collect_robotwin2_seed.sh # 预收集 RoboTwin2.0 可行种子（避免训练时反复验证）
```

---

## 四、关键模块详解

### 1. 启动入口：`verl/trainer/main_ppo.py`

整个程序的起点。主要做两件事：

```
main() 
  ├── 初始化 Ray（分布式计算框架）
  └── main_task.remote(config)   ← 在 Ray 上运行的主任务
        ├── 加载 tokenizer
        ├── 选择 Worker 类（FSDP 或 Megatron）
        ├── 定义各种"角色"：ActorRollout / Critic / RefPolicy
        ├── 创建 RobRewardManager（负责把 0/1 成功信号转成奖励）
        └── 创建 RayTrainer → trainer.fit()  ← 开始训练
```

- **`RobRewardManager`**：把 rollout 收集到的 `complete`（成功=1/失败=0）字段转成 token 级别的奖励张量。本项目用纯结果奖励，没有复杂的 reward shaping。

### 2. 训练主循环：`verl/trainer/ppo/ray_trainer.py`

这是 RL 训练的核心循环，每个 epoch 大致流程：

```
fit() 循环（每个 epoch）:
  1. 从 dataset 加载一批任务（task_id, trial_id, seed）
  2. 调用 actor_rollout 生成轨迹（在仿真环境里实际跑）
  3. 用 reward_fn 计算奖励（0/1）
  4. 计算优势函数（GRPO：组内相对优势）
  5. 更新 Actor 网络（PPO clip loss）
  6. 定期评估（val）+ 保存 checkpoint
```

- **`apply_kl_penalty`**：用 Ref Policy 计算 KL 散度，加到 reward 上防止策略偏离 SFT 太远。
- **GRPO 优势估计**：同一任务采样多条轨迹，用组内归一化的成功率作为优势，不需要单独的 Critic 网络。

### 3. Worker 核心：`verl/workers/fsdp_workers.py`

`RobActorRolloutRefWorker` 是一个"万能工人"，根据 `role` 参数可以扮演 Actor / Rollout / Ref 三种角色。核心方法：

| 方法 | 作用 |
|------|------|
| `init_model` | 加载 OpenVLA-OFT 模型，用 FSDP 包装 |
| `generate_sequences` | 用模型生成动作 token 序列 |
| `compute_log_prob` | 计算动作序列的对数概率（RL 更新要用） |
| `compute_entropy` | 计算策略熵（鼓励探索） |
| `update_actor` | 执行 PPO 策略梯度更新 |
| `save_checkpoint / load_checkpoint` | 存/读模型权重 |

### 4. 策略更新：`verl/workers/actor/dp_rob.py`

`RobDataParallelPPOActor` 实现具体的 RL loss 计算：

- **PPO clip objective**：`min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv)`
- **entropy bonus**：加一点熵奖励鼓励探索（`entropy_coeff`）
- **mask 处理**：只在有效动作 token 上计算 loss，padding 部分被 mask 掉

### 5. 采样（Rollout）：`verl/workers/rollout/rob_rollout.py`

这是 VLA 项目最特别的部分——模型要**真正在仿真环境里跑**，而不是像 LLM 那样只生成文本。流程：

```
generate_sequences():
  for 每个任务（并行）:
    ├── 1. 创建仿真环境（LIBERO / RoboTwin2.0）
    ├── 2. 重置环境，获取初始图像
    ├── 3. 循环直到任务结束或达到最大步数:
    │     ├── 把图像 + 语言指令喂给 VLA 模型
    │     ├── 模型输出一段 action chunk（如 8 步动作）
    │     ├── 把动作反归一化，发送给环境执行
    │     ├── 环境返回新图像、是否成功
    │     └── 把这一步的 (图像, 动作) 存入轨迹
    ├── 4. 任务结束：成功→reward=1，失败→reward=0
    └── 5. 保存轨迹视频（可选）+ 返回完整轨迹数据
```

- **多环境并行**：用 `ThreadPoolExecutor` 同时跑多个环境实例，大幅加速采样。
- **action token 处理**：VLA 模型输出的是离散 token，需要映射回连续动作空间。

### 6. 数据集：`verl/utils/dataset/rob_dataset.py`

构建训练/验证用的任务列表。每个样本包含：

- `task_suite_name`：任务集合名（如 `libero_10`、`robotwin2_lift_pot`）
- `task_id`：任务编号
- `trial_id` / `trial_seed`：试验编号/随机种子

**注意**：这里的 dataset 不是图像数据，而是"要跑哪些任务"的索引表。真正的图像在 rollout 时从环境实时渲染。

### 7. VLA 模型：`verl/utils/vla_utils/openvla_oft/`

OpenVLA-OFT 模型的实现（来自官方代码）。关键文件：

| 文件 | 作用 |
|------|------|
| `constants.py` | ★ 定义动作维度、chunk 长度、归一化方式等（LIBERO vs ALOHA 不同） |
| `configuration_prismatic.py` | 模型配置类 |
| `modeling_prismatic.py` | 模型前向传播 |
| `processing_prismatic.py` | 图像/文本预处理 |

`constants.py` 里的关键常量（通过环境变量 `ROBOT_PLATFORM` 切换）：

| 平台 | NUM_ACTIONS_CHUNK | ACTION_DIM | PROPRIO_DIM |
|------|-------------------|------------|-------------|
| LIBERO | 8 | 7 | 8 |
| ALOHA (RoboTwin) | 25 | 14 | 14 |

### 8. 配置文件：`verl/trainer/config/ppo_trainer.yaml`

Hydra 配置文件，定义所有默认超参数。运行时通过命令行参数覆盖（见启动脚本）。

---

## 五、数据流全景图

```
启动脚本 (.sh)
    │
    ▼
main_ppo.py  ──读取 align.json（环境变量）──→ Ray 初始化
    │
    ▼
ray_trainer.fit()  ← 主循环
    │
    ├── rob_dataset.py  → 任务列表（task_id, seed）
    │
    ├── rob_rollout.py  → 在仿真环境里跑，产出 (图像, 动作, 成功/失败)
    │       ↑
    │   openvla_oft 模型 生成动作 token
    │       ↑
    │   环境（LIBERO / RoboTwin2.0）返回观测与奖励
    │
    ├── RobRewardManager  → 把 0/1 成功信号转成 token 级奖励
    │
    ├── dp_rob.py  → 计算 PPO loss，更新 Actor 网络
    │
    └── 定期：评估 + 保存 checkpoint
```

---

## 六、快速上手（跑通代码）

### 第一步：安装环境

详细步骤见 **[SETUP.md](SETUP.md)**，这里给精简版：

```bash
# 1. 创建 conda 环境
conda create -n simplevla python==3.10
conda activate simplevla

# 2. 安装 PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 veRL（建议 v0.2.x）
git clone -b v0.2.x https://github.com/volcengine/verl.git
cd verl && pip3 install -e . && cd ..

# 4. 安装 OpenVLA-OFT
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft && pip install -e . && cd ..

# 5. 安装环境（二选一或都装）
# LIBERO:
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
# RoboTwin2.0: 见 SETUP.md Option 2
```

> ⚠️ **目录布局建议**：把 `SimpleVLA-RL`、`verl`、`openvla-oft`、`LIBERO`、`RoboTwin` 放在**同一层级**目录下，不要把它们互相嵌套。

### 第二步：准备 SFT 模型

RL 训练需要一个**已经监督微调过的 VLA 模型**作为起点。

- 下载官方提供的 SFT 模型：https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86
- 记下模型路径，后面要填到启动脚本里。

### 第三步：配置 WandB（训练可视化）

1. 注册 [WandB](https://wandb.ai/) 账号，获取 API Key。
2. 打开项目根目录的 `align.json`，把 `WANDB_API_KEY` 的 `"xxx"` 改成你自己的 key。

### 第四步：修改启动脚本

以 LIBERO 为例，编辑 `examples/run_openvla_oft_rl_libero.sh`，改这几个变量：

```bash
export WANDB_API_KEY='你的 wandb key'
SFT_MODEL_PATH="你的 SFT 模型路径"
CKPT_PATH="你想存 checkpoint 的路径"
DATASET_NAME="libero_10"          # 可选: libero_10, libero_90, libero_spatial, libero_object, libero_goal
NUM_GPUS=2                        # 你机器的 GPU 数量
NUM_NODES=1                       # 节点数（单机就写 1）
ALIGN_PATH=".../SimpleVLA-RL/align.json"   # 改成你机器上的绝对路径
```

> 💡 **小机器怎么跑？** 官方测试用 8 张 A800。如果 GPU 少/显存小，可以调小 `NUM_GPUS`、`data.train_batch_size`、`actor_rollout_ref.actor.ppo_mini_batch_size`，并开启 `param_offload=True` 节省显存。

### 第五步：启动训练

```bash
# LIBERO 环境
bash examples/run_openvla_oft_rl_libero.sh

# 或 RoboTwin2.0 环境
bash examples/run_openvla_oft_rl_twin2.sh
```

脚本会先调用 `overwrite_vla_ckpt_utils.sh` 把模型代码覆盖到 checkpoint 目录（确保模型定义和 checkpoint 匹配），然后启动 `python -m verl.trainer.main_ppo`。

### 第六步：只跑评估（验证模型）

在启动脚本里把 `trainer.val_only=False` 改成 `trainer.val_only=True`，然后运行同样的命令即可只做评估不训练。

---

## 七、RoboTwin2.0 专属说明

RoboTwin2.0 需要额外的配置步骤：

### 1. 复制并覆盖 RoboTwin 代码

```bash
bash copy_overwrite_robotwin2.sh <robotwin_path> <simplevlarl_path>
# 例: bash copy_overwrite_robotwin2.sh /mnt/robots/RoboTwin /mnt/robots/SimpleVLA-RL
```

这会把官方 RoboTwin 代码复制到 `verl/utils/envs/robotwin2/`，再用 `modified_codes/robotwin2/` 里的修改版覆盖。

### 2. 预收集可行种子

RoboTwin 有些 seed 下物体在机械臂够不到的位置，训练时会浪费时间。先跑一遍收集可行种子：

```bash
# 先改 pre_collect_robotwin2_seed.sh 里的 DATASET_NAME
sh pre_collect_robotwin2_seed.sh
```

生成的 `robotwin2_train_seeds.json` 放到 `verl/utils/envs/robotwin2/seeds/` 下（已有默认种子文件）。

### 3. 支持的任务

见 `examples/robotwin2_tasks_info.txt`，如 `lift_pot`、`place_empty_cup`、`stack_bowls_two` 等，每个任务有对应的 `traj_mini_batch_size`。

---

## 八、常见问题（FAQ）

**Q1: 报错 `can't import libero`？**
A: 没装 LIBERO 或没激活正确的 conda 环境。`conda activate simplevla` 后确认 `import libero` 不报错。

**Q2: 显存不够 OOM？**
A: ① 减小 `NUM_GPUS` 下的 batch size；② 开启 FSDP offload（脚本里 `param_offload=True`）；③ 减小 `actor_rollout_ref.rollout.tensor_model_parallel_size`。

**Q3: `align.json` 找不到？**
A: 启动脚本里 `ALIGN_PATH` 必须是绝对路径，且 `align.json` 里的 `WANDB_API_KEY` 要正确。

**Q4: 训练时 reward 一直是 0？**
A: 正常现象。RL 初期模型几乎不会成功，需要几十个 epoch 后才开始出现成功轨迹。可以先 `val_only=True` 看 SFT 模型的初始成功率。

**Q5: 怎么看训练曲线？**
A: 训练启动后会输出 WandB 链接，打开即可看 reward、success_rate、loss 等曲线。

**Q6: 想加新任务怎么办？**
A: 见 `SETUP.md` 最后一节"Supporting Additional Tasks in RoboTwin 2.0"，需要在 `rob_dataset.py` 注册任务名、在 `rob_rollout.py` 加 max_steps、在环境文件里实现 `get_info()`。

---

## 九、下一步学习建议

1. **先跑通**：按第六步跑通 LIBERO 的 val 模式，确认环境没问题。
2. **读主循环**：打开 `verl/trainer/ppo/ray_trainer.py`，跟着 `fit()` 函数走一遍数据流。
3. **读 rollout**：`verl/workers/rollout/rob_rollout.py` 是 VLA 最核心的部分，理解"图像→动作→环境→奖励"的循环。
4. **读模型**：`verl/utils/vla_utils/openvla_oft/` 了解 VLA 模型如何把图像和语言融合并输出动作。
5. **改参数做实验**：调 `temperature`、`clip_ratio`、`lr` 看对成功率的影响。

---

> 📖 更多信息请参考官方 [README.md](README.md)（论文、结果、引用）和 [SETUP.md](SETUP.md)（完整安装步骤）。
