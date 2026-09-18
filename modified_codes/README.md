# modified_codes/robotwin2 目录导读

> 本目录包含 SimpleVLA-RL 对 **RoboTwin 2.0** 仿真环境的**修改版代码**。在运行 RoboTwin2.0 的 RL 训练前，需要通过 `copy_overwrite_robotwin2.sh` 脚本把这些代码复制并覆盖到官方 RoboTwin 代码之上。

---

## 一、这个目录是干什么的？

RoboTwin 2.0 是一个双臂机器人操作仿真平台。SimpleVLA-RL 对其做了以下关键改造，以适配 RL 训练：

1. **运动规划器改造**：将原本依赖 NVIDIA curobo（需要 GPU）的规划器，替换为基于 **mplib** 的纯 CPU 规划器，使 RL rollout 能在 CPU 上高效运行。
2. **新增任务**：添加了 22 个操作任务（搬杯子、叠碗、敲钉子等），每个任务都定义了 `load_actors`（场景搭建）、`play_once`（演示动作）、`check_success`（成功判定）、`get_info`（语言指令模板参数）。
3. **指令生成**：为每个任务生成"seen / unseen"语言指令，用于 VLA 模型的语言条件训练。
4. **实体配置**：统一管理不同机器人型号（aloha-agilex、piper、franka-panda 等）的配置路径。

---

## 二、目录结构

```
modified_codes/robotwin2/
├── assets/objects/objaverse/list.json   # 杂乱桌面物体清单（用于随机背景干扰物）
│
├── description/utils/
│   └── generate_episode_instructions.py # 语言指令生成：把 {A}{B} 等占位符替换为物体名
│
├── envs/                                 # 【核心】仿真环境与任务定义
│   ├── _base_task.py                     # ★ 任务基类：场景搭建、机器人控制、抓取/放置等通用 API
│   ├── robot/                            # ★ 机器人与运动规划器
│   │   ├── robot.py                      # 双臂机器人类（mplib 规划器版，RL 训练时使用）
│   │   ├── robot_curobo.py               # 双臂机器人类（curobo 规划器版，预收集种子时使用）
│   │   ├── planner.py                    # 规划器（含 CuroboPlanner 接口 + MplibPlanner 实现）
│   │   └── planner_curobo.py             # 原始 curobo 规划器（GPU，预收集种子时使用）
│   ├── utils/                            # 工具函数
│   │   ├── create_actor.py               # 创建物体（盒/球/圆柱/OBJ/GLB/URDF）
│   │   ├── rand_create_cluttered_actor.py# 随机生成杂乱桌面物体
│   │   └── transforms.py                 # 坐标变换、旋转、抓取位姿计算等几何工具
│   └── <task_name>.py                    # 22 个具体任务（见下表）
│
├── script/
│   └── update_embodiment_config_path.py  # 把 ${ASSETS_PATH} 替换为实际路径
│
└── task_config/
    ├── _embodiment_config.yml            # 各机器人型号的配置路径
    ├── demo_clean.yml                    # 干净场景配置（无随机化）
    └── demo_randomized.yml               # 随机化场景配置（随机背景/光照/桌面杂物）
```

---

## 三、核心模块详解

### 1. 任务基类 `envs/_base_task.py`

所有任务的父类 `Base_Task`，提供了一套完整的"操作动词"API。每个具体任务只需实现 4 个方法。

**核心 API 一览**：

| API | 作用 |
|-----|------|
| `setup_scene()` | 创建 SAPIEN 场景、灯光、地面、物理材质 |
| `create_table_and_wall()` | 创建桌子和背景墙 |
| `load_robot()` | 加载双臂机器人并初始化规划器 |
| `load_camera()` | 加载头部/腕部相机 |
| `get_obs()` | 获取观测（RGB、深度、点云、关节角、末端位姿等） |
| `move(actions)` | 执行动作序列（移动+夹爪的组合） |
| `grasp_actor(actor, arm_tag, ...)` | 规划并执行抓取 |
| `place_actor(actor, arm_tag, ...)` | 规划并执行放置 |
| `move_by_displacement(arm_tag, x/y/z)` | 按位移移动机械臂 |
| `open_gripper() / close_gripper()` | 开合夹爪 |
| `back_to_origin(arm_tag)` | 机械臂回到初始位姿 |
| `choose_grasp_pose()` | 从物体的接触点中选择最佳抓取位姿 |
| `check_actors_contact(a, b)` | 检查两个物体是否接触 |
| `add_prohibit_area(actor)` | 标记禁止放置区域（避免物体重叠） |
| `check_stable()` | 仿真步进检查物体是否稳定 |

**任务子类必须实现的 4 个方法**：

| 方法 | 作用 | 何时调用 |
|------|------|----------|
| `setup_demo(**kwags)` | 调用 `_init_task_env_` 初始化 | 任务开始 |
| `load_actors()` | 在场景中放置任务物体 | 初始化时 |
| `play_once()` | 执行一次完整的任务演示动作（用于数据采集） | 采集数据时 |
| `check_success()` | 判断任务是否成功（返回 True/False） | RL rollout 结束时 |
| `get_info()` | 返回语言指令模板的参数（如 `{"{A}": "cup/base0"}`） | 生成语言指令时 |

> 💡 **RL 训练时只用到 `check_success()` 和 `get_info()`**，`play_once()` 是数据采集用的。RL 模型的动作由 VLA 模型直接输出关节角，不经过 `play_once`。

---

### 2. 机器人与规划器 `envs/robot/`

| 文件 | 作用 |
|------|------|
| `robot.py` | `Robot` 类：管理双臂（左/右）的关节、夹爪、末端位姿、运动规划。内部使用 **mplib 规划器**（CPU），适合 RL 训练时大量并行 rollout。 |
| `robot_curobo.py` | 同上，但内部使用 **curobo 规划器**（GPU），用于预收集种子阶段（`pre_collect_robotwin2_seed.sh`）。 |
| `planner.py` | 定义 `CuroboPlanner`（接口）和 `MplibPlanner`（实际实现）。`CuroboPlanner` 内部实际调用 `MplibPlanner`，对外保持 curobo 的接口不变。 |
| `planner_curobo.py` | 真正的 curobo 规划器实现（需要 NVIDIA GPU + curobo 库），预收集种子时使用。 |

**规划器工作流程**：

```
任务调用 robot.left_plan_path(target_pose)
  → CuroboPlanner.plan_path()
    → 提取 active joints 的当前关节角
    → MplibPlanner.plan_pose() 用 RRT 算法规划无碰撞路径
    → 返回关节轨迹 (position, velocity)
```

> 🔄 **切换机制**：`pre_collect_robotwin2_seed.sh` 会在运行前把 `planner_curobo.py` 重命名为 `planner.py`，跑完后再改回来。RL 训练时用 mplib 版。

---

### 3. 工具函数 `envs/utils/`

| 文件 | 作用 |
|------|------|
| `create_actor.py` | 创建各种形状物体：`create_box`、`create_sphere`、`create_cylinder`、`create_obj`（OBJ 模型）、`create_glb`（GLB 模型）、`create_urdf_obj`（URDF 机械/物体）、`create_table`（桌子） |
| `rand_create_cluttered_actor.py` | 在桌面上随机放置"干扰物"（杂乱背景），自动避开已有物体和禁止区域 |
| `transforms.py` | 几何变换工具：`rotate_along_axis`（绕轴旋转）、`get_place_pose`（计算放置位姿）、`cal_quat_dis`（四元数距离）、`Point` 类（可视化辅助点）等 |

---

### 4. 任务列表 `envs/<task_name>.py`

每个文件定义一个操作任务，类名与文件名相同。下表列出所有 22 个任务及简介：

| 任务名 | 简介 | 主要物体 |
|--------|------|----------|
| `beat_block_hammer` | 用锤子敲钉子 | 锤子、方块 |
| `blocks_ranking_rgb` | 把红/绿/蓝三色方块按顺序排列 | 三个彩色方块 |
| `click_bell` | 按响铃铛 | 铃铛 |
| `handover_block` | 左右手交接方块 | 方块 |
| `handover_mic` | 左右手交接麦克风 | 麦克风 |
| `lift_pot` | 双手举起锅 | 锅 |
| `move_can_pot` | 把罐子移到锅里 | 罐子、锅 |
| `move_pillbottle_pad` | 把药瓶移到垫子上 | 药瓶、垫子 |
| `move_stapler_pad` | 把订书机移到垫子上 | 订书机、垫子 |
| `pick_dual_bottles` | 双手各拿起一个瓶子 | 两个瓶子 |
| `place_a2b_left` | 把物体 A 放到 B 上（左手） | 两个物体 |
| `place_a2b_right` | 把物体 A 放到 B 上（右手） | 两个物体 |
| `place_container_plate` | 把容器放到盘子上 | 容器、盘子 |
| `place_empty_cup` | 把空杯子放到杯垫上 | 杯子、杯垫 |
| `place_mouse_pad` | 把鼠标放到鼠标垫上 | 鼠标、鼠标垫 |
| `place_phone_stand` | 把手机放到手机支架上 | 手机、支架 |
| `place_shoe` | 把鞋子放到指定位置 | 鞋子 |
| `put_bottles_dustbin` | 把瓶子扔进垃圾桶 | 瓶子、垃圾桶 |
| `shake_bottle` | 摇晃瓶子 | 瓶子 |
| `stack_blocks_two` | 叠两个方块 | 两个方块 |
| `stack_bowls_two` | 叠两个碗 | 两个碗 |

---

### 5. 语言指令生成 `description/utils/generate_episode_instructions.py`

VLA 模型需要"语言指令"作为输入。这个脚本负责把指令模板中的占位符替换为实际物体名。

**工作流程**：

```
任务模板 "put the {A} on the {B}"
  + episode 参数 {"{A}": "021_cup/base0", "{B}": "019_coaster/base0"}
  → 从 objects_description/021_cup/base0.json 随机选一个描述（如 "white cup"）
  → 生成指令: "put the white cup on the round coaster"
```

- `seen` 指令：训练时见过的描述。
- `unseen` 指令：测试时用的新描述，考验模型泛化能力。
- 机械臂占位符 `{a}` 会被替换为 "the left arm" / "the right arm"。

---

### 6. 配置文件 `task_config/`

| 文件 | 作用 |
|------|------|
| `_embodiment_config.yml` | 各机器人型号（aloha-agilex、piper、franka-panda、ARX-X5、ur5-wsg）的资源路径，由 `update_embodiment_config_path.py` 把 `${ASSETS_PATH}` 替换为实际路径。 |
| `demo_clean.yml` | 干净场景：无随机背景、无杂物、固定光照。用于调试。 |
| `demo_randomized.yml` | 随机化场景：随机背景纹理、桌面杂物、光照、桌面高度。用于训练数据增强。 |

---

## 四、快速上手

### 1. 复制代码到 RoboTwin

```bash
cd SimpleVLA-RL
bash copy_overwrite_robotwin2.sh <robotwin_path> <simplevlarl_path>
```

这会把 `modified_codes/robotwin2/` 下的文件复制到 `verl/utils/envs/robotwin2/` 并覆盖。

### 2. 更新资源路径

```bash
cd verl/utils/envs/robotwin2
python script/update_embodiment_config_path.py
```

### 3. 运行一个任务（交互式调试）

```python
from envs.lift_pot import lift_pot
import yaml

with open("task_config/demo_clean.yml") as f:
    config = yaml.safe_load(f)
config["task_name"] = "lift_pot"

task = lift_pot()
task.setup_demo(**config)
task.play_once()          # 执行一次演示
print(task.check_success())  # 检查是否成功
task.close_env()
```

### 4. 生成语言指令

```bash
python description/utils/generate_episode_instructions.py lift_pot demo_randomized 100
```

---

## 五、如何添加新任务

以添加一个 `my_new_task` 为例：

1. **创建任务文件** `envs/my_new_task.py`：
   ```python
   from ._base_task import Base_Task
   from .utils import *

   class my_new_task(Base_Task):
       def setup_demo(self, **kwags):
           super()._init_task_env_(**kwags)

       def load_actors(self):
           # 在场景中放置你的物体
           self.box = create_box(scene=self, pose=sapien.Pose([0, 0, 0.76]),
                                 half_size=(0.03, 0.03, 0.03), color=(1,0,0), name="box")

       def play_once(self):
           # 执行任务演示（数据采集用）
           arm_tag = ArmTag("right")
           self.move(self.grasp_actor(self.box, arm_tag=arm_tag, pre_grasp_dis=0.1))
           self.move(self.move_by_displacement(arm_tag, z=0.1))
           return self.info

       def get_info(self):
           return {"{A}": "red box", "{a}": "right"}

       def check_success(self):
           # 判断任务成功的条件
           return self.box.get_pose().p[2] > 0.85
   ```

2. **在 `verl/utils/dataset/rob_dataset.py` 中注册任务名**。
3. **在 `verl/workers/rollout/rob_rollout.py` 中添加任务的 max_steps**。
4. **（可选）创建语言指令模板** `description/task_instruction/my_new_task.json`。

---

## 六、关键概念

| 概念 | 解释 |
|------|------|
| **Actor** | 场景中的物体（盒子、杯子等），封装了位姿、接触点、功能点等信息 |
| **接触点 (contact point)** | 物体上预定义的可抓取位置，每个物体有多个接触点 |
| **功能点 (functional point)** | 物体上的关键位置（如杯底中心），用于放置对齐 |
| **ArmTag** | 机械臂标签，`"left"` 或 `"right"`，有 `.opposite` 属性 |
| **Action** | 动作对象，包含 `arm_tag`、`action`（"move" 或 "gripper"）、目标位姿等 |
| **prohibited_area** | 禁止放置区域，防止物体互相重叠 |
| **domain randomization** | 域随机化：随机背景、光照、桌面高度等，增强模型泛化 |

---

## 七、与 SimpleVLA-RL 主框架的关系

```
SimpleVLA-RL 主框架
  └── verl/workers/rollout/rob_rollout.py   ← 调用本目录的环境
        └── from envs.<task_name> import <task_name>  ← 创建任务实例
              ├── task.setup_demo()        ← 初始化场景
              ├── 循环：VLA 模型输出动作 → task 执行 → 获取观测
              └── task.check_success()     ← 得到 0/1 奖励
```

本目录的代码只负责**仿真环境**部分（场景、机器人、物体、成功判定），VLA 模型和 RL 算法在主框架中。
