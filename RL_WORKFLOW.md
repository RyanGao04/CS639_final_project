# RL Workflow

本地 RL 迭代现在统一走 `scripts/rl_workflow.py`。

## 1. 查看当前状态

```bash
python scripts/rl_workflow.py status
```

## 2. 录制轨迹

默认情况下，脚本会直接启动 `soccer_solo.wbt` 并注入 trace 环境变量：

```bash
python scripts/rl_workflow.py record run_001
```

如果你只想看将要启动的命令，不真正打开 Webots：

```bash
python scripts/rl_workflow.py record run_001 --dry-run
```

如果你想指定 world：

```bash
python scripts/rl_workflow.py record run_001 --world final_project/worlds/soccer_dual.wbt
```

如果你仍然希望手动从当前 shell 启动 Webots：

```bash
python scripts/rl_workflow.py record run_001 --no-launch-webots
```

它会打印需要导出的环境变量。然后在同一个 shell 里启动 Webots。

如果你希望脚本直接带着环境变量运行一个命令：

```bash
python scripts/rl_workflow.py record run_001 -- webots final_project/worlds/soccer_solo.wbt
```

录制出的 trace 默认放在 `tmp/rl_traces/`。

## 3. 训练并激活新策略

```bash
python scripts/rl_workflow.py train-activate --name rl_v1
```

默认会读取 `tmp/rl_traces/*.jsonl`，训练好的 checkpoint 放到 `tmp/rl_checkpoints/`，并把运行时权重写到：

```text
final_project/controllers/robot_one_controller/rl_policy_weights.npz
```

控制器本地运行时会自动加载这个文件。

## 3b. Deepbots 在线 SAC 训练

如果要让 policy action 直接实时控制 Webots 仿真并收集 transition，使用 deepbots harness。它会打开专用 world：

```text
final_project/worlds/soccer_solo_deepbots.wbt
```

先安装 SAC 训练依赖：

```bash
python scripts/rl_workflow.py install-rl-deps
```

用当前嵌入式 actor 或随机动作跑仿真并记录 transition：

```bash
python scripts/rl_workflow.py deepbots-record run_001 --policy actor
python scripts/rl_workflow.py deepbots-record explore_001 --policy random
```

启动 SAC 在线训练。SAC 的 action 会在每个 Webots step 直接写入左右轮，transition 会同步写入 `tmp/rl_traces/`：

```bash
python scripts/rl_workflow.py deepbots-train --name sac_v1 --timesteps 50000 --activate
```

默认训练时用 `--mode=fast --batch` 自动启动 Webots。想看 GUI 可以加：

```bash
python scripts/rl_workflow.py deepbots-train --name sac_debug --timesteps 5000 --no-batch --webots-mode realtime
```

如果已有 SB3 SAC `.zip` 模型，可以导出 deterministic actor 到运行时 `.npz`：

```bash
python scripts/rl_workflow.py activate-sb3 tmp/rl_checkpoints/sac_v1_sac.zip
```

注意：`deepbots` harness 是本地训练 scaffolding，不是最终提交依赖。最终提交仍只依赖 `starter_controller.py` 和导出的 `rl_policy_weights.npz`。

## 4. 只训练，不激活

```bash
python scripts/rl_workflow.py train --name rl_v2
```

## 5. 手动激活一个 checkpoint

```bash
python scripts/rl_workflow.py activate tmp/rl_checkpoints/rl_v2.pt
```

## 6. 回退到内置 bootstrap actor

```bash
python scripts/rl_workflow.py deactivate
```

这会删除本地 `rl_policy_weights.npz`，控制器就会重新使用 `starter_controller.py` 内嵌的权重。
