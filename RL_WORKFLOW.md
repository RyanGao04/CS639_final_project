# RL Workflow

本地 RL 迭代现在统一走 `scripts/rl_workflow.py`。

## 1. 查看当前状态

```bash
python scripts/rl_workflow.py status
```

## 2. 录制轨迹

如果你从当前 shell 手动启动 Webots：

```bash
python scripts/rl_workflow.py record run_001
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
