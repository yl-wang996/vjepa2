# LIBERO 模型评估

本文档说明如何更系统地评估当前训练好的 `V-JEPA 2.1 ViT-B encoder + AC predictor` 模型。

## 评估分层

建议按下面顺序看：

1. 离线单样本可视化  
   看 clip 和 energy landscape，确认模型行为是否大致合理。
2. 批量离线统计  
   在一批固定 clip 上统计 latent prediction、action retrieval、CEM 规划误差。
3. 仿真闭环 success rate  
   等安装好 LIBERO 仿真依赖后，再看任务成功率。

当前 repo 已补齐前两层。

## 1. 单样本验证

命令：

```bash
.venv/bin/python scripts/libero_vitb_ac_validate.py \
  --config configs/train/vitb16/libero-256px-8f-main.yaml \
  --checkpoint outputs/train/libero-vjepa21-vitb-ac-main/latest.pt \
  --output-dir outputs/validation/libero-vjepa21-vitb-ac-main
```

输出：

- `libero_clip_frames.png`
- `energy_landscape_dx_dz.png`
- `metrics.json`

用途：

- 快速看输入 clip 是否正确。
- 看 ground-truth action 附近是不是低 energy 区域。
- 看最低 energy 动作是否离 GT 很远。

## 2. 批量离线评估

命令：

```bash
.venv/bin/python scripts/libero_vitb_ac_eval_batch.py \
  --config configs/train/vitb16/libero-256px-8f-main.yaml \
  --checkpoint outputs/train/libero-vjepa21-vitb-ac-main/latest.pt \
  --output-dir outputs/eval/libero-vjepa21-vitb-ac-main \
  --sample-mode tail \
  --max-samples 32 \
  --goal-steps 1,2,4,7
```

输出：

- `per_sample_metrics.csv`
- `summary.json`
- `examples/*.png`

### `per_sample_metrics.csv`

逐条样本记录，主要字段：

- `one_step_loss@k`  
  第 `k-1 -> k` 步的单步 latent 预测误差。
- `rollout_loss@k`  
  从第 0 帧开始自回归 rollout 到第 `k` 帧的 latent 误差。
- `retrieval_gt_rank`  
  在 `1 + N` 个候选动作里，GT action 的 energy 排名。
- `retrieval_top1/top5`  
  GT action 是否排第 1 / 前 5。
- `retrieval_energy_margin`  
  随机负样本平均 energy 减去 GT energy，越大越好。
- `cem_first_action_xyz_l2`
- `cem_first_action_xyz_cosine`
- `cem_rollout_l2`  
  CEM 规划动作与离线 GT action 的误差。

### `summary.json`

聚合后的均值、方差、最小值、最大值。

更适合横向比较不同 checkpoint 或不同训练配置。

## 如何解读指标

### 1. `one_step_loss@1`

最基础的局部动态预测能力。  
如果它都不稳定，后面的 planning 一般不会好。

### 2. `rollout_loss@2/4/7`

反映多步 rollout 漂不漂。  
如果 `one_step` 还行，但 `rollout_loss@4`、`@7` 很快变差，通常说明 world model 误差累积严重。

### 3. `retrieval_gt_rank`

反映模型会不会给真实动作更低 energy。  

理想情况：

- rank 越接近 1 越好
- `retrieval_top1` 越高越好
- `retrieval_energy_margin` 为正且越大越好

### 4. `cem_first_action_xyz_l2`

反映用 world model 做动作搜索时，第一步规划动作离 GT 有多远。  

如果 retrieval 还行，但 CEM 误差很大，通常说明：

- energy landscape 不够尖锐
- 或 CEM 参数不合适

## 推荐默认评估配置

先这样跑：

```bash
.venv/bin/python scripts/libero_vitb_ac_eval_batch.py \
  --config configs/train/vitb16/libero-256px-8f-main.yaml \
  --checkpoint outputs/train/libero-vjepa21-vitb-ac-main/latest.pt \
  --output-dir outputs/eval/libero-vjepa21-vitb-ac-main \
  --sample-mode tail \
  --max-samples 32 \
  --goal-steps 1,2,4,7 \
  --num-negatives 32 \
  --cem-goal-step 2 \
  --cem-rollout 2
```

## 仿真闭环评估

这是下一层，也是最重要的一层，但当前环境里还没有安装：

- `libero`
- `gymnasium`
- `robosuite`

建议后续单独准备仿真环境，再做：

1. reset task
2. 用当前 observation 和目标 observation 做 CEM 规划
3. 执行动作
4. 统计 success rate

在那之前，`batch offline eval` 就是当前最稳妥、最可复现的主评估方式。
