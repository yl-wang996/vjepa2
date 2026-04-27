# V-JEPA 机器人数据集微调方案

本文说明如果要在自己的机器人数据集上适配 V-JEPA / V-JEPA 2-AC，应该如何选择微调策略、准备数据、设计训练目标，并逐步验证效果。

## 总体建议

不要一开始就全量微调整个 V-JEPA encoder。

更推荐从轻到重逐步尝试：

```text
1. 冻结 V-JEPA encoder，只训练 action-conditioned predictor
2. 冻结大部分 encoder，只解冻最后几层 + 训练 predictor
3. 在 encoder 上加 LoRA / Adapter + 训练 predictor
4. 全量微调 encoder + predictor
```

对大多数自有机器人数据集，第一种或第二种通常更稳。

## 你要训练的不是普通 action head

V-JEPA 2-AC 的核心不是直接回归动作，而是训练一个 latent world model：

```text
当前图像 -> V-JEPA encoder -> z_t
下一帧图像 -> V-JEPA encoder -> z_{t+1}

输入: z_t, 当前 robot state s_t, 动作 a_t
目标: 预测 z_{t+1}
```

也就是训练：

```text
predictor(z_t, s_t, a_t) -> z_hat_{t+1}
loss = distance(z_hat_{t+1}, z_{t+1})
```

这样做的好处是：之后可以用 goal image 做规划。

```text
当前图像 + 目标图像
  -> 在 action 空间搜索
  -> 选择让 predicted latent 最接近 goal latent 的 action
```

## 推荐路线

### 方案 A：冻结 encoder，只训练 predictor

这是最推荐的起点。

训练内容：

```text
V-JEPA encoder: frozen
target encoder: frozen
AC predictor: trainable
```

优点：

- 显存需求最低。
- 不容易破坏 pretrained 表征。
- 对小中型机器人数据集更稳。
- 适合先验证数据格式、训练 loop、loss 是否正常下降。

缺点：

- 如果你的视觉域和 V-JEPA 预训练数据差异很大，encoder 可能不够适配。

适用场景：

- 单机单卡或少量 GPU。
- 数据量不大。
- 先做最小可行验证。
- 目标是训练一个 robot dynamics predictor，而不是重新训练视觉 backbone。

建议先跑这个。

### 方案 B：解冻 encoder 最后几层

如果方案 A 的 energy landscape 不理想，可以解冻 encoder 的最后 2-4 个 transformer blocks。

训练内容：

```text
encoder early blocks: frozen
encoder last blocks: trainable
AC predictor: trainable
```

优点：

- 能适配你的相机视角、机器人外观、场景材质。
- 比全量微调稳定。
- 显存和过拟合风险可控。

缺点：

- 训练成本比方案 A 高。
- 需要更小 learning rate，避免破坏表示。

适用场景：

- 你的数据和 V-JEPA 原始视觉域差异明显。
- predictor loss 能下降，但规划效果不好。
- ground-truth action 在 energy landscape 上没有明显低能量区域。

### 方案 C：LoRA / Adapter

在 encoder 的 attention 或 MLP 层加轻量可训练参数，同时训练 predictor。

训练内容：

```text
encoder 原始权重: frozen
LoRA / Adapter: trainable
AC predictor: trainable
```

优点：

- 比解冻整层更省显存。
- 不容易破坏 pretrained 权重。
- 适合中等规模数据集。

缺点：

- 当前 repo 没有现成的 V-JEPA 2-AC LoRA 训练脚本，需要额外接入。
- 需要决定在哪些层加 LoRA / Adapter。

适用场景：

- 想适配视觉域，但不想全量微调。
- 数据规模中等。
- 需要保存多个任务/场景的小适配器。

### 方案 D：全量微调

训练内容：

```text
encoder: trainable
AC predictor: trainable
```

优点：

- 适配能力最强。

缺点：

- 显存和算力要求最高。
- 小数据集非常容易过拟合。
- 容易破坏通用视觉表征。
- 调参成本高。

适用场景：

- 你有大规模机器人轨迹。
- 有多卡训练资源。
- 已经验证轻量方案不够。
- 明确要训练一个高度领域化的机器人 world model。

不建议作为第一步。

## 数据准备

当前 repo 的机器人训练代码在：

- [app/vjepa_droid/train.py](../app/vjepa_droid/train.py)
- [app/vjepa_droid/droid.py](../app/vjepa_droid/droid.py)
- [configs/train/vitg16/droid-256px-8f.yaml](../configs/train/vitg16/droid-256px-8f.yaml)

它期望 DROID-like 数据格式。CSV 文件每行是一个 trajectory 目录：

```text
/path/to/traj_000001
/path/to/traj_000002
```

每个 trajectory 目录中至少需要：

```text
traj_xxx/
  trajectory.h5
  metadata.json
  recordings/MP4/<camera_video>.mp4
```

代码会读取：

```text
metadata[<camera_view_key>]
trajectory.h5/observation/robot_state/cartesian_position
trajectory.h5/observation/robot_state/gripper_position
trajectory.h5/observation/camera_extrinsics/<camera_name>_left
```

其中 robot state 由下面两部分拼成 7D：

```text
cartesian_position: x, y, z, roll, pitch, yaw
gripper_position:  gripper
```

如果你的数据不是 DROID 格式，有两种选择：

1. 把数据转成 DROID-like layout。
2. 自己改 [app/vjepa_droid/droid.py](../app/vjepa_droid/droid.py)，让它读取你的数据格式。

建议优先写一个转换脚本，把自有数据转成 DROID-like 格式。这样可以少改训练代码。

## 训练配置怎么改

官方机器人训练配置是：

[configs/train/vitg16/droid-256px-8f.yaml](../configs/train/vitg16/droid-256px-8f.yaml)

这是大规模训练配置，不适合直接在单卡上跑：

```yaml
model_name: vit_giant_xformers
batch_size: 8
nodes: 4
tasks_per_node: 8
mem_per_gpu: 220G
```

单卡调试建议另建一个 local config，例如：

```text
configs/local/franka_vjepa_ac_debug.yaml
```

建议改动：

```yaml
folder: outputs/train/franka_vjepa_ac_debug
nodes: 1
tasks_per_node: 1

data:
  batch_size: 1
  datasets:
    - /path/to/your_train_paths.csv
  dataset_fpcs:
    - 8
  num_workers: 2
  crop_size: 256
  fps: 4
  camera_views:
    - left_mp4_path

optimization:
  epochs: 5
  ipe: 100
  lr: 0.0001
  start_lr: 0.00001
  warmup: 1
```

如果你用小模型，需要同步改：

```yaml
model:
  model_name: vit_base
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
```

## V-JEPA 2 还是 V-JEPA 2.1

repo 现成的 V-JEPA 2-AC robot post-training 配置是从 V-JEPA 2 ViT-g 出发：

```yaml
meta:
  pretrain_checkpoint: /your_vjepa2_checkpoints/vitg.pt
  context_encoder_key: target_encoder
  target_encoder_key: target_encoder
```

如果要从 V-JEPA 2.1 base 出发，需要注意：

- V-JEPA 2.1 base checkpoint 使用 `ema_encoder` key。
- V-JEPA 2.1 的模型代码在 `app/vjepa_2_1/` 下。
- `app/vjepa_droid` 当前默认使用 `src.models.vision_transformer`。
- 直接替换 checkpoint 路径不一定是完整适配。

可以先尝试“共享结构权重加载”的近似方案：

```yaml
meta:
  pretrain_checkpoint: checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
  context_encoder_key: ema_encoder
  target_encoder_key: ema_encoder
  load_predictor: false

model:
  model_name: vit_base
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
```

本 repo 已提供对应的 LIBERO 单卡 smoke test 配置：

```text
configs/train/vitb16/libero-256px-8f-debug.yaml
```

运行命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m app.main \
  --fname configs/train/vitb16/libero-256px-8f-debug.yaml \
  --devices cuda:0 \
  --debugmode True
```

当前已验证：该配置可在 24GB 单卡上完成 2 个训练 step，并保存 `outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt`。

但更严谨的方案是：

1. 让机器人训练代码使用 V-JEPA 2.1 encoder。
2. 从 `ema_encoder` 加载 V-JEPA 2.1 base 权重。
3. 接一个 action-conditioned predictor。
4. 先冻结 encoder，只训练 predictor。

## 推荐实验顺序

### 第 1 步：数据读取 smoke test

目标：确认你的 CSV、trajectory.h5、视频路径、robot state 都能被 dataset 正确读取。

建议：

- batch size 设为 1。
- num_workers 设为 0 或 1。
- 只跑一两个 batch。
- 打印 `clips/actions/states/extrinsics` shape。

期望 shape：

```text
clips:      [B, C, T, H, W]
actions:    [B, T-1, 7]
states:     [B, T, 7]
extrinsics: [B, T, 6]
```

### 第 2 步：冻结 encoder，只训练 predictor

目标：验证 action-conditioned prediction loss 能下降。

建议：

- 使用小 batch。
- 先跑 1-5 个 epoch。
- 观察 `loss / jloss / sloss`。
- 保存 `latest.pt`。

### 第 3 步：做离线验证

用你的验证轨迹生成类似 demo 的结果：

```text
当前帧
目标帧
ground-truth action
energy landscape
CEM planning action
```

判断标准：

- ground-truth action 附近是否有较低 energy。
- CEM 规划动作是否方向合理。
- 换不同目标帧时，规划动作是否随目标变化。

### 第 4 步：再考虑微调 encoder

如果 predictor loss 下降但规划效果差，再考虑：

```text
解冻 encoder 最后几层
或
加入 LoRA / Adapter
```

不要直接跳到全量微调。

## 评估指标

建议至少看这些：

- prediction loss：`predicted latent` 与 `target latent` 的距离。
- energy landscape：ground-truth action 是否靠近低 energy 区域。
- action retrieval：在一批候选动作中，ground-truth action 的 energy 排名是否靠前。
- CEM action error：CEM 输出动作与 ground-truth action 的 L2 / cosine / direction error。
- offline rollout consistency：多步预测 latent 是否越来越偏。
- real/sim success rate：如果后续接真实机器人或仿真，再看任务成功率。

## 推荐默认方案

如果你的数据量不是特别大，建议从这里开始：

```text
V-JEPA encoder: frozen
AC predictor: trainable
training objective: latent next-state prediction
validation: energy landscape + CEM planning on held-out trajectories
```

等这个基线跑通后，再尝试：

```text
解冻 encoder 最后几层
或
LoRA / Adapter
```

最后才考虑全量微调整个模型。

## 最小 Checklist

1. 准备 DROID-like trajectory 数据。
2. 如果使用 LIBERO，也可以直接准备 HDF5 path CSV 并设置 `dataset_format: libero`。
3. 跑通 dataset smoke test。
4. 用小模型/小 batch 跑通训练 loop。
5. 先冻结 encoder，只训练 AC predictor。
6. 在验证轨迹上画 energy landscape。
7. 如果效果不够，再做轻量 encoder adaptation。
