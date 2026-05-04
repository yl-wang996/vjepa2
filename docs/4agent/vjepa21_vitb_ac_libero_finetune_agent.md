# Agent 对齐文档：V-JEPA 2.1 ViT-B + LIBERO + AC Predictor 微调

本文给执行 agent 使用。目标是在当前 repo 中跑通这条链路：

```text
加载 V-JEPA 2.1 ViT-B encoder
直接读取 LIBERO HDF5 数据
训练 action-conditioned predictor
保存训练 checkpoint
```

当前已在 24GB 单卡上验证通过。

## 结论先读

优先使用这个配置：

```text
configs/train/vitb16/libero-256px-8f-debug.yaml
```

运行命令：

```bash
cd /home/yunlong/workspace/vjepa2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m app.main \
  --fname configs/train/vitb16/libero-256px-8f-debug.yaml \
  --devices cuda:0 \
  --debugmode True
```

期望结果：

```text
dataset length: 500
model: vit_base encoder + AC predictor
checkpoint loaded from epoch 40
max cuda memory allocated: about 1.85 GB
latest.pt saved
```

已验证产物：

```text
outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt
outputs/train/libero-vjepa21-vitb-ac-debug/e0.pt
outputs/train/libero-vjepa21-vitb-ac-debug/log_r0.csv
outputs/train/libero-vjepa21-vitb-ac-debug/params-pretrain.yaml
```

## Agent 执行 Checklist

按顺序执行：

```text
1. 确认 cwd 是 /home/yunlong/workspace/vjepa2
2. 确认 .venv/bin/python 可用
3. 确认 checkpoints/vjepa2_1_vitb_dist_vitG_384.pt 存在且大小约 1.66GB
4. 确认 data/libero/libero_spatial_h5_paths.csv 存在
5. py_compile 检查 app/vjepa_droid 相关文件
6. 清理 outputs/train/libero-vjepa21-vitb-ac-debug
7. 运行 configs/train/vitb16/libero-256px-8f-debug.yaml
8. 检查 latest.pt、log_r0.csv、loss 和显存日志
9. 运行 scripts/libero_vitb_ac_validate.py
10. 检查 validation 下的 PNG 和 metrics.json
```

## 路线解释

这条线不是直接预测机器人动作，也不是全量微调 V-JEPA encoder。

训练目标是 latent dynamics prediction：

```text
RGB clip -> frozen target_encoder -> target latent h
past latent h + robot state s + action a -> AC predictor -> predicted latent z
loss = distance(predicted latent z, future target latent h)
```

也就是说，AC predictor 学的是：

```text
给定当前视觉 latent、机器人状态、动作，预测下一步视觉 latent
```

当前 `app.vjepa_droid.train` 的训练 step 中：

- `target_encoder` 在 `torch.no_grad()` 下产生训练目标 latent；
- `predictor` 参与 forward/backward；
- `encoder` 会初始化并加载权重，主要用于 checkpoint 结构对齐，当前 loss 路径没有用它做梯度训练。

## 相关文件

核心训练入口：

```text
app/main.py
app/vjepa_droid/train.py
app/vjepa_droid/droid.py
app/vjepa_droid/libero.py
app/vjepa_droid/utils.py
```

核心配置：

```text
configs/train/vitb16/libero-256px-8f-debug.yaml
```

相关文档：

```text
docs/LIBERO_test_dataset.md
docs/V-JEPA_robot_finetuning_guide.md
docs/contracts/dataset_format_contract.md
```

不要用下面这个配置做 24GB 单卡 smoke test：

```text
configs/train/vitg16/libero-256px-8f-debug.yaml
```

它保留 ViT-g / V-JEPA 2-AC 大模型设置，之前已验证会在 Adam optimizer step 附近 OOM。

## 前置条件

工作目录：

```text
/home/yunlong/workspace/vjepa2
```

Python 环境：

```text
.venv/
pyproject.toml
```

如果环境还没装好，先执行：

```bash
uv sync
```

如果 `uv sync` 没覆盖系统 CUDA/PyTorch 环境，沿用已有 `.venv`。当前本地 `.venv/bin/python` 已能运行训练。

## Checkpoint 要求

需要这个 checkpoint：

```text
checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

本地已验证大小：

```text
1664223428 bytes
```

检查命令：

```bash
stat -c '%n %s bytes' checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
```

该 checkpoint 的关键字段：

```text
encoder
predictor
ema_encoder
epoch
```

本路线加载：

```yaml
meta:
  pretrain_checkpoint: checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
  context_encoder_key: ema_encoder
  target_encoder_key: ema_encoder
  load_predictor: false
```

说明：

- `ema_encoder` 用作 V-JEPA 2.1 ViT-B encoder 权重来源。
- `load_predictor: false` 表示 AC predictor 随机初始化并从 LIBERO 数据开始训练。
- loader 会跳过 V-JEPA 2.1 里当前 robot encoder 不需要或 shape 不匹配的 key。

预期日志会出现：

```text
Skipped 12 incompatible pretrained keys
loaded pretrained encoder from epoch 40
loaded pretrained target encoder from epoch 40
```

这是正常现象。

## LIBERO 数据要求

需要这个 CSV：

```text
data/libero/libero_spatial_h5_paths.csv
```

当前本地数据：

```text
data/libero/libero_spatial/
10 HDF5 files
500 demos
about 5.9 GB
```

CSV 支持两种格式：

```text
/path/to/file.hdf5
/path/to/file.hdf5 demo_0
```

如果只写 HDF5 文件路径，loader 会枚举该文件下所有 demo。

检查命令：

```bash
head data/libero/libero_spatial_h5_paths.csv
find data/libero/libero_spatial -name '*.hdf5' | wc -l
```

## LIBERO Loader 映射

实现文件：

```text
app/vjepa_droid/libero.py
```

配置开关：

```yaml
data:
  dataset_format: libero
  datasets:
  - data/libero/libero_spatial_h5_paths.csv
  libero_camera_key: agentview_rgb
  libero_frame_stride: 1
```

字段映射：

```text
obs/agentview_rgb                  -> RGB frames
obs/ee_states[:, :6]               -> cartesian pose: x y z roll pitch yaw
obs/gripper_states 或 actions[:, -1] -> gripper state
dummy zeros [T, 6]                 -> camera extrinsics
```

读取时会额外做一次：

```text
vertical flip (flipud)
```

原因是当前 LIBERO 原始 RGB 帧方向与我们期望的训练/可视化方向相反。

返回给训练 loop 的 tuple：

```text
clips:      [B, C, T, H, W]
actions:    [B, T-1, 7]
states:     [B, T, 7]
extrinsics: [B, T, 6]
indices:    [B, T]
```

当前 debug 配置中：

```text
B = 1
T = 8
H = W = 256
```

因此单条 sample 预期为：

```text
clip:       [3, 8, 256, 256]
actions:    [7, 7]
states:     [8, 7]
extrinsics: [8, 6]
```

## 训练配置关键点

配置文件：

```text
configs/train/vitb16/libero-256px-8f-debug.yaml
```

模型：

```yaml
model:
  model_name: vit_base
  pred_depth: 12
  pred_embed_dim: 384
  pred_num_heads: 12
  use_rope: true
  use_extrinsics: false
```

数据：

```yaml
data:
  dataset_format: libero
  batch_size: 1
  crop_size: 256
  dataset_fpcs:
  - 8
  tubelet_size: 2
  num_workers: 0
```

优化：

```yaml
optimization:
  epochs: 1
  ipe: 2
  lr: 0.0001
  start_lr: 0.00001
  warmup: 0
  enc_lr_scale: 0.1
```

说明：

- `ipe: 2` 表示只跑 2 个 iteration，用于 smoke test。
- `warmup: 0` 是为了避免唯一几步学习率直接变成 0。
- `dtype: bfloat16` 降低显存占用。
- `use_extrinsics: false` 是因为 LIBERO loader 目前返回 dummy extrinsics。
- `enc_lr_scale` 保留在配置中；当前训练 loss 路径没有使用 `encoder` 做 forward，因此实际梯度主要来自 AC predictor。

## 运行前检查

建议执行：

```bash
cd /home/yunlong/workspace/vjepa2

.venv/bin/python -m py_compile \
  app/vjepa_droid/libero.py \
  app/vjepa_droid/droid.py \
  app/vjepa_droid/train.py \
  app/vjepa_droid/utils.py

stat -c '%n %s bytes' checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
head data/libero/libero_spatial_h5_paths.csv
nvidia-smi
```

如果要重跑干净实验：

```bash
rm -rf outputs/train/libero-vjepa21-vitb-ac-debug
```

## 运行训练

执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m app.main \
  --fname configs/train/vitb16/libero-256px-8f-debug.yaml \
  --devices cuda:0 \
  --debugmode True
```

正常日志应包含：

```text
Running pre-training of app: vjepa_droid
which_dtype='bfloat16'
Initialized (rank/world-size) 0/1
Encoder number of parameters: 86236416
Predictor number of parameters: 21894144
VideoDataset unsupervised data loader created
iterations per epoch/dataset length: 2/500
Loading pretrained model from checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
loaded pretrained encoder from epoch 40
loaded pretrained target encoder from epoch 40
Epoch 1
loss: ...
avg. loss ...
```

已验证的一次运行结果：

```text
[1, 0] loss: 1.871 [0.93, 0.94] [lr: 5.00e-05] [mem: 1.85e+03]
[1, 1] loss: 1.796 [0.90, 0.90] [lr: 0.00e+00] [mem: 1.85e+03]
avg. loss 1.796
```

`mem: 1.85e+03` 单位是 MiB。

## 运行后检查

检查文件：

```bash
find outputs/train/libero-vjepa21-vitb-ac-debug -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
cat outputs/train/libero-vjepa21-vitb-ac-debug/log_r0.csv
```

预期：

```text
e0.pt
latest.pt
log_r0.csv
params-pretrain.yaml
```

检查 checkpoint 内容：

```bash
.venv/bin/python - <<'PY'
import torch
p = "outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt"
ckpt = torch.load(p, map_location="cpu")
print(list(ckpt.keys()))
print("epoch", ckpt.get("epoch"), "loss", ckpt.get("loss"), "batch_size", ckpt.get("batch_size"))
print("encoder keys", len(ckpt["encoder"]), "predictor keys", len(ckpt["predictor"]))
PY
```

已验证输出：

```text
epoch 1
loss 1.7961266040802002
batch_size 1
encoder keys 148
predictor keys 156
```

## 离线验证和 Energy Landscape

训练产物确认后，运行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python scripts/libero_vitb_ac_validate.py \
  --config configs/train/vitb16/libero-256px-8f-debug.yaml \
  --checkpoint outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt \
  --output-dir outputs/validation/libero-vjepa21-vitb-ac-debug \
  --device cuda:0 \
  --sample-index 0 \
  --seed 0 \
  --energy-nsamples 21 \
  --energy-grid-size 0.05
```

这个脚本会：

```text
1. 加载训练好的 latest.pt
2. 从 LIBERO HDF5 中取一段 clip
3. 用 target encoder 计算 frame latents
4. 用 AC predictor 计算 ground-truth action 的 one-step latent prediction energy
5. 在 action 的 dx/dz 平面生成 energy landscape
```

预期日志：

```text
Loaded checkpoint: path=outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt; epoch=1; loss=1.796...
Validation sample: ...demo.hdf5:demo_0 indices=[...]
Latents: h=(1, 2048, 768) tokens_per_frame=256
Energy: gt=...; center_grid=...; best_grid=...; better_than_center=.../441
Saved: outputs/validation/libero-vjepa21-vitb-ac-debug/libero_clip_frames.png ...
```

产物：

```text
outputs/validation/libero-vjepa21-vitb-ac-debug/libero_clip_frames.png
outputs/validation/libero-vjepa21-vitb-ac-debug/energy_landscape_dx_dz.png
outputs/validation/libero-vjepa21-vitb-ac-debug/metrics.json
```

已验证的一次结果：

```text
gt_energy: 0.8542167544364929
center_grid_energy: 0.8541468381881714
best_grid_energy: 0.8541215658187866
grid_actions_better_than_center: 199 / 441
```

注意：当前 checkpoint 只训练了 2 个 step，这里的 energy landscape 只能说明验证链路跑通，不代表模型已经学到可靠策略。正式判断效果时，应换成长训练 checkpoint，并在 held-out demos 上统计。

## 常见问题

### 1. ViT-g 配置 OOM

现象：

```text
CUDA out of memory
OOM in Adam optimizer state allocation
```

处理：

```text
不要在 24GB 单卡上使用 configs/train/vitg16/libero-256px-8f-debug.yaml 做训练 smoke test。
改用 configs/train/vitb16/libero-256px-8f-debug.yaml。
```

### 2. 看到 skipped incompatible keys

现象：

```text
Skipped 12 incompatible pretrained keys
missing_keys=['module.norm.weight', 'module.norm.bias']
```

处理：

```text
这是预期行为。V-JEPA 2.1 checkpoint 含有当前 robot encoder 不使用的 image/modality 相关参数。
utils.py 已按目标模型 state_dict 和 tensor shape 做过滤。
```

### 3. LIBERO camera key 不存在

现象：

```text
Camera key 'agentview_rgb' not found in LIBERO obs
```

处理：

```text
检查 HDF5 内 obs 的相机字段。
如果要用 wrist camera，把配置改成 libero_camera_key: eye_in_hand_rgb。
```

### 4. 没有 data/libero/libero_spatial_h5_paths.csv

处理：

```text
先下载 LIBERO HDF5 数据，并生成一行一个 .hdf5 路径的 CSV。
当前 loader 不要求转换成 DROID-like 目录。
```

### 5. NCCL destroy_process_group warning

现象：

```text
Warning: destroy_process_group() was not called before program exit
```

处理：

```text
当前 debug run 可以忽略；训练已保存产物。
```

## 后续扩展

如果 smoke test 通过，下一步可以扩大训练：

```yaml
optimization:
  epochs: 5
  ipe: 100
```

更正式的实验建议：

```text
1. 划分 train/val HDF5 或 demo list。
2. 记录 validation latent prediction loss。
3. 用 held-out trajectory 画 energy landscape。
4. 再接 LIBERO env 做闭环 success rate。
```

如果要从“只训练 AC predictor”升级到更强适配：

```text
优先考虑解冻 encoder 最后几层或加 LoRA / Adapter。
不要一开始全量微调 ViT-g。
```
