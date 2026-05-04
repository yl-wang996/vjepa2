# LIBERO 测试数据集接入方案

本文说明如何使用 LIBERO 作为 V-JEPA 2-AC 机器人微调/验证的小型测试数据源。

## 为什么选 LIBERO

LIBERO 是机器人学习 benchmark，包含 Franka/Panda 机械臂仿真任务、RGB 观测、末端状态和动作。相比完整 DROID 数据集，LIBERO 更适合先做 pipeline smoke test 和离线/仿真性能验证。

适合用途：

- 测试自有 dataset loader / converter。
- 打通 V-JEPA 2-AC robot post-training 数据管线。
- 训练小规模 action-conditioned predictor。
- 用仿真任务做后续性能验证。

不适合用途：

- 直接代表真实机器人效果。
- 直接代表真实机器人数据分布；它主要用于先验证 pipeline。

参考：

- LIBERO GitHub：https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot LIBERO 文档：https://huggingface.co/docs/lerobot/libero

## 当前 repo 需要的格式

当前机器人训练代码读取 DROID-like 格式：

```text
traj_xxx/
  trajectory.h5
  metadata.json
  recordings/
    MP4/
      <camera>.mp4
```

详细 contract 见：

[docs/contracts/dataset_format_contract.md](contracts/dataset_format_contract.md)

LIBERO 原始数据通常是 HDF5 demo 格式。当前 repo 已经支持两种接入方式：

1. **直接读取 LIBERO HDF5**：推荐，用 `dataset_format: libero`。
2. **转换成 DROID-like 格式**：可选，用于兼容旧 loader 或导出检查。

## 推荐方式：直接 LIBERO Dataloader

已提供直接读取 LIBERO HDF5 的 dataloader：

[app/vjepa_droid/libero.py](../app/vjepa_droid/libero.py)

当前本地已下载 `libero_spatial`：

```text
data/libero/libero_spatial/
data/libero/libero_spatial_h5_paths.csv
```

数据规模：

```text
10 HDF5 files
500 demos
about 5.9 GB
```

CSV 可以直接列出 HDF5 文件：

```text
/data/libero/demo_file_1.hdf5
/data/libero/demo_file_2.hdf5
```

如果只想使用某个 HDF5 里的特定 demo，也可以写第二列：

```text
/data/libero/demo_file_1.hdf5 demo_0
/data/libero/demo_file_1.hdf5 demo_1
```

训练配置中打开：

```yaml
data:
  dataset_format: libero
  datasets:
    - /data/libero/libero_h5_paths.csv
  libero_camera_key: agentview_rgb
  libero_frame_stride: 1
  dataset_fpcs:
    - 8
```

字段读取逻辑：

```text
obs/<libero_camera_key>              -> clips
obs/ee_states[:, :6]                 -> robot cartesian state
obs/gripper_states 或 actions[:, -1] -> gripper state
dummy zeros [T, 6]                   -> extrinsics
```

说明：

- 默认 `libero_camera_key` 是 `agentview_rgb`。
- 如果使用 wrist camera，可以改为 `eye_in_hand_rgb`。
- 当前 loader 会对 LIBERO 原始 RGB 帧做一次垂直翻转（`flipud`），使训练和可视化方向与预期一致。
- LIBERO 通常没有 DROID-style camera extrinsics，所以 loader 返回全零 extrinsics。
- 训练配置中建议保持 `model.use_extrinsics: false`。

直接 dataloader 的优点：

- 不需要把 HDF5 转成 MP4，少占磁盘。
- 不引入视频编码损失。
- 更容易和后续 LIBERO 仿真评估对齐。

## 可选方式：转换成 DROID-like 格式

如果需要用旧 DROID-like 目录结构，或者想把中间数据落盘检查，可以使用转换脚本。

已提供转换脚本：

[scripts/libero_to_droid_like.py](../scripts/libero_to_droid_like.py)

它会读取 LIBERO HDF5，并输出：

```text
output_root/
  <h5_name>_demo_0/
    metadata.json
    trajectory.h5
    recordings/MP4/agentview_rgb.mp4
  <h5_name>_demo_1/
    ...

train_paths.csv
```

转换后的数据可以被当前 `app.vjepa_droid.droid.DROIDVideoDataset` 读取。

## 字段映射

LIBERO -> DROID-like：

```text
obs/<camera_key>                    -> recordings/MP4/<camera_key>.mp4
obs/ee_states[:, :6]                -> observation/robot_state/cartesian_position
obs/gripper_states 或 actions[:, -1] -> observation/robot_state/gripper_position
dummy zeros [T, 6]                  -> observation/camera_extrinsics/<camera>_left
```

说明：

- 默认 camera key 是 `agentview_rgb`。
- 如果数据中使用其他相机，例如 `eye_in_hand_rgb`，用 `--camera-key` 指定。
- LIBERO 通常没有 DROID-style camera extrinsics；脚本写入全零 extrinsics，仅用于满足当前 loader 的字段要求。
- 训练配置中应保持 `use_extrinsics: false`。

## 转换命令

示例：只转换每个 HDF5 文件前 10 条 demo，用于快速测试。

```bash
cd /home/yunlong/workspace/vjepa2

.venv/bin/python scripts/libero_to_droid_like.py \
  --input /path/to/libero_hdf5_or_dir \
  --output-root /data/libero_droid_like \
  --csv /data/libero_droid_like/train_paths.csv \
  --camera-key agentview_rgb \
  --metadata-camera-key left_mp4_path \
  --max-demos-per-file 10
```

如果要用 wrist camera：

```bash
.venv/bin/python scripts/libero_to_droid_like.py \
  --input /path/to/libero_hdf5_or_dir \
  --output-root /data/libero_droid_like_wrist \
  --csv /data/libero_droid_like_wrist/train_paths.csv \
  --camera-key eye_in_hand_rgb \
  --metadata-camera-key left_mp4_path \
  --max-demos-per-file 10
```

## 转换后检查

检查 CSV：

```bash
head /data/libero_droid_like/train_paths.csv
```

检查单条 trajectory：

```bash
traj=$(head -n 1 /data/libero_droid_like/train_paths.csv)
find "$traj" -maxdepth 3 -type f
```

应看到：

```text
metadata.json
trajectory.h5
recordings/MP4/agentview_rgb.mp4
```

检查 HDF5：

```bash
.venv/bin/python - <<'PY'
import h5py
from pathlib import Path

traj = Path(open("/data/libero_droid_like/train_paths.csv").readline().strip())
with h5py.File(traj / "trajectory.h5", "r") as f:
    def show(name, obj):
        if hasattr(obj, "shape"):
            print(name, obj.shape)
    f.visititems(show)
PY
```

应包含：

```text
observation/robot_state/cartesian_position      (T, 6)
observation/robot_state/gripper_position        (T,)
observation/camera_extrinsics/agentview_rgb_left (T, 6)
```

## 训练配置建议

基于官方配置：

[configs/train/vitg16/droid-256px-8f.yaml](../configs/train/vitg16/droid-256px-8f.yaml)

已提供两个 LIBERO 调试配置：

```text
configs/train/vitg16/libero-256px-8f-debug.yaml
configs/train/vitb16/libero-256px-8f-debug.yaml
```

说明：

- `vitg16`：保留官方 V-JEPA 2-AC / ViT-g 结构，适合作为大显存训练模板。24GB 单卡会在 Adam optimizer step 附近 OOM。
- `vitb16`：单卡 smoke test 配置，加载 V-JEPA 2.1 ViT-B `ema_encoder`，随机初始化 AC predictor，用于确认 LIBERO 数据、预训练加载、forward/backward、checkpoint 保存都能跑通。

核心改动：

```yaml
app: vjepa_droid
folder: outputs/train/libero_vjepa_ac_debug
nodes: 1
tasks_per_node: 1

data:
  dataset_format: libero
  batch_size: 1
  datasets:
    - /data/libero/libero_h5_paths.csv
  dataset_fpcs:
    - 8
  libero_camera_key: agentview_rgb
  libero_frame_stride: 1
  fps: 4
  num_workers: 2
  crop_size: 256
  patch_size: 16
  tubelet_size: 2

model:
  use_extrinsics: false

optimization:
  epochs: 5
  ipe: 100
```

## 已跑通的训练 Smoke Test

本地已用 `libero_spatial` 跑通 24GB 单卡训练 smoke test：

```bash
cd /home/yunlong/workspace/vjepa2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m app.main \
  --fname configs/train/vitb16/libero-256px-8f-debug.yaml \
  --devices cuda:0 \
  --debugmode True
```

结果：

```text
dataset length: 500
checkpoint: checkpoints/vjepa2_1_vitb_dist_vitG_384.pt
model: vit_base encoder + AC predictor
max cuda memory allocated: about 1.85 GB
loss step 0: 1.871
loss step 1: 1.721
```

产物：

```text
outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt
outputs/train/libero-vjepa21-vitb-ac-debug/e0.pt
outputs/train/libero-vjepa21-vitb-ac-debug/log_r0.csv
outputs/train/libero-vjepa21-vitb-ac-debug/params-pretrain.yaml
```

## 已跑通的离线验证和 Energy Landscape

训练 smoke test 之后，可以用验证脚本读取 `latest.pt`，在 LIBERO demo 上生成一段 clip 可视化和 `dx/dz` action energy landscape：

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

产物：

```text
outputs/validation/libero-vjepa21-vitb-ac-debug/libero_clip_frames.png
outputs/validation/libero-vjepa21-vitb-ac-debug/energy_landscape_dx_dz.png
outputs/validation/libero-vjepa21-vitb-ac-debug/metrics.json
```

当前 debug checkpoint 只训练了 2 个 step，energy landscape 只能说明验证管线跑通，不能当作真实性能结论。正式比较时应使用更长训练后的 checkpoint，并在 held-out demos 上统计指标。

先用最小数据和小训练步数确认：

- dataset 能读；
- loss 能 forward/backward；
- checkpoint 能保存；
- 用 held-out trajectory 能画 energy landscape。

## 性能验证思路

LIBERO 的价值是后面可以接仿真验证。建议分三层评估：

### 1. 数据管线验证

目标：确认转换数据可被 V-JEPA 2-AC 训练代码读取。

检查：

- `clips/actions/states/extrinsics` shape 正确。
- 视频和 state 时间长度对齐。
- loss 不为 NaN。

### 2. 离线 world model 验证

目标：确认模型学到 action-conditioned latent dynamics。

方法：

- 从验证 demo 中取当前帧和目标帧。
- 计算 ground-truth action。
- 画 energy landscape。
- 看 ground-truth action 是否处在低 energy 区域附近。

### 3. 仿真闭环验证

目标：在 LIBERO 仿真任务中测试规划动作是否能提升任务成功率。

高层流程：

```text
reset LIBERO env
获取当前 RGB + robot state
给定目标图像或子目标图像
CEM 用 V-JEPA 2-AC world model 搜索 action
env.step(action)
循环直到完成或超时
统计 success rate
```

这一步需要单独接 LIBERO env API，不是当前 repo 已有功能。

## 注意事项

- LIBERO 是仿真数据，视觉域和真实机器人有差异。
- 转换脚本写入的是 dummy camera extrinsics，所以训练配置应保持 `use_extrinsics: false`。
- LIBERO action/state 定义可能和真实 Franka 控制接口不同，做真实机器人迁移时需要重新对齐 action convention。
- 当前转换脚本主要用于 pipeline smoke test；正式训练前建议抽样可视化视频、state 曲线和 action 分布。

## 最小 Checklist

1. 下载一小份 LIBERO HDF5 数据。
2. 生成 HDF5 path CSV，例如 `data/libero/libero_spatial_h5_paths.csv`。
3. 用 `dataset_format: libero` 直接跑 dataloader。
4. 用 `configs/train/vitb16/libero-256px-8f-debug.yaml` 跑通小 batch 训练。
5. 检查 `latest.pt` 和 `log_r0.csv`。
6. 做 held-out trajectory 的 energy landscape。
7. 再考虑接 LIBERO 仿真闭环评估。
