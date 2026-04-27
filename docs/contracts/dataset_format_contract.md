# Robot Dataset Format Contract

本文档面向数据生成同事，定义当前 V-JEPA 2-AC 机器人训练代码所需的数据格式。

对应读取代码：

- [app/vjepa_droid/droid.py](../../app/vjepa_droid/droid.py)

当前代码读取的是 DROID-like trajectory 数据。训练配置中通过 CSV 指定所有 trajectory 目录。

## 1. 总体结构

训练集入口是一个 CSV 文件，例如：

```text
/data/robot/train_paths.csv
```

CSV 每行一个 trajectory 目录路径：

```text
/data/robot/traj_000001
/data/robot/traj_000002
/data/robot/traj_000003
```

每个 trajectory 目录必须包含：

```text
traj_xxxxxx/
  trajectory.h5
  metadata.json
  recordings/
    MP4/
      <camera_video>.mp4
```

示例：

```text
traj_000001/
  trajectory.h5
  metadata.json
  recordings/
    MP4/
      left_000001.mp4
      right_000001.mp4
      wrist_000001.mp4
```

## 2. CSV 文件要求

代码读取方式：

```python
pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0]
```

因此 CSV/文本文件要求：

- 每行第 1 列是 trajectory 目录路径。
- 分隔符为空格。
- 不要写 header。
- 路径可以是绝对路径，推荐使用绝对路径。
- 如果后面有额外列，当前 loader 会忽略，只使用第 1 列。

推荐格式：

```text
/abs/path/to/traj_000001
/abs/path/to/traj_000002
```

不推荐格式：

```text
path,label
/abs/path/to/traj_000001,0
```

## 3. metadata.json 要求

`trajectory` 目录下需要有一个 `.json` 文件。文件名不固定，但建议统一命名为：

```text
metadata.json
```

当前代码会遍历目录中第一个 `.json` 文件并读取。

metadata 中必须包含训练配置 `camera_views` 指定的 key。

配置示例：

```yaml
data:
  camera_views:
    - left_mp4_path
```

则 `metadata.json` 必须包含：

```json
{
  "left_mp4_path": "recordings/MP4/left_000001.mp4"
}
```

如果配置多个 camera view：

```yaml
data:
  camera_views:
    - left_mp4_path
    - right_mp4_path
    - wrist_mp4_path
```

则 metadata 应包含：

```json
{
  "left_mp4_path": "recordings/MP4/left_000001.mp4",
  "right_mp4_path": "recordings/MP4/right_000001.mp4",
  "wrist_mp4_path": "recordings/MP4/wrist_000001.mp4"
}
```

注意：

- 当前代码会用 `metadata[camera_view].split("recordings/MP4/")[-1]` 取得视频文件名。
- 因此 value 推荐写成 `recordings/MP4/<filename>.mp4`。
- `<filename>` 不包含路径后会被拼成 `traj_dir/recordings/MP4/<filename>`。

## 4. 视频文件要求

视频路径由 metadata 决定，最终代码会读取：

```text
<trajectory_dir>/recordings/MP4/<camera_video>.mp4
```

要求：

- 格式：MP4。
- 可被 `decord.VideoReader` 读取。
- RGB 视频。
- 视频帧数必须足够长。
- 视频帧顺序必须与 `trajectory.h5` 中 robot state 的时间顺序对齐。

### 帧数要求

代码中会计算：

```python
vfps = video_fps
fps = config_fps
fstp = ceil(vfps / fps)
nframes = frames_per_clip * fstp
```

视频长度必须满足：

```text
len(video_frames) >= nframes
```

以默认机器人训练配置为例：

```yaml
data:
  dataset_fpcs:
    - 8
  fps: 4
```

如果视频实际 fps 是 30：

```text
fstp = ceil(30 / 4) = 8
nframes = 8 * 8 = 64
```

所以视频至少需要 64 帧。

## 5. trajectory.h5 要求

每个 trajectory 目录必须包含：

```text
trajectory.h5
```

代码会读取以下 HDF5 key。

### 5.1 Robot Cartesian Position

路径：

```text
observation/robot_state/cartesian_position
```

shape：

```text
[T, 6]
```

含义：

```text
x, y, z, roll, pitch, yaw
```

要求：

- 单位建议使用米和弧度。
- 欧拉角顺序必须是 `xyz`。
- 时间维度 `T` 必须与视频帧时间轴对齐。

### 5.2 Gripper Position

路径：

```text
observation/robot_state/gripper_position
```

shape：

```text
[T]
```

或可被读取后变成：

```text
[T, 1]
```

含义：

```text
gripper
```

建议：

- 使用连续值。
- 保持全数据集含义一致，例如 `0=open, 1=closed`。

代码会拼出 7D state：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

shape：

```text
[T, 7]
```

### 5.3 Camera Extrinsics

路径格式：

```text
observation/camera_extrinsics/<camera_name>_left
```

shape：

```text
[T, 6]
```

含义：

```text
x, y, z, roll, pitch, yaw
```

`camera_name` 的推导方式：

```python
mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
camera_name = mp4_name.split(".")[0]
extrinsics_key = f"{camera_name}_left"
```

示例：

metadata：

```json
{
  "left_mp4_path": "recordings/MP4/left_000001.mp4"
}
```

则代码会读取：

```text
observation/camera_extrinsics/left_000001_left
```

因此 `trajectory.h5` 必须包含这个 key。

如果你希望 key 更稳定，建议视频文件名使用稳定 camera name，例如：

```text
recordings/MP4/left.mp4
```

对应 extrinsics key：

```text
observation/camera_extrinsics/left_left
```

## 6. 时间对齐要求

视频帧、robot state、camera extrinsics 必须按同一时间轴对齐。

代码会先从视频中随机采样原始帧 index：

```python
indices = np.arange(sf, sf + nframes, fstp)
```

然后用同一组 `indices` 读取：

```python
states = states[indices, :]
extrinsics = extrinsics[indices, :]
frames = video[indices]
```

因此要求：

```text
len(video_frames) <= T_state 时，所有 sampled video indices 必须能索引 state/extrinsics
```

更稳妥的要求：

```text
T_video == T_state == T_extrinsics
```

如果视频和 robot state 频率不同，建议在生成数据时先重采样对齐，而不是依赖训练代码处理。

## 7. Action 的生成方式

数据中不需要显式保存 action。

当前 loader 会从连续 robot states 自动计算 action：

```text
state_t     = [x, y, z, roll, pitch, yaw, gripper]
state_{t+1} = [x, y, z, roll, pitch, yaw, gripper]
```

生成：

```text
action_t = [
  delta_x,
  delta_y,
  delta_z,
  delta_roll,
  delta_pitch,
  delta_yaw,
  delta_gripper
]
```

其中：

- 平移部分：直接相减。
- 旋转部分：先把 `xyz` 欧拉角转 rotation matrix，再计算相对旋转。
- gripper：直接相减。

如果采样后有 `T_clip` 个 state，则 action shape 是：

```text
[T_clip - 1, 7]
```

## 8. 训练时返回的数据 shape

单条 sample 返回：

```text
buffer, actions, states, extrinsics, indices
```

经过 batch collate 后：

```text
clips:      [B, C, T_raw, H, W]
actions:    [B, T_model - 1, 7]
states:     [B, T_model, 7]
extrinsics: [B, T_model, 6]
indices:    [B, T_raw]
```

说明：

- `clips` 使用视频 transform 后是 channel-first。
- `T_raw` 是视频采样帧数，通常为 `frames_per_clip * fstp`。
- `T_model` 是应用 `frameskip` 后的 state/action 时间长度。
- 当前配置中 `frameskip` 来自 `tubelet_size`。

## 9. 推荐数据质量要求

每条 trajectory 建议满足：

- 视频无损坏，可用 decord 读取。
- 视频帧和 robot state 对齐。
- robot pose 没有 NaN/Inf。
- gripper 数值范围统一。
- 欧拉角单位为弧度。
- 每条视频长度大于训练配置所需最短帧数。
- metadata 中所有配置的 camera view key 都存在。
- HDF5 中 camera extrinsics key 和视频文件名匹配。

## 10. 最小可用样例

目录：

```text
/data/robot/traj_000001/
  metadata.json
  trajectory.h5
  recordings/
    MP4/
      left.mp4
```

metadata.json：

```json
{
  "left_mp4_path": "recordings/MP4/left.mp4"
}
```

trajectory.h5：

```text
observation/robot_state/cartesian_position      [T, 6]
observation/robot_state/gripper_position        [T]
observation/camera_extrinsics/left_left         [T, 6]
```

train_paths.csv：

```text
/data/robot/traj_000001
```

config：

```yaml
data:
  datasets:
    - /data/robot/train_paths.csv
  camera_views:
    - left_mp4_path
```

## 11. 交付 Checklist

数据生成完成后，请逐项确认：

1. 有一个 train CSV，且每行第 1 列是 trajectory 目录绝对路径。
2. 每个 trajectory 目录下有 `trajectory.h5`。
3. 每个 trajectory 目录下有一个 `.json` metadata 文件。
4. metadata 中包含训练配置指定的 `camera_views` key。
5. metadata 中的视频路径指向 `recordings/MP4/<file>.mp4`。
6. MP4 文件存在且可被 decord 打开。
7. HDF5 中存在 `observation/robot_state/cartesian_position`，shape 为 `[T, 6]`。
8. HDF5 中存在 `observation/robot_state/gripper_position`，shape 为 `[T]`。
9. HDF5 中存在对应 camera 的 extrinsics key，shape 为 `[T, 6]`。
10. 视频帧数、robot state 长度、extrinsics 长度在时间上对齐。
11. 所有 pose 数值无 NaN/Inf。
12. 欧拉角为弧度，顺序为 `xyz`。
