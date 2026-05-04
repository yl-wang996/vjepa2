# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

logger = getLogger()


def _read_table(path):
    # 本仓库的数据 CSV 约定是空格分隔、无表头。
    # 示例行：
    #   /abs/path/pick_up_the_black_bowl_on_the_stove_demo.hdf5
    #   /abs/path/pick_up_the_black_bowl_on_the_stove_demo.hdf5 demo_0
    data = pd.read_csv(path, header=None, delimiter=" ")
    return data.values


def _iter_demo_keys(h5_file):
    # LIBERO 文件通常是 /data/demo_N 结构；这里保留 root fallback，
    # 兼容少数直接把 demo_N 放在 HDF5 根目录下的导出格式。
    root = h5_file["data"] if "data" in h5_file else h5_file
    for key in sorted(root.keys()):
        if isinstance(root[key], h5py.Group):
            yield key


class LIBEROHDF5Dataset(torch.utils.data.Dataset):
    """读取 LIBERO HDF5，并返回和 DROIDVideoDataset 对齐的元组。

    当配置里使用下面选项时，会走这条直接读取 HDF5 的路径：
      data.dataset_format: libero

    典型 LIBERO HDF5 结构：
      data/
        demo_0/
          actions                 [T, 7]
          obs/
            agentview_rgb         [T, 128, 128, 3] uint8
            eye_in_hand_rgb       [T, 128, 128, 3] uint8
            ee_states             [T, 6]  xyz + rpy
            gripper_states        [T, 2]  左右夹爪数值
          rewards                 [T]
          dones                   [T]

    支持的 CSV 格式：
      /path/to/file.hdf5
      /path/to/file.hdf5 demo_0

    如果只提供 HDF5 路径，不指定 demo key，会使用该文件里的所有 demo_N。

    __getitem__ 返回：
      buffer      transform 后 [C, T, H, W]，transform 前 [T, H, W, 3]
      actions     [T - 1, 7]  delta xyz、delta rpy、delta gripper
      states      [T, 7]      xyz, rpy, gripper
      extrinsics  [T, 6]      LIBERO 路径下为全 0
      indices     [T]         采样到的原始帧索引

    frames_per_clip=8、frame_stride=1 时的示例：
      indices = [125, 126, 127, 128, 129, 130, 131, 132]
      states[0] ~= [-0.20, 0.00, 1.17, 3.14, -0.00, -0.05, 0.00]
      actions[0] 是由 states[1] 和 states[0] 推出的位姿差分。
    """

    def __init__(
        self,
        data_path,
        camera_key="agentview_rgb",
        frames_per_clip=8,
        frame_stride=1,
        transform=None,
        camera_frame=False,
    ):
        self.data_path = data_path
        self.camera_key = camera_key
        self.frames_per_clip = frames_per_clip
        self.frame_stride = frame_stride
        self.transform = transform
        self.camera_frame = camera_frame
        self.samples = self._load_samples(data_path)

    def _load_samples(self, data_path):
        samples = []
        for row in _read_table(data_path):
            h5_path = Path(str(row[0]))
            # 第二列用于把这一行固定到某个 demo；如果没有第二列，
            # 文件里的每个 demo_N group 都会成为一个单独样本。
            if len(row) > 1 and isinstance(row[1], str):
                samples.append((h5_path, row[1]))
                continue
            with h5py.File(h5_path, "r") as h5_file:
                samples.extend((h5_path, demo_key) for demo_key in _iter_demo_keys(h5_file))
        if not samples:
            raise ValueError(f"No LIBERO demos found from {data_path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        loaded = False
        while not loaded:
            h5_path, demo_key = self.samples[index]
            try:
                sample = self.load_demo(h5_path, demo_key)
                loaded = True
            except Exception as e:
                # 单个 demo 损坏或太短时，让长训练继续跑下去。
                # 调试具体样本时，建议直接调用 load_demo，这样可以看到原始异常。
                logger.info(f"Encountered exception when loading LIBERO sample {h5_path=} {demo_key=} {e=}")
                index = np.random.randint(self.__len__())
        return sample

    def _randint(self, low, high, rng=None):
        if rng is None:
            return int(np.random.randint(low, high))
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))

    def sample_indices(self, length, rng=None, end=None):
        """采样一个固定长度 clip，clip 结束位置是随机有效帧。

        应用 frame_stride 后，采样窗口在时间上仍是连续的。
        示例：
          length=155, frames_per_clip=8, frame_stride=2, end=132
          required=16, start=116, indices=[116,118,120,122,124,126,128,130]

        注意：YAML 里的 data.fps 不会被 LIBERO 分支使用；控制 LIBERO
        时间间隔的旋钮是这里的 frame_stride。
        """
        required = self.frames_per_clip * self.frame_stride
        if length < required:
            raise ValueError(f"LIBERO demo too short: length={length} required={required}")
        if end is None:
            end = self._randint(required, length + 1, rng=rng)
        if end < required or end > length:
            raise ValueError(f"Invalid end index: {end} for length={length} required={required}")
        start = end - required
        return np.arange(start, end, self.frame_stride).astype(np.int64)

    def _demo(self, h5_file, demo_key):
        root = h5_file["data"] if "data" in h5_file else h5_file
        return root[demo_key]

    def _obs(self, demo):
        if "obs" in demo:
            return demo["obs"]
        if "observations" in demo:
            return demo["observations"]
        raise KeyError("LIBERO demo must contain obs or observations group")

    def _frames(self, obs):
        """读取一路 RGB 相机，并整理成 uint8 NHWC。

        常见输入 shape：
          LIBERO 原生 [T, H, W, 3]，例如 [155, 128, 128, 3]
          某些其它导出格式可能是 [T, 3, H, W]

        返回值固定为 [T, H, W, 3] uint8。
        """
        if self.camera_key not in obs:
            raise KeyError(f"Camera key '{self.camera_key}' not found in LIBERO obs")
        frames = np.asarray(obs[self.camera_key])
        if frames.ndim != 4:
            raise ValueError(f"Expected camera frames rank 4, got {frames.shape}")
        if frames.shape[1] == 3 and frames.shape[-1] != 3:
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames, got {frames.shape}")
        if frames.dtype != np.uint8:
            if frames.size and float(np.nanmax(frames)) <= 1.0:
                frames = frames * 255.0
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        # LIBERO RGB observation 相对训练/可视化期望方向是上下倒置的，
        # 因此这里统一做一次垂直翻转。
        frames = np.flip(frames, axis=1).copy()
        return frames

    def _cartesian_position(self, obs):
        """返回末端执行器位姿：[x, y, z, roll, pitch, yaw]。

        对本地 libero_spatial 文件，obs/ee_states 已经是 [T, 6]。
        第一帧示例：
          [-0.2008, 0.0016, 1.1730, 3.1378, -0.0025, -0.0524]

        某些 robosuite 风格文件会存 position + quaternion；这里会把它们
        转成 xyz 欧拉角，使下游看到一致的 DROID-like cartesian_position 契约。
        """
        if "ee_states" in obs:
            ee_states = np.asarray(obs["ee_states"])
            if ee_states.shape[-1] < 6:
                raise ValueError(f"obs/ee_states must have at least 6 dims, got {ee_states.shape}")
            return ee_states[:, :6].astype(np.float32)

        if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
            eef_pos = np.asarray(obs["robot0_eef_pos"])
            eef_quat = np.asarray(obs["robot0_eef_quat"])
            euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
            return np.concatenate([eef_pos, euler], axis=-1).astype(np.float32)

        raise KeyError("Could not infer cartesian pose; expected obs/ee_states or robot0_eef_pos+robot0_eef_quat")

    def _gripper_position(self, obs, demo, target_len):
        """每帧返回一个标量夹爪状态。

        当前 DROID-like 状态向量需要最后一维夹爪值：
          states[t] = [x, y, z, roll, pitch, yaw, gripper]

        LIBERO gripper_states 通常是 [T, 2]，每个夹爪 finger 一个值。示例：
          [0.0362, -0.0363]

        当前实现为了兼容已有状态形状，会对两个 finger 取平均。
        在这个数据集里，两个 finger 经常一正一负，所以平均值可能接近 0。
        如果实验关心夹爪开合，建议改成夹爪宽度（例如 abs(left - right)）
        或直接使用 demo/actions[:, -1]。
        """
        if "gripper_states" in obs:
            gripper = np.asarray(obs["gripper_states"])
            if gripper.ndim == 2:
                gripper = gripper.mean(axis=-1)
            return gripper[:target_len].astype(np.float32)

        if "actions" in demo:
            actions = np.asarray(demo["actions"])
            if actions.shape[-1] >= 1:
                return actions[:target_len, -1].astype(np.float32)

        return np.zeros((target_len,), dtype=np.float32)

    def poses_to_diffs(self, poses):
        """把绝对 states [T, 7] 转成类似动作的差分 [T - 1, 7]。

        AC 预测器需要用相邻帧之间的动作做条件。
        这里没有使用 LIBERO 原始 demo/actions，而是从 state 推导：
          delta xyz          = xyz[t+1] - xyz[t]
          delta rpy          = 相对旋转，并表示成 xyz Euler
          delta gripper      = gripper[t+1] - gripper[t]

        相邻 LIBERO 帧的输出量级示例：
          dx/dy/dz 可能约为每帧 1e-4 到 1e-2 米。
          旋转差分可能约为每帧 1e-4 到 1e-2 弧度。
        """
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness_delta = poses[1:, -1:] - poses[:-1, -1:]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def transform_frame(self, poses, extrinsics):
        """可选：把位姿表达到相机坐标系。

        DROID trajectory 可以提供真实相机外参，但 LIBERO HDF5 通常没有。
        因此 load_demo 会提供全 0 外参。全 0 外参下，这个变换基本
        等价于 no-op，所以 LIBERO 配置通常应保持：
          data.camera_frame: false
          model.use_extrinsics: false
        """
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]
            theta = pose[3:6]
            rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rot
            transform[:3, 3] = trans
            return transform

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            rot = transform[:3, :3]
            angle = Rotation.from_matrix(rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for pose, extrinsic in zip(poses, extrinsics):
            pose_transform = pose_to_transform(pose)
            extrinsic_transform = pose_to_transform(extrinsic)
            new_pose_transform = np.linalg.inv(extrinsic_transform) @ pose_transform
            new_pose.append(transform_to_pose(new_pose_transform))
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1).astype(np.float32)

    def load_demo(self, h5_path, demo_key, rng=None, end=None):
        """从一个 LIBERO demo 中加载一个随机固定长度训练样本。

        本地 libero_spatial demo_0 示例：
          完整 demo 长度：155 帧
          frames_per_clip: 8
          frame_stride: 1
          transform 前选中帧形状: [8, 128, 128, 3]
          states shape: [8, 7]
          actions shape: [7, 7]
          extrinsics shape: [8, 6]，全 0

        可选的 end 参数会固定采样窗口，适合确定性验证或可视化。
        """
        with h5py.File(h5_path, "r") as h5_file:
            demo = self._demo(h5_file, demo_key)
            obs = self._obs(demo)
            frames = self._frames(obs)
            cartesian = self._cartesian_position(obs)
            length = min(len(frames), len(cartesian))
            gripper = self._gripper_position(obs, demo, length)
            indices = self.sample_indices(length, rng=rng, end=end)
            buffer = frames[indices]
            states = np.concatenate([cartesian[indices], gripper[indices, None]], axis=1).astype(np.float32)
            # LIBERO 不包含 DROID 风格相机外参；但模型仍然期望 DROID 元组形状。
            # 因此这里返回全 0，并在 LIBERO 配置里保持 model.use_extrinsics: false。
            extrinsics = np.zeros((len(indices), 6), dtype=np.float32)
            if self.camera_frame:
                states = self.transform_frame(states, extrinsics)
            actions = self.poses_to_diffs(states).astype(np.float32)

        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices
