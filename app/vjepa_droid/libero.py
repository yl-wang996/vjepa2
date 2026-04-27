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
    data = pd.read_csv(path, header=None, delimiter=" ")
    return data.values


def _iter_demo_keys(h5_file):
    root = h5_file["data"] if "data" in h5_file else h5_file
    for key in sorted(root.keys()):
        if isinstance(root[key], h5py.Group):
            yield key


class LIBEROHDF5Dataset(torch.utils.data.Dataset):
    """LIBERO HDF5 dataset returning the same tuple as DROIDVideoDataset.

    Expected CSV formats:
      /path/to/file.hdf5
      /path/to/file.hdf5 demo_0

    If only the file path is provided, all demos in that HDF5 file are used.
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
                logger.info(f"Encountered exception when loading LIBERO sample {h5_path=} {demo_key=} {e=}")
                index = np.random.randint(self.__len__())
        return sample

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
        return frames

    def _cartesian_position(self, obs):
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
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack([d for d in angle_diff], axis=0)
        closedness_delta = poses[1:, -1:] - poses[:-1, -1:]
        return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)

    def load_demo(self, h5_path, demo_key):
        with h5py.File(h5_path, "r") as h5_file:
            demo = self._demo(h5_file, demo_key)
            obs = self._obs(demo)
            frames = self._frames(obs)
            cartesian = self._cartesian_position(obs)
            length = min(len(frames), len(cartesian))
            gripper = self._gripper_position(obs, demo, length)

            required = self.frames_per_clip * self.frame_stride
            if length < required:
                raise ValueError(f"LIBERO demo too short: {h5_path}:{demo_key} length={length} required={required}")

            end = np.random.randint(required, length + 1)
            start = end - required
            indices = np.arange(start, end, self.frame_stride).astype(np.int64)
            buffer = frames[indices]
            states = np.concatenate([cartesian[indices], gripper[indices, None]], axis=1).astype(np.float32)
            extrinsics = np.zeros((len(indices), 6), dtype=np.float32)
            actions = self.poses_to_diffs(states).astype(np.float32)

        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices
