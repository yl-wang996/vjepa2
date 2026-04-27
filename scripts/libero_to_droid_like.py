#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert LIBERO HDF5 demonstrations to the DROID-like layout used by app.vjepa_droid.

The converter is intentionally conservative: it writes the fields that the current
DROIDVideoDataset actually reads, and stores dummy camera extrinsics because LIBERO
datasets usually do not carry DROID-style camera extrinsics.
"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def _as_uint8_rgb(frames):
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with rank 4, got shape {frames.shape}")

    if frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames, got shape {frames.shape}")

    if frames.dtype != np.uint8:
        max_value = float(np.nanmax(frames)) if frames.size else 0.0
        if max_value <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _write_mp4(frames, path, fps):
    frames = _as_uint8_rgb(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _dataset(group, name):
    return np.asarray(group[name]) if name in group else None


def _find_obs(demo):
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise KeyError("Could not find obs/observations group in LIBERO demo")


def _find_cartesian_position(obs):
    ee_states = _dataset(obs, "ee_states")
    if ee_states is not None:
        if ee_states.shape[-1] < 6:
            raise ValueError(f"obs/ee_states must have at least 6 dims, got {ee_states.shape}")
        return ee_states[:, :6].astype(np.float32)

    eef_pos = _dataset(obs, "robot0_eef_pos")
    eef_quat = _dataset(obs, "robot0_eef_quat")
    if eef_pos is not None and eef_quat is not None:
        euler = Rotation.from_quat(eef_quat).as_euler("xyz", degrees=False)
        return np.concatenate([eef_pos, euler], axis=-1).astype(np.float32)

    raise KeyError("Could not infer [x,y,z,roll,pitch,yaw]; expected obs/ee_states or robot0_eef_pos+quat")


def _find_gripper(obs, demo, target_len):
    gripper = _dataset(obs, "gripper_states")
    if gripper is not None:
        gripper = np.asarray(gripper)
        if gripper.ndim == 2:
            gripper = gripper.mean(axis=-1)
        return gripper[:target_len].astype(np.float32)

    actions = _dataset(demo, "actions")
    if actions is not None and actions.shape[-1] >= 1:
        return actions[:target_len, -1].astype(np.float32)

    return np.zeros((target_len,), dtype=np.float32)


def _demo_items(h5_file):
    root = h5_file["data"] if "data" in h5_file else h5_file
    for key in sorted(root.keys()):
        item = root[key]
        if isinstance(item, h5py.Group):
            yield key, item


def convert_file(
    h5_path,
    output_root,
    camera_key,
    metadata_camera_key,
    output_fps,
    max_demos=None,
):
    written = []
    h5_path = Path(h5_path)
    output_root = Path(output_root)
    with h5py.File(h5_path, "r") as h5_file:
        for demo_index, (demo_name, demo) in enumerate(_demo_items(h5_file)):
            if max_demos is not None and demo_index >= max_demos:
                break

            obs = _find_obs(demo)
            if camera_key not in obs:
                raise KeyError(f"Camera key '{camera_key}' not found in {h5_path}:{demo_name}/obs")

            frames = _as_uint8_rgb(obs[camera_key])
            cartesian = _find_cartesian_position(obs)
            traj_len = min(len(frames), len(cartesian))
            frames = frames[:traj_len]
            cartesian = cartesian[:traj_len]
            gripper = _find_gripper(obs, demo, traj_len)

            traj_dir = output_root / f"{h5_path.stem}_{demo_name}"
            mp4_name = f"{camera_key}.mp4"
            mp4_rel = f"recordings/MP4/{mp4_name}"
            mp4_path = traj_dir / mp4_rel
            _write_mp4(frames, mp4_path, output_fps)

            metadata = {metadata_camera_key: mp4_rel}
            with open(traj_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            camera_name = Path(mp4_name).stem
            extrinsics_key = f"{camera_name}_left"
            with h5py.File(traj_dir / "trajectory.h5", "w") as out_h5:
                obs_group = out_h5.create_group("observation")
                robot_group = obs_group.create_group("robot_state")
                camera_group = obs_group.create_group("camera_extrinsics")
                robot_group.create_dataset("cartesian_position", data=cartesian.astype(np.float32))
                robot_group.create_dataset("gripper_position", data=gripper.astype(np.float32))
                camera_group.create_dataset(extrinsics_key, data=np.zeros((traj_len, 6), dtype=np.float32))

            written.append(traj_dir)
    return written


def iter_h5_inputs(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        yield input_path
        return
    for pattern in ("*.hdf5", "*.h5"):
        yield from sorted(input_path.rglob(pattern))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="LIBERO HDF5 file or directory containing HDF5 files")
    parser.add_argument("--output-root", required=True, help="Output directory for DROID-like trajectories")
    parser.add_argument("--csv", required=True, help="Output train_paths.csv")
    parser.add_argument("--camera-key", default="agentview_rgb", help="LIBERO obs camera dataset to export")
    parser.add_argument("--metadata-camera-key", default="left_mp4_path", help="Camera key expected by vjepa_droid config")
    parser.add_argument("--fps", type=int, default=30, help="FPS for written MP4 files")
    parser.add_argument("--max-demos-per-file", type=int, default=None)
    args = parser.parse_args()

    all_trajs = []
    for h5_path in iter_h5_inputs(args.input):
        all_trajs.extend(
            convert_file(
                h5_path=h5_path,
                output_root=args.output_root,
                camera_key=args.camera_key,
                metadata_camera_key=args.metadata_camera_key,
                output_fps=args.fps,
                max_demos=args.max_demos_per_file,
            )
        )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        for traj_dir in all_trajs:
            f.write(str(traj_dir.resolve()) + "\n")

    print(f"Wrote {len(all_trajs)} trajectories")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
