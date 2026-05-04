#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize a LIBERO HDF5 episode with Rerun.

Examples:
  .venv/bin/python scripts/visualize_libero_rerun.py \
    --task on_the_stove \
    --episode-id 0 \
    --spawn

  .venv/bin/rerun outputs/rerun/libero_on_the_stove_demo_0.rrd
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation


DEFAULT_CSV = Path("data/libero/libero_spatial_h5_paths.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/rerun")
DEFAULT_CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


@dataclass(frozen=True)
class DemoData:
    task: str
    h5_path: Path
    demo_key: str
    cameras: dict[str, np.ndarray]
    ee_states: np.ndarray
    gripper: np.ndarray
    actions: np.ndarray | None
    rewards: np.ndarray | None
    dones: np.ndarray | None


def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def task_name_from_path(path: Path) -> str:
    name = path.stem
    return name[: -len("_demo")] if name.endswith("_demo") else name


def read_h5_paths(csv_path: Path) -> list[Path]:
    paths = []
    for line in csv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line.split()[0]).expanduser())
    if not paths:
        raise ValueError(f"No HDF5 paths found in {csv_path}")
    return paths


def resolve_task(task: str, h5_paths: list[Path]) -> Path:
    if task.isdigit():
        index = int(task)
        if index < 0 or index >= len(h5_paths):
            raise IndexError(f"Task index {index} out of range 0..{len(h5_paths) - 1}")
        return h5_paths[index]

    task_path = Path(task).expanduser()
    if task_path.exists():
        return task_path

    query = task.lower().replace(" ", "_")
    matches = [path for path in h5_paths if query in task_name_from_path(path).lower()]
    if not matches:
        available = "\n".join(f"  {i}: {task_name_from_path(path)}" for i, path in enumerate(h5_paths))
        raise ValueError(f"No task matched '{task}'. Available tasks:\n{available}")
    if len(matches) > 1:
        available = "\n".join(f"  {task_name_from_path(path)}" for path in matches)
        raise ValueError(f"Task '{task}' is ambiguous. Matches:\n{available}")
    return matches[0]


def normalize_demo_key(episode_id: str) -> str:
    return f"demo_{episode_id}" if episode_id.isdigit() else episode_id


def as_uint8_rgb(frames: h5py.Dataset | np.ndarray, flip_vertical: bool) -> np.ndarray:
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
    if flip_vertical:
        frames = np.flip(frames, axis=1).copy()
    return frames


def obs_group(demo: h5py.Group) -> h5py.Group:
    if "obs" in demo:
        return demo["obs"]
    if "observations" in demo:
        return demo["observations"]
    raise KeyError("Demo must contain obs or observations group")


def read_ee_states(obs: h5py.Group) -> np.ndarray:
    if "ee_states" in obs:
        ee_states = np.asarray(obs["ee_states"])
        if ee_states.shape[-1] < 6:
            raise ValueError(f"obs/ee_states must have at least 6 dims, got {ee_states.shape}")
        return ee_states[:, :6].astype(np.float32)
    if "ee_pos" in obs and "ee_ori" in obs:
        return np.concatenate([np.asarray(obs["ee_pos"]), np.asarray(obs["ee_ori"])], axis=-1).astype(np.float32)
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        euler = Rotation.from_quat(np.asarray(obs["robot0_eef_quat"])).as_euler("xyz", degrees=False)
        return np.concatenate([np.asarray(obs["robot0_eef_pos"]), euler], axis=-1).astype(np.float32)
    raise KeyError("Could not infer ee pose; expected ee_states, ee_pos+ee_ori, or robot0_eef_pos+quat")


def read_gripper(obs: h5py.Group, demo: h5py.Group, target_len: int) -> np.ndarray:
    if "gripper_states" in obs:
        gripper = np.asarray(obs["gripper_states"])
        if gripper.ndim == 2:
            gripper = gripper.mean(axis=-1)
        return gripper[:target_len].astype(np.float32)
    if "actions" in demo and demo["actions"].shape[-1] >= 1:
        return np.asarray(demo["actions"])[:target_len, -1].astype(np.float32)
    return np.zeros((target_len,), dtype=np.float32)


def load_demo(
    h5_path: Path,
    demo_key: str,
    camera_keys: list[str],
    flip_vertical: bool,
    max_frames: int | None,
) -> DemoData:
    with h5py.File(h5_path, "r") as h5_file:
        root = h5_file["data"] if "data" in h5_file else h5_file
        if demo_key not in root:
            available = ", ".join(sorted(root.keys(), key=natural_key)[:12])
            raise KeyError(f"{demo_key} not found in {h5_path}. First demos: {available}")

        demo = root[demo_key]
        obs = obs_group(demo)
        cameras = {
            key: as_uint8_rgb(obs[key], flip_vertical=flip_vertical)
            for key in camera_keys
            if key in obs
        }
        if not cameras:
            available = ", ".join(key for key in obs.keys() if key.endswith("rgb"))
            raise KeyError(f"None of the requested cameras exist. Requested={camera_keys}; available={available}")

        ee_states = read_ee_states(obs)
        length = min([len(ee_states), *(len(frames) for frames in cameras.values())])
        if max_frames is not None:
            length = min(length, max_frames)

        cameras = {key: frames[:length] for key, frames in cameras.items()}
        ee_states = ee_states[:length]
        gripper = read_gripper(obs, demo, length)
        actions = np.asarray(demo["actions"])[:length].astype(np.float32) if "actions" in demo else None
        rewards = np.asarray(demo["rewards"])[:length].astype(np.float32) if "rewards" in demo else None
        dones = np.asarray(demo["dones"])[:length].astype(np.float32) if "dones" in demo else None

    return DemoData(
        task=task_name_from_path(h5_path),
        h5_path=h5_path,
        demo_key=demo_key,
        cameras=cameras,
        ee_states=ee_states,
        gripper=gripper,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )


def safe_output_name(task: str, demo_key: str) -> str:
    task_slug = re.sub(r"[^a-zA-Z0-9]+", "_", task).strip("_")
    return f"libero_{task_slug}_{demo_key}.rrd"


def log_series_styles(data: DemoData) -> None:
    rr.log(
        "plots/ee_xyz",
        rr.SeriesLines(
            names=["x", "y", "z"],
            colors=[[230, 66, 66], [70, 160, 90], [70, 110, 220]],
        ),
        static=True,
    )
    rr.log(
        "plots/ee_rpy",
        rr.SeriesLines(
            names=["roll", "pitch", "yaw"],
            colors=[[220, 120, 45], [140, 90, 220], [30, 160, 170]],
        ),
        static=True,
    )
    rr.log("plots/gripper", rr.SeriesLines(names=["gripper"], colors=[[245, 180, 60]]), static=True)
    if data.actions is not None:
        names = [f"a{i}" for i in range(data.actions.shape[-1])]
        rr.log("plots/actions", rr.SeriesLines(names=names), static=True)
    if data.rewards is not None:
        rr.log("plots/reward", rr.SeriesLines(names=["reward"], colors=[[70, 170, 230]]), static=True)
    if data.dones is not None:
        rr.log("plots/done", rr.SeriesLines(names=["done"], colors=[[230, 90, 180]]), static=True)


def log_static_scene(data: DemoData, fps: float) -> None:
    summary = "\n".join(
        [
            f"# LIBERO episode",
            f"- task: `{data.task}`",
            f"- h5: `{data.h5_path}`",
            f"- episode: `{data.demo_key}`",
            f"- frames: `{len(data.ee_states)}` at `{fps:g}` fps",
            f"- cameras: `{', '.join(data.cameras.keys())}`",
            "- RGB frames are vertically flipped to match the repo's LIBERO loader.",
        ]
    )
    rr.log("episode/summary", rr.TextDocument(summary, media_type="text/markdown"), static=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    xyz = data.ee_states[:, :3]
    rr.log("world/ee_trajectory", rr.LineStrips3D([xyz], radii=0.004, colors=[[50, 140, 240]]), static=True)
    rr.log(
        "world/trajectory_endpoints",
        rr.Points3D(
            [xyz[0], xyz[-1]],
            radii=[0.015, 0.015],
            colors=[[50, 180, 80], [230, 80, 80]],
            labels=["start", "end"],
            show_labels=True,
        ),
        static=True,
    )
    log_series_styles(data)


def log_frame(data: DemoData, index: int, fps: float, axis_scale: float) -> None:
    rr.set_time("frame", sequence=index)
    rr.set_time("time", duration=index / fps)

    for camera_name, frames in data.cameras.items():
        rr.log(f"video/{camera_name}", rr.Image(frames[index]))

    pose = data.ee_states[index]
    xyz = pose[:3]
    rot = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    rr.log("world/ee_current", rr.Points3D([xyz], radii=0.012, colors=[[255, 210, 60]], labels=[str(index)]))
    rr.log("world/ee_pose", rr.Transform3D(translation=xyz, mat3x3=rot))
    rr.log(
        "world/ee_axes",
        rr.Arrows3D(
            origins=[xyz, xyz, xyz],
            vectors=[rot[:, 0] * axis_scale, rot[:, 1] * axis_scale, rot[:, 2] * axis_scale],
            colors=[[230, 66, 66], [70, 160, 90], [70, 110, 220]],
        ),
    )

    rr.log("plots/ee_xyz", rr.Scalars(pose[:3]))
    rr.log("plots/ee_rpy", rr.Scalars(pose[3:6]))
    rr.log("plots/gripper", rr.Scalars([data.gripper[index]]))
    if data.actions is not None:
        rr.log("plots/actions", rr.Scalars(data.actions[index]))
    if data.rewards is not None:
        rr.log("plots/reward", rr.Scalars([data.rewards[index]]))
    if data.dones is not None:
        rr.log("plots/done", rr.Scalars([data.dones[index]]))


def write_rerun(data: DemoData, output_path: Path, fps: float, axis_scale: float, spawn: bool) -> None:
    rr.init("libero_spatial_episode", recording_id=f"{data.task}/{data.demo_key}", spawn=False)
    rr.save(output_path)
    log_static_scene(data, fps=fps)
    for index in range(len(data.ee_states)):
        log_frame(data, index=index, fps=fps, axis_scale=axis_scale)
    if spawn:
        spawn_rerun_viewer(output_path)


def rerun_executable_path() -> str | None:
    """Prefer the Rerun viewer installed next to the active Python executable."""
    executable = Path(sys.executable)
    candidates = [executable.with_name("rerun"), executable.with_name("rerun.exe")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("rerun")


def spawn_rerun_viewer(output_path: Path) -> None:
    executable = rerun_executable_path()
    if executable is None:
        raise RuntimeError("Could not find the Rerun viewer executable. Try `.venv/bin/rerun <file.rrd>`.")
    subprocess.Popen(
        [executable, str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def list_tasks(csv_path: Path) -> None:
    for index, path in enumerate(read_h5_paths(csv_path)):
        print(f"{index:02d}  {task_name_from_path(path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV containing LIBERO HDF5 paths")
    parser.add_argument("--task", help="Task index, full HDF5 path, or unique substring of the task filename")
    parser.add_argument("--episode-id", default="0", help="Episode id, e.g. 0 or demo_0")
    parser.add_argument("--camera-key", action="append", dest="camera_keys", help="Camera key to log; can repeat")
    parser.add_argument("--output", type=Path, help="Output .rrd path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for default .rrd output")
    parser.add_argument("--fps", type=float, default=20.0, help="Timeline FPS used by Rerun")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick visualization")
    parser.add_argument("--axis-scale", type=float, default=0.04, help="Scale for the end-effector orientation axes")
    parser.add_argument("--no-flip", action="store_true", help="Disable the vertical RGB flip used by the repo loader")
    parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer after writing")
    parser.add_argument("--list-tasks", action="store_true", help="Print available tasks and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        list_tasks(args.csv)
        return
    if not args.task:
        raise SystemExit("--task is required unless --list-tasks is used")

    h5_paths = read_h5_paths(args.csv)
    h5_path = resolve_task(args.task, h5_paths)
    demo_key = normalize_demo_key(args.episode_id)
    camera_keys = args.camera_keys or list(DEFAULT_CAMERAS)
    data = load_demo(
        h5_path=h5_path,
        demo_key=demo_key,
        camera_keys=camera_keys,
        flip_vertical=not args.no_flip,
        max_frames=args.max_frames,
    )

    output_path = args.output or args.output_dir / safe_output_name(data.task, data.demo_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rerun(data, output_path=output_path, fps=args.fps, axis_scale=args.axis_scale, spawn=args.spawn)
    print(f"Wrote {output_path}")
    print(f"Task: {data.task}")
    print(f"Episode: {data.demo_key}")
    print(f"Frames: {len(data.ee_states)}")
    print(f"Cameras: {', '.join(data.cameras.keys())}")


if __name__ == "__main__":
    main()
