#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""Validate a LIBERO-trained ViT-B AC predictor and render an action energy landscape."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_droid.eval_libero import (
    compute_dx_dz_landscape,
    forward_target,
    load_eval_model,
    load_yaml,
    save_energy_landscape,
    save_frame_strip,
)
from app.vjepa_droid.libero import LIBEROHDF5Dataset
from app.vjepa_droid.transforms import make_transforms


@torch.inference_mode()
def compute_one_step_energy(predictor, h, actions, states, tokens_per_frame, goal_step, normalize_reps=True):
    z0 = h[:, :tokens_per_frame]
    z_goal = h[:, goal_step * tokens_per_frame : (goal_step + 1) * tokens_per_frame]
    pred_next = predictor(z0, actions[:, :1].to(h.dtype), states[:, :1].to(h.dtype))[:, -tokens_per_frame:]
    if normalize_reps:
        pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
    energy = torch.mean(torch.abs(pred_next - z_goal), dim=[1, 2])
    return energy, z0, z_goal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/vitb16/libero-256px-8f-debug.yaml")
    parser.add_argument("--checkpoint", default="outputs/train/libero-vjepa21-vitb-ac-debug/latest.pt")
    parser.add_argument("--output-dir", default="outputs/validation/libero-vjepa21-vitb-ac-debug")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-step", type=int, default=1)
    parser.add_argument("--energy-nsamples", type=int, default=21)
    parser.add_argument("--energy-grid-size", type=float, default=0.05)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml(args.config)
    data_cfg = config["data"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LIBEROHDF5Dataset(
        data_path=data_cfg["datasets"][0],
        camera_key=data_cfg.get("libero_camera_key", "agentview_rgb"),
        frames_per_clip=max(data_cfg["dataset_fpcs"]),
        frame_stride=data_cfg.get("libero_frame_stride", 1),
        transform=None,
    )
    frames, actions_np, states_np, _, indices = dataset[args.sample_index]
    h5_path, demo_key = dataset.samples[args.sample_index]

    if args.goal_step < 1 or args.goal_step >= len(frames):
        raise ValueError(f"--goal-step must be in [1, {len(frames) - 1}], got {args.goal_step}")

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=data_cfg.get("random_resize_aspect_ratio", (1.0, 1.0)),
        random_resize_scale=data_cfg.get("random_resize_scale", (1.0, 1.0)),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg["crop_size"],
    )
    clips = transform(frames).unsqueeze(0)
    states = torch.tensor(states_np, dtype=torch.float32).unsqueeze(0)
    actions = torch.tensor(actions_np, dtype=torch.float32).unsqueeze(0)

    device = torch.device(args.device)
    encoder, predictor = load_eval_model(config, args.checkpoint, device)
    clips = clips.to(device)
    states = states.to(device)
    actions = actions.to(device)

    tokens_per_frame = int((data_cfg["crop_size"] // data_cfg["patch_size"]) ** 2)
    use_cuda_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        h = forward_target(encoder, clips)
        gt_energy, z0, z_goal = compute_one_step_energy(
            predictor=predictor,
            h=h,
            actions=actions,
            states=states,
            tokens_per_frame=tokens_per_frame,
            goal_step=args.goal_step,
        )
        action_grid, grid_energy = compute_dx_dz_landscape(
            predictor=predictor,
            z0=z0,
            z_goal=z_goal,
            states=states,
            gt_action=actions_np[0],
            nsamples=args.energy_nsamples,
            grid_size=args.energy_grid_size,
        )

    frames_path = output_dir / "libero_clip_frames.png"
    landscape_path = output_dir / "energy_landscape_dx_dz.png"
    metrics_path = output_dir / "metrics.json"
    save_frame_strip(frames, frames_path)
    save_energy_landscape(
        action_grid,
        grid_energy,
        actions_np[0],
        args.energy_nsamples,
        landscape_path,
        title="LIBERO ViT-B AC energy landscape",
    )

    best_index = int(np.argmin(grid_energy))
    center_index = (args.energy_nsamples // 2) * args.energy_nsamples + (args.energy_nsamples // 2)
    gt_energy_value = float(gt_energy.detach().float().cpu().item())
    center_grid_energy = float(grid_energy[center_index])
    metrics = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "h5_path": str(h5_path),
        "demo_key": demo_key,
        "sample_index": args.sample_index,
        "frame_indices": indices.tolist(),
        "goal_step": args.goal_step,
        "gt_action": actions_np[0].astype(float).tolist(),
        "gt_energy": gt_energy_value,
        "center_grid_action": action_grid[center_index].astype(float).tolist(),
        "center_grid_energy": center_grid_energy,
        "best_grid_action": action_grid[best_index].astype(float).tolist(),
        "best_grid_energy": float(grid_energy[best_index]),
        "grid_actions_better_than_center": int(np.sum(grid_energy < center_grid_energy - 1.0e-7)),
        "grid_size": args.energy_grid_size,
        "energy_nsamples": args.energy_nsamples,
    }
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("Validation sample:", f"{h5_path}:{demo_key}", f"indices={indices.tolist()}")
    print("Latents:", f"h={tuple(h.shape)}", f"tokens_per_frame={tokens_per_frame}")
    print(
        "Energy:",
        f"gt={metrics['gt_energy']:.6f};",
        f"center_grid={metrics['center_grid_energy']:.6f};",
        f"best_grid={metrics['best_grid_energy']:.6f};",
        f"better_than_center={metrics['grid_actions_better_than_center']}/{len(grid_energy)}",
    )
    print("Saved:", str(frames_path), str(landscape_path), str(metrics_path))


if __name__ == "__main__":
    main()
