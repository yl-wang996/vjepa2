#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_droid.eval_libero import (
    compute_dx_dz_landscape,
    deterministic_sample_seed,
    ensure_output_dir,
    forward_target,
    load_eval_model,
    load_yaml,
    rollout_autoregressive,
    save_energy_landscape,
    save_frame_strip,
)
from app.vjepa_droid.libero import LIBEROHDF5Dataset
from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.world_model_wrapper import WorldModel


def parse_goal_steps(text):
    goal_steps = []
    for value in text.split(","):
        value = value.strip()
        if value:
            goal_steps.append(int(value))
    if not goal_steps:
        raise ValueError("Expected at least one goal step")
    return sorted(set(goal_steps))


def build_sample_indices(dataset_len, mode, max_samples, offset, stride, rng):
    if mode == "sequential":
        indices = list(range(offset, dataset_len, stride))
    elif mode == "tail":
        start = max(0, dataset_len - offset - max_samples * stride)
        indices = list(range(start, dataset_len - offset, stride))
    elif mode == "random":
        indices = rng.permutation(dataset_len).tolist()
    else:
        raise ValueError(f"Unsupported sample mode: {mode}")
    return indices[:max_samples]


def get_sample(dataset, sample_index, seed):
    h5_path, demo_key = dataset.samples[sample_index]
    rng = np.random.default_rng(deterministic_sample_seed(seed, sample_index))
    frames, actions, states, extrinsics, frame_indices = dataset.load_demo(h5_path, demo_key, rng=rng)
    return frames, actions, states, extrinsics, frame_indices, h5_path, demo_key


def frame_latent(h, step, tokens_per_frame):
    start = step * tokens_per_frame
    end = (step + 1) * tokens_per_frame
    return h[:, start:end]


def l1_energy(pred, target):
    return torch.mean(torch.abs(pred - target), dim=[1, 2])


def make_negative_candidates(gt_action, num_negatives, rng, xyz_sigma, rot_sigma, grip_sigma):
    candidates = np.repeat(gt_action[None, :], num_negatives + 1, axis=0)
    if num_negatives > 0:
        noise = np.zeros((num_negatives, gt_action.shape[0]), dtype=np.float32)
        noise[:, :3] = rng.normal(loc=0.0, scale=xyz_sigma, size=(num_negatives, 3))
        noise[:, 3:6] = rng.normal(loc=0.0, scale=rot_sigma, size=(num_negatives, 3))
        noise[:, 6] = rng.normal(loc=0.0, scale=grip_sigma, size=(num_negatives,))
        candidates[1:] += noise
    candidates[:, :3] = np.clip(candidates[:, :3], gt_action[:3] - 0.08, gt_action[:3] + 0.08)
    candidates[:, 6] = np.clip(candidates[:, 6], -1.0, 1.0)
    return candidates.astype(np.float32)


def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1.0e-8:
        return 1.0
    return float(np.dot(a, b) / denom)


def summarize_metric(rows, key):
    values = [row[key] for row in rows if key in row and row[key] is not None and not math.isnan(row[key])]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch offline evaluation for a LIBERO-trained ViT-B AC predictor.")
    parser.add_argument("--config", default="configs/train/vitb16/libero-256px-8f-main.yaml")
    parser.add_argument("--checkpoint", default="outputs/train/libero-vjepa21-vitb-ac-main/latest.pt")
    parser.add_argument("--output-dir", default="outputs/eval/libero-vjepa21-vitb-ac-main")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-mode", choices=["sequential", "tail", "random"], default="tail")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--goal-steps", default="1,2,4,7")
    parser.add_argument("--num-negatives", type=int, default=32)
    parser.add_argument("--negative-xyz-sigma", type=float, default=0.02)
    parser.add_argument("--negative-rot-sigma", type=float, default=0.10)
    parser.add_argument("--negative-grip-sigma", type=float, default=0.20)
    parser.add_argument("--skip-cem", action="store_true")
    parser.add_argument("--cem-goal-step", type=int, default=2)
    parser.add_argument("--cem-rollout", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=64)
    parser.add_argument("--cem-topk", type=int, default=8)
    parser.add_argument("--cem-steps", type=int, default=4)
    parser.add_argument("--save-landscape-count", type=int, default=6)
    parser.add_argument("--landscape-goal-step", type=int, default=1)
    parser.add_argument("--energy-nsamples", type=int, default=21)
    parser.add_argument("--energy-grid-size", type=float, default=0.05)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml(args.config)
    data_cfg = config["data"]
    output_dir = ensure_output_dir(args.output_dir)
    examples_dir = ensure_output_dir(output_dir / "examples")
    goal_steps = parse_goal_steps(args.goal_steps)

    dataset = LIBEROHDF5Dataset(
        data_path=data_cfg["datasets"][0],
        camera_key=data_cfg.get("libero_camera_key", "agentview_rgb"),
        frames_per_clip=max(data_cfg["dataset_fpcs"]),
        frame_stride=data_cfg.get("libero_frame_stride", 1),
        transform=None,
    )
    sample_rng = np.random.default_rng(args.seed)
    sample_indices = build_sample_indices(
        dataset_len=len(dataset),
        mode=args.sample_mode,
        max_samples=args.max_samples,
        offset=args.sample_offset,
        stride=args.sample_stride,
        rng=sample_rng,
    )

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=data_cfg.get("random_resize_aspect_ratio", (1.0, 1.0)),
        random_resize_scale=data_cfg.get("random_resize_scale", (1.0, 1.0)),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=data_cfg["crop_size"],
    )

    device = torch.device(args.device)
    encoder, predictor = load_eval_model(config, args.checkpoint, device)
    tokens_per_frame = int((data_cfg["crop_size"] // data_cfg["patch_size"]) ** 2)
    use_cuda_amp = device.type == "cuda"

    world_model = None
    if not args.skip_cem:
        world_model = WorldModel(
            encoder=encoder,
            predictor=predictor,
            tokens_per_frame=tokens_per_frame,
            transform=transform,
            mpc_args={
                "rollout": args.cem_rollout,
                "samples": args.cem_samples,
                "topk": args.cem_topk,
                "cem_steps": args.cem_steps,
                "momentum_mean": 0.15,
                "momentum_mean_gripper": 0.15,
                "momentum_std": 0.75,
                "momentum_std_gripper": 0.15,
                "maxnorm": 0.075,
                "verbose": False,
            },
            normalize_reps=True,
            device=str(device),
        )

    rows = []
    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset_csv": data_cfg["datasets"][0],
        "dataset_size": len(dataset),
        "sample_indices": sample_indices,
        "sample_mode": args.sample_mode,
        "max_samples": args.max_samples,
        "goal_steps": goal_steps,
        "cem_enabled": not args.skip_cem,
    }

    for example_rank, sample_index in enumerate(tqdm(sample_indices, desc="Evaluating")):
        frames, actions_np, states_np, extrinsics_np, frame_indices, h5_path, demo_key = get_sample(
            dataset, sample_index, args.seed
        )
        clips = transform(frames).unsqueeze(0).to(device)
        states = torch.tensor(states_np, dtype=torch.float32, device=device).unsqueeze(0)
        actions = torch.tensor(actions_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
            h = forward_target(encoder, clips)

            sample_row = {
                "sample_index": int(sample_index),
                "h5_path": str(h5_path),
                "demo_key": demo_key,
                "frame_indices": "-".join(str(int(v)) for v in frame_indices.tolist()),
            }

            num_transitions = len(frames) - 1
            for step in range(1, num_transitions + 1):
                z_prev = frame_latent(h, step - 1, tokens_per_frame)
                z_goal = frame_latent(h, step, tokens_per_frame)
                pred = predictor(
                    z_prev,
                    actions[:, step - 1 : step].to(h.dtype),
                    states[:, step - 1 : step].to(h.dtype),
                )[:, -tokens_per_frame:]
                pred = F.layer_norm(pred, (pred.size(-1),))
                sample_row[f"one_step_loss@{step}"] = float(l1_energy(pred, z_goal).detach().float().cpu().item())

            max_rollout_step = min(max(goal_steps), num_transitions)
            rollout_predictions = rollout_autoregressive(
                predictor=predictor,
                z0=frame_latent(h, 0, tokens_per_frame),
                actions=actions,
                states=states,
                steps=max_rollout_step,
            )
            for goal_step in goal_steps:
                if goal_step > num_transitions:
                    sample_row[f"rollout_loss@{goal_step}"] = None
                    continue
                z_goal = frame_latent(h, goal_step, tokens_per_frame)
                sample_row[f"rollout_loss@{goal_step}"] = float(
                    l1_energy(rollout_predictions[goal_step - 1], z_goal).detach().float().cpu().item()
                )

            gt_action = actions_np[0].astype(np.float32)
            z0 = frame_latent(h, 0, tokens_per_frame)
            z1 = frame_latent(h, 1, tokens_per_frame)
            retrieval_rng = np.random.default_rng(deterministic_sample_seed(args.seed + 17, sample_index))
            action_candidates = make_negative_candidates(
                gt_action=gt_action,
                num_negatives=args.num_negatives,
                rng=retrieval_rng,
                xyz_sigma=args.negative_xyz_sigma,
                rot_sigma=args.negative_rot_sigma,
                grip_sigma=args.negative_grip_sigma,
            )
            action_candidates_t = torch.tensor(action_candidates, device=device, dtype=z0.dtype).unsqueeze(1)
            z0_grid = z0.repeat(action_candidates_t.size(0), 1, 1)
            z1_grid = z1.repeat(action_candidates_t.size(0), 1, 1)
            states_grid = states[:, :1].to(z0.dtype).repeat(action_candidates_t.size(0), 1, 1)
            pred_next = predictor(z0_grid, action_candidates_t, states_grid)[:, -tokens_per_frame:]
            pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
            candidate_energy = l1_energy(pred_next, z1_grid).detach().float().cpu().numpy()
            gt_energy = float(candidate_energy[0])
            gt_rank = int(1 + np.sum(candidate_energy[1:] < gt_energy - 1.0e-7))
            negative_energy = candidate_energy[1:]
            sample_row["gt_energy"] = gt_energy
            sample_row["retrieval_gt_rank"] = gt_rank
            sample_row["retrieval_top1"] = int(gt_rank == 1)
            sample_row["retrieval_top5"] = int(gt_rank <= min(5, len(candidate_energy)))
            sample_row["retrieval_energy_margin"] = float(np.mean(negative_energy) - gt_energy)
            sample_row["retrieval_negative_energy_mean"] = float(np.mean(negative_energy))

            cem_goal_step = min(max(1, args.cem_goal_step), num_transitions)
            if world_model is not None and cem_goal_step <= num_transitions:
                planned_actions = world_model.infer_next_action(
                    z0.unsqueeze(1),
                    states[:, :1].to(h.dtype),
                    frame_latent(h, cem_goal_step, tokens_per_frame).unsqueeze(1),
                )
                planned_np = planned_actions.detach().float().cpu().numpy()
                gt_traj = actions_np[: planned_np.shape[0]].astype(np.float32)
                sample_row["cem_goal_step"] = cem_goal_step
                sample_row["cem_first_action_l2"] = float(np.linalg.norm(planned_np[0] - gt_traj[0]))
                sample_row["cem_first_action_xyz_l2"] = float(np.linalg.norm(planned_np[0, :3] - gt_traj[0, :3]))
                sample_row["cem_first_action_xyz_cosine"] = cosine_similarity(planned_np[0, :3], gt_traj[0, :3])
                sample_row["cem_rollout_l2"] = float(np.linalg.norm(planned_np - gt_traj))
            else:
                sample_row["cem_goal_step"] = cem_goal_step
                sample_row["cem_first_action_l2"] = None
                sample_row["cem_first_action_xyz_l2"] = None
                sample_row["cem_first_action_xyz_cosine"] = None
                sample_row["cem_rollout_l2"] = None

            if example_rank < args.save_landscape_count and args.landscape_goal_step <= num_transitions:
                z_goal = frame_latent(h, args.landscape_goal_step, tokens_per_frame)
                action_grid, grid_energy = compute_dx_dz_landscape(
                    predictor=predictor,
                    z0=z0,
                    z_goal=z_goal,
                    states=states,
                    gt_action=gt_action,
                    nsamples=args.energy_nsamples,
                    grid_size=args.energy_grid_size,
                )
                example_prefix = f"{example_rank:02d}_sample{sample_index}"
                frames_path = examples_dir / f"{example_prefix}_clip.png"
                landscape_path = examples_dir / f"{example_prefix}_energy.png"
                save_frame_strip(frames, frames_path)
                save_energy_landscape(
                    action_grid=action_grid,
                    energy=grid_energy,
                    gt_action=gt_action,
                    nsamples=args.energy_nsamples,
                    output_path=landscape_path,
                    title=f"Sample {sample_index} goal_step={args.landscape_goal_step}",
                )
                sample_row["example_frames"] = str(frames_path)
                sample_row["example_energy"] = str(landscape_path)
                sample_row["example_best_grid_energy"] = float(np.min(grid_energy))
                sample_row["example_grid_actions_better_than_gt"] = int(np.sum(grid_energy < gt_energy - 1.0e-7))

        rows.append(sample_row)

    metrics_df = pd.DataFrame(rows)
    metrics_csv = output_dir / "per_sample_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    summary["per_sample_metrics_csv"] = str(metrics_csv)
    summary["metrics"] = {}
    for step in range(1, max(goal_steps) + 1):
        key = f"one_step_loss@{step}"
        metric = summarize_metric(rows, key)
        if metric is not None:
            summary["metrics"][key] = metric
    for goal_step in goal_steps:
        key = f"rollout_loss@{goal_step}"
        metric = summarize_metric(rows, key)
        if metric is not None:
            summary["metrics"][key] = metric
    for key in [
        "gt_energy",
        "retrieval_gt_rank",
        "retrieval_top1",
        "retrieval_top5",
        "retrieval_energy_margin",
        "retrieval_negative_energy_mean",
        "cem_first_action_l2",
        "cem_first_action_xyz_l2",
        "cem_first_action_xyz_cosine",
        "cem_rollout_l2",
        "example_best_grid_energy",
        "example_grid_actions_better_than_gt",
    ]:
        metric = summarize_metric(rows, key)
        if metric is not None:
            summary["metrics"][key] = metric

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("Evaluated samples:", len(rows))
    print("Saved:", metrics_csv, summary_path)
    if "retrieval_top1" in summary["metrics"]:
        print("retrieval_top1:", f"{summary['metrics']['retrieval_top1']['mean']:.4f}")
    if "cem_first_action_xyz_l2" in summary["metrics"]:
        print("cem_first_action_xyz_l2:", f"{summary['metrics']['cem_first_action_xyz_l2']['mean']:.6f}")


if __name__ == "__main__":
    main()
