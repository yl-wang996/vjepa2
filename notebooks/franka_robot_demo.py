# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import poses_to_diff
from notebooks.utils.world_model_wrapper import WorldModel
from src.hub.backbones import vjepa2_ac_vit_giant


def clean_backbone_key(state_dict):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


@torch.inference_mode()
def forward_target(encoder, clips, normalize_reps=True):
    bsz, channels, frames, height, width = clips.size()
    clips = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(clips)
    h = h.view(bsz, frames, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


def load_model(checkpoint_path, device):
    print(f"Building V-JEPA 2-AC model on {device}")
    encoder, predictor = vjepa2_ac_vit_giant(pretrained=False)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    encoder_msg = encoder.load_state_dict(clean_backbone_key(checkpoint["encoder"]), strict=False)
    predictor_msg = predictor.load_state_dict(clean_backbone_key(checkpoint["predictor"]), strict=True)
    print(
        "Loaded weights:",
        f"encoder missing/unexpected={len(encoder_msg.missing_keys)}/{len(encoder_msg.unexpected_keys)};",
        f"predictor missing/unexpected={len(predictor_msg.missing_keys)}/{len(predictor_msg.unexpected_keys)}",
    )

    del checkpoint
    gc.collect()
    return encoder.to(device).eval(), predictor.to(device).eval()


def save_frame_strip(np_clips, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np_clips[0]
    num_frames, height, width, channels = frames.shape
    frame_strip = np.transpose(frames, (1, 0, 2, 3)).reshape(height, num_frames * width, channels)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(frame_strip)
    ax.set_axis_off()
    ax.set_title("Franka trajectory frames")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def compute_energy_landscape(
    predictor,
    h,
    states,
    tokens_per_frame,
    nsamples,
    grid_size,
    normalize_reps=True,
):
    def make_action_grid():
        action_samples = []
        for dx in np.linspace(-grid_size, grid_size, nsamples):
            for dy in np.linspace(-grid_size, grid_size, nsamples):
                for dz in np.linspace(-grid_size, grid_size, nsamples):
                    action_samples.append([dx, dy, dz, 0.0, 0.0, 0.0, 0.0])
        return torch.tensor(action_samples, device=h.device, dtype=h.dtype).unsqueeze(1)

    action_samples = make_action_grid()
    z0 = h[:, :tokens_per_frame].repeat(len(action_samples), 1, 1)
    s0 = states[:, :1].to(h.dtype).repeat(len(action_samples), 1, 1)
    z_goal = h[:, -tokens_per_frame:].repeat(len(action_samples), 1, 1)

    z_hat = predictor(z0, action_samples, s0)[:, -tokens_per_frame:]
    if normalize_reps:
        z_hat = F.layer_norm(z_hat, (z_hat.size(-1),))
    loss = torch.mean(torch.abs(z_hat - z_goal), dim=[1, 2])
    return action_samples[:, 0].detach().float().cpu().numpy(), loss.detach().float().cpu().numpy()


def save_energy_landscape(action_grid, energy, gt_action, nsamples, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap, xedges, zedges = np.histogram2d(action_grid[:, 0], action_grid[:, 2], weights=energy, bins=nsamples)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]],
        cmap="viridis",
        aspect="auto",
    )
    ax.scatter([gt_action[0]], [gt_action[2]], c="red", marker="x", s=80, label="ground truth")
    ax.set_xlabel("Action delta x")
    ax.set_ylabel("Action delta z")
    ax.set_title("V-JEPA 2-AC energy landscape")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="prediction energy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run the V-JEPA 2-AC Franka trajectory demo with a local checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/vjepa2-ac-vitg.pt")
    parser.add_argument("--trajectory", default="notebooks/franka_example_traj.npz")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cem-samples", type=int, default=8)
    parser.add_argument("--cem-steps", type=int, default=1)
    parser.add_argument("--cem-topk", type=int, default=4)
    parser.add_argument("--skip-cem", action="store_true")
    parser.add_argument("--save-viz", default=None, help="optional output directory for PNG visualizations")
    parser.add_argument("--energy-nsamples", type=int, default=5)
    parser.add_argument("--energy-grid-size", type=float, default=0.075)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, predictor = load_model(args.checkpoint, device)

    crop_size = 256
    tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    trajectory = np.load(args.trajectory)
    np_clips = trajectory["observations"]
    np_states = trajectory["states"]
    np_actions = np.expand_dims(poses_to_diff(np_states[0, 0], np_states[0, 1]), axis=(0, 1))

    clips = transform(np_clips[0]).unsqueeze(0).to(device)
    states = torch.tensor(np_states, device=device)
    actions = torch.tensor(np_actions, device=device, dtype=clips.dtype)
    print(
        "Loaded trajectory:",
        f"clips={tuple(clips.shape)};",
        f"states={tuple(states.shape)};",
        f"gt_action_xyz_grip={actions[0, 0, [0, 1, 2, 6]].detach().cpu().numpy().round(4).tolist()}",
    )

    use_cuda_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        h = forward_target(encoder, clips)
        z0 = h[:, :tokens_per_frame]
        z_goal = h[:, -tokens_per_frame:]
        s0 = states[:, :1].to(h.dtype)
        pred_next = predictor(z0, actions.to(h.dtype), s0)[:, -tokens_per_frame:]
        pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
        energy = torch.mean(torch.abs(pred_next - z_goal), dim=[1, 2])

    print(
        "One-step inference:",
        f"repr={tuple(h.shape)};",
        f"pred_next={tuple(pred_next.shape)};",
        f"energy={energy.detach().float().cpu().numpy().round(6).tolist()}",
    )

    if args.save_viz is not None:
        output_dir = Path(args.save_viz)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_path = output_dir / "trajectory_frames.png"
        energy_path = output_dir / "energy_landscape.png"
        save_frame_strip(np_clips, frames_path)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
            action_grid, grid_energy = compute_energy_landscape(
                predictor=predictor,
                h=h,
                states=states,
                tokens_per_frame=tokens_per_frame,
                nsamples=args.energy_nsamples,
                grid_size=args.energy_grid_size,
            )
        save_energy_landscape(action_grid, grid_energy, np_actions[0, 0], args.energy_nsamples, energy_path)
        print("Saved visualizations:", str(frames_path), str(energy_path))

    if args.skip_cem:
        return

    world_model = WorldModel(
        encoder=encoder,
        predictor=predictor,
        tokens_per_frame=tokens_per_frame,
        transform=transform,
        mpc_args={
            "rollout": 2,
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

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp):
        planned = world_model.infer_next_action(
            h[:, :tokens_per_frame].unsqueeze(1),
            states[:, :1].to(h.dtype),
            h[:, -tokens_per_frame:].unsqueeze(1),
        )

    print(
        "CEM planning:",
        f"planned_shape={tuple(planned.shape)};",
        f"first_action_xyz_grip={planned[0, [0, 1, 2, 6]].detach().float().cpu().numpy().round(4).tolist()}",
    )


if __name__ == "__main__":
    main()
