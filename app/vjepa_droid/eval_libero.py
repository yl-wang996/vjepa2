from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from app.vjepa_droid.utils import init_video_model


def clean_module_key(state_dict):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def load_yaml(path):
    with open(path, "r") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def load_eval_model(config, checkpoint_path, device):
    data_cfg = config["data"]
    model_cfg = config["model"]
    max_num_frames = data_cfg.get("eval_max_num_frames", 512)

    encoder, predictor = init_video_model(
        device=device,
        patch_size=data_cfg["patch_size"],
        max_num_frames=max_num_frames,
        tubelet_size=data_cfg["tubelet_size"],
        model_name=model_cfg["model_name"],
        crop_size=data_cfg["crop_size"],
        pred_depth=model_cfg["pred_depth"],
        pred_num_heads=model_cfg.get("pred_num_heads"),
        pred_embed_dim=model_cfg["pred_embed_dim"],
        pred_is_frame_causal=model_cfg.get("pred_is_frame_causal", True),
        uniform_power=model_cfg.get("uniform_power", False),
        use_sdpa=config["meta"].get("use_sdpa", False),
        use_rope=model_cfg.get("use_rope", False),
        use_silu=model_cfg.get("use_silu", False),
        use_pred_silu=model_cfg.get("use_pred_silu", False),
        wide_silu=model_cfg.get("wide_silu", True),
        use_activation_checkpointing=False,
        use_extrinsics=model_cfg.get("use_extrinsics", False),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = checkpoint.get("target_encoder", checkpoint.get("encoder"))
    encoder_msg = encoder.load_state_dict(clean_module_key(encoder_state), strict=False)
    predictor_msg = predictor.load_state_dict(clean_module_key(checkpoint["predictor"]), strict=True)
    print(
        "Loaded checkpoint:",
        f"path={checkpoint_path};",
        f"epoch={checkpoint.get('epoch')};",
        f"loss={checkpoint.get('loss')};",
        f"encoder missing/unexpected={len(encoder_msg.missing_keys)}/{len(encoder_msg.unexpected_keys)};",
        f"predictor missing/unexpected={len(predictor_msg.missing_keys)}/{len(predictor_msg.unexpected_keys)}",
    )
    encoder.eval()
    predictor.eval()
    return encoder, predictor


@torch.inference_mode()
def forward_target(encoder, clips, normalize_reps=True):
    batch_size, channels, frames, height, width = clips.size()
    clips = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(clips)
    h = h.view(batch_size, frames, -1, h.size(-1)).flatten(1, 2)
    if normalize_reps:
        h = F.layer_norm(h, (h.size(-1),))
    return h


@torch.inference_mode()
def predict_next_latent(predictor, z_context, actions, states, normalize_reps=True):
    tokens_per_frame = z_context.size(1) // actions.size(1)
    pred_next = predictor(z_context, actions.to(z_context.dtype), states.to(z_context.dtype))[:, -tokens_per_frame:]
    if normalize_reps:
        pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
    return pred_next


@torch.inference_mode()
def rollout_autoregressive(predictor, z0, actions, states, steps, normalize_reps=True):
    tokens_per_frame = z0.size(1)
    z_history = z0
    predictions = []
    for step in range(steps):
        pred_next = predictor(
            z_history,
            actions[:, : step + 1].to(z0.dtype),
            states[:, : step + 1].to(z0.dtype),
        )[:, -tokens_per_frame:]
        if normalize_reps:
            pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
        predictions.append(pred_next)
        z_history = torch.cat([z_history, pred_next], dim=1)
    return predictions


@torch.inference_mode()
def compute_dx_dz_landscape(
    predictor,
    z0,
    z_goal,
    states,
    gt_action,
    nsamples,
    grid_size,
    normalize_reps=True,
):
    dx_values = np.linspace(float(gt_action[0] - grid_size), float(gt_action[0] + grid_size), nsamples)
    dz_values = np.linspace(float(gt_action[2] - grid_size), float(gt_action[2] + grid_size), nsamples)
    action_grid = []
    for dx in dx_values:
        for dz in dz_values:
            action = gt_action.copy()
            action[0] = dx
            action[2] = dz
            action_grid.append(action)

    action_grid = torch.tensor(np.stack(action_grid), device=z0.device, dtype=z0.dtype).unsqueeze(1)
    z0_grid = z0.repeat(action_grid.size(0), 1, 1)
    z_goal_grid = z_goal.repeat(action_grid.size(0), 1, 1)
    states_grid = states[:, :1].to(z0.dtype).repeat(action_grid.size(0), 1, 1)

    pred_next = predictor(z0_grid, action_grid, states_grid)[:, -z_goal.size(1) :]
    if normalize_reps:
        pred_next = F.layer_norm(pred_next, (pred_next.size(-1),))
    energy = torch.mean(torch.abs(pred_next - z_goal_grid), dim=[1, 2])
    return action_grid[:, 0].detach().float().cpu().numpy(), energy.detach().float().cpu().numpy()


def save_frame_strip(frames, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_strip = np.transpose(frames, (1, 0, 2, 3)).reshape(frames.shape[1], frames.shape[0] * frames.shape[2], 3)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(frame_strip)
    ax.set_axis_off()
    ax.set_title("LIBERO sampled clip")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_energy_landscape(action_grid, energy, gt_action, nsamples, output_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap = energy.reshape(nsamples, nsamples)
    x_values = action_grid[:, 0].reshape(nsamples, nsamples)[:, 0]
    z_values = action_grid[:, 2].reshape(nsamples, nsamples)[0, :]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(
        heatmap.T,
        origin="lower",
        extent=[x_values.min(), x_values.max(), z_values.min(), z_values.max()],
        cmap="viridis",
        aspect="auto",
    )
    ax.scatter([gt_action[0]], [gt_action[2]], c="red", marker="x", s=80, label="ground truth")
    best_index = int(np.argmin(energy))
    ax.scatter([action_grid[best_index, 0]], [action_grid[best_index, 2]], c="white", marker="o", s=40, label="best")
    ax.set_xlabel("Action delta x")
    ax.set_ylabel("Action delta z")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="prediction energy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def deterministic_sample_seed(seed, sample_index):
    return int(seed + sample_index * 1009)


def ensure_output_dir(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
