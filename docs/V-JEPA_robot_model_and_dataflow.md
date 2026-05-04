# V-JEPA 机器人模型结构与 Data Flow

本文档说明当前 repo 中 `app.vjepa_droid` 这条机器人训练链路的模型结构、张量流向，以及基于 LIBERO 的正式训练建议。

## 1. 这条训练链路到底在训练什么

当前这条机器人链路不是在重新做 V-JEPA / V-JEPA 2.1 主干预训练。

它做的是：

```text
输入一段机器人视频 clip
用 target encoder 提取每帧 latent
再用 action-conditioned predictor 学：
  (过去 latent, 当前 action, 当前 robot state, 可选 extrinsics)
  -> 预测未来 latent
```

也就是一个基于视觉 latent 的 robot world model。

## 2. 模型结构

当前 ViT-B 训练配置对应的主要模块：

```text
LIBERO / DROID dataloader
    -> 视频 transform
    -> target encoder (V-JEPA 2.1 ViT-B)
    -> latent sequence h
    -> AC predictor
    -> latent prediction loss
```

更细一点：

```text
clips [B, C, T, H, W]
  -> target_encoder
  -> h [B, T * tokens_per_frame, D]

actions [B, T-1, 7]
states  [B, T,   7]
extrinsics [B, T, 6]   # 当前 LIBERO 配置里不启用

h 的过去部分 + actions/states/(extrinsics)
  -> predictor
  -> z_tf, z_ar
  -> 和 h 的未来部分做 L1-style latent loss
```

## 3. 当前代码里的真实角色分工

要注意一个很重要的实现细节：

```text
target_encoder: 参与 forward，冻结
predictor:      参与 forward/backward，实际被训练
encoder:        当前这条 loss 路径里没有真正参与 forward
```

也就是说，当前 `app.vjepa_droid.train` 这条线在实际效果上更接近：

```text
加载 V-JEPA 2.1 ViT-B encoder 作为固定特征提取器
训练 AC predictor
```

这也是为什么它能在单卡上更稳地跑起来。

## 4. Data Flow

### 4.1 Dataloader 输出 contract

训练 loop 收到的 sample 是：

```text
sample[0] = clips
sample[1] = actions
sample[2] = states
sample[3] = extrinsics
sample[4] = indices
```

batch 后的 shape：

```text
clips:      [B, C, T, H, W]
actions:    [B, T-1, 7]
states:     [B, T, 7]
extrinsics: [B, T, 6]
indices:    [B, T] 或 [B, T_raw]
```

语义：

```text
actions    = [dx, dy, dz, droll, dpitch, dyaw, dgripper]
states     = [x, y, z, roll, pitch, yaw, gripper]
extrinsics = [x, y, z, roll, pitch, yaw]
```

## 4.2 视频到 latent

在训练里，每帧会先被整理成 2-frame tubelet 形式再进 encoder：

```text
[B, C, T, H, W]
-> [B*T, C, 2, H, W]
-> target_encoder
-> [B, T * tokens_per_frame, D]
```

当前 ViT-B 配置：

```text
crop_size = 256
patch_size = 16
tokens_per_frame = (256 / 16)^2 = 256
D = 768
T = 8
```

所以：

```text
h.shape = [B, 8 * 256, 768] = [B, 2048, 768]
```

## 4.3 什么是 N_ctxt

`N_ctxt` 是 predictor 输入的上下文 visual token 数。

在 predictor 里：

```text
x: [B, N_ctxt, D]
```

这里的 `x` 还只是视觉 latent tokens，还没有拼 action/state token。

因此：

```text
N_ctxt = 上下文帧数 * tokens_per_frame
```

例如当前 8 帧训练里，一步 teacher forcing 用的是前 7 帧做上下文：

```text
N_ctxt = 7 * 256 = 1792
```

## 4.4 predictor 如何接收 action / state / extrinsics

predictor 会把每个时间步的条件信息先映射成 token：

```text
action_t     -> action token
state_t      -> state token
extrinsics_t -> extrinsics token   # 仅当 use_extrinsics=True
```

然后和该时间步的 patch tokens 拼在一起：

```text
[action_t, state_t, extrinsics_t, patch_1, ..., patch_N]
```

当前 LIBERO 训练配置中：

```yaml
model:
  use_extrinsics: false
```

所以实际只会用：

```text
[action_t, state_t, patch_1, ..., patch_N]
```

## 4.5 Loss

当前训练里有两部分 latent loss：

```text
jloss: one-step teacher forcing latent prediction
sloss: short autoregressive rollout latent prediction
loss = jloss + sloss
```

这两项都是拿 predictor 输出的 latent 去对齐 target encoder 产出的未来 latent。

## 5. extrinsics 在哪里用

`extrinsics` 不是 V-JEPA / V-JEPA 2.1 主干预训练的一部分。

它只出现在机器人 action-conditioned predictor 这条线里，用作可选条件输入。

因此更准确地说：

```text
V-JEPA / V-JEPA 2.1 主干预训练：不使用 extrinsics
机器人 AC predictor 微调：可选使用 extrinsics
```

当前 LIBERO 路线因为没有可靠的 DROID-style camera extrinsics，所以：

```text
loader 返回 dummy zeros
训练配置保持 use_extrinsics: false
```

## 6. LIBERO 与默认 DROID loader 是否对齐

现在两者主 contract 已对齐：

```text
buffer, actions, states, extrinsics, indices
```

并且 shape 对齐为：

```text
actions:    [T-1, 7]
states:     [T, 7]
extrinsics: [T, 6]
```

当前差异主要不是接口，而是语义：

- DROID: extrinsics 来自真实相机外参
- LIBERO: extrinsics 当前是全 0 占位

## 7. 当前正式训练建议

### 7.1 训练目标

先做一个稳妥的 ViT-B baseline：

```text
固定 target encoder
训练 AC predictor
在 LIBERO 500 demos 上看 loss 是否稳定下降
再用 validation script 看 energy landscape 是否更有结构
```

### 7.2 推荐训练阶段

推荐分两段走。

第一段：pilot

```text
目的：确认正式训练不会中途炸、loss 能继续下降、checkpoint 保存正常
配置：ViT-B, batch_size=1, 全量 500 demos, 5 epochs
预计时间：15-20 分钟
```

第二段：main baseline

```text
目的：得到一个真正可用的 ViT-B baseline checkpoint
配置：在 pilot 没问题后，继续跑 20 epochs
预计时间：50-70 分钟
```

合起来，如果直接从头跑完整 baseline：

```text
总时间大约 65-90 分钟
```

这个估计基于当前单卡 24GB 机器、debug run 实测单 step 约 0.17-0.33 秒、以及每个 epoch 500 iter 的量级。

### 7.3 为什么先跑 pilot

因为当前我们刚修过两件事：

- LIBERO 图像方向 `flipud`
- loader / contract 文档对齐

所以最稳的做法是先用 5 epoch pilot 确认：

```text
新数据方向没问题
loss 曲线正常
validation 脚本能接最新 checkpoint
```

确认这些都稳定，再继续拉长训练时长。

## 8. 建议先跑的配置

建议先用：

```text
configs/train/vitb16/libero-256px-8f-pilot.yaml
```

然后用下面的验证脚本检查：

```text
scripts/libero_vitb_ac_validate.py
```

## 9. 推荐执行顺序

```text
1. 跑 5 epoch pilot
2. 检查 latest.pt 和 log_r0.csv
3. 在固定 sample 上跑 validation + energy landscape
4. 如果 loss 和可视化正常，再继续跑到 20 epoch main baseline
```
