from dataclasses import dataclass
import time

import dm_pix as pix
import einops
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
from PIL import Image, ImageDraw
import tyro

from genie import Genie
from utils.dataloader import get_dataloader


@dataclass
class Args:
    # Experiment
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    data_dir: str = "data/coinrun_episodes"
    checkpoint: str = ""
    # Sampling
    batch_size: int = 1
    maskgit_steps: int = 25
    temperature: float = 1.0
    sample_argmax: bool = True
    start_frame: int = 0
    # Tokenizer checkpoint
    tokenizer_dim: int = 512
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 8
    tokenizer_num_heads: int = 8
    # LAM checkpoint
    lam_dim: int = 512
    latent_action_dim: int = 32
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 8
    lam_num_heads: int = 8
    # Dynamics checkpoint
    dyna_dim: int = 512
    dyna_num_blocks: int = 12
    dyna_num_heads: int = 8


args = tyro.cli(Args)
rng = jax.random.PRNGKey(args.seed)

# --- Load Genie checkpoint ---
genie = Genie(
    # Tokenizer
    in_dim=args.image_channels,
    tokenizer_dim=args.tokenizer_dim,
    latent_patch_dim=args.latent_patch_dim,
    num_patch_latents=args.num_patch_latents,
    patch_size=args.patch_size,
    tokenizer_num_blocks=args.tokenizer_num_blocks,
    tokenizer_num_heads=args.tokenizer_num_heads,
    # LAM
    lam_dim=args.lam_dim,
    latent_action_dim=args.latent_action_dim,
    num_latent_actions=args.num_latent_actions,
    lam_patch_size=args.lam_patch_size,
    lam_num_blocks=args.lam_num_blocks,
    lam_num_heads=args.lam_num_heads,
    # Dynamics
    dyna_dim=args.dyna_dim,
    dyna_num_blocks=args.dyna_num_blocks,
    dyna_num_heads=args.dyna_num_heads,
)
rng, _rng = jax.random.split(rng)
image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
dummy_inputs = dict(
    videos=jnp.zeros((args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32),
    mask_rng=_rng,
)
rng, _rng = jax.random.split(rng)
params = genie.init(_rng, dummy_inputs)
ckpt = PyTreeCheckpointer().restore(args.checkpoint)["model"]["params"]["params"]
params["params"].update(ckpt)

# --- Define autoregressive sampling loop ---
def _autoreg_sample(rng, video_batch, action_batch):
    vid = video_batch[:, : args.start_frame + 1]
    for frame_idx in range(args.start_frame + 1, args.seq_len):
        # --- Sample next frame ---
        print("Frame", frame_idx)
        rng, _rng = jax.random.split(rng)
        batch = dict(videos=vid, latent_actions=action_batch[:, :frame_idx], rng=_rng)
        new_frame = genie.apply(
            params,
            batch,
            args.maskgit_steps,
            args.temperature,
            args.sample_argmax,
            method=Genie.sample,
        )
        vid = jnp.concatenate([vid, new_frame], axis=1)
    return vid


# --- Get video + latent actions ---
dataloader = get_dataloader(args.data_dir, args.seq_len, args.batch_size)
video_batch = next(iter(dataloader))
# Get latent actions from first video only
first_video = video_batch[:1]
batch = dict(videos=first_video)
action_batch = genie.apply(params, batch, False, method=Genie.vq_encode)
action_batch = action_batch.reshape(1, args.seq_len - 1, 1)
# Use actions from first video for all videos
action_batch = jnp.repeat(action_batch, video_batch.shape[0], axis=0)

# --- Sample + evaluate video ---
vid = _autoreg_sample(rng, video_batch, action_batch)
gt = video_batch[:, : vid.shape[1]].clip(0, 1).reshape(-1, *video_batch.shape[2:])
recon = vid.clip(0, 1).reshape(-1, *vid.shape[2:])
ssim = pix.ssim(gt[:, args.start_frame + 1 :], recon[:, args.start_frame + 1 :]).mean()
print(f"SSIM: {ssim}")

# --- Construct video ---
first_true = (video_batch[0:1] * 255).astype(np.uint8)
first_pred = (vid[0:1] * 255).astype(np.uint8)
first_video_comparison = np.zeros((2, *vid.shape[1:5]), dtype=np.uint8)
first_video_comparison[0] = first_true[:, : vid.shape[1]]
first_video_comparison[1] = first_pred
# For other videos, only show generated video
other_preds = (vid[1:] * 255).astype(np.uint8)
all_frames = np.concatenate([first_video_comparison, other_preds], axis=0)
flat_vid = einops.rearrange(all_frames, "n t h w c -> t h (n w) c")

# --- Save video ---
imgs = [Image.fromarray(img) for img in flat_vid]
# Write actions on each frame
for img, action in zip(imgs[1:], action_batch[0, :, 0]):
    d = ImageDraw.Draw(img)
    d.text((2, 2), f"{action}", fill=255)
imgs[0].save(
    f"generation_{time.time()}.gif",
    save_all=True,
    append_images=imgs[1:],
    duration=250,
    loop=0,
)
