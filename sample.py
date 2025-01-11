from dataclasses import dataclass
import time

import dm_pix as pix
import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer
import tyro

from data.dataloader import get_dataloader
from genie import Genie


@dataclass
class Args:
    # Experiment
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    file_path: str = "data/coinrun.npy"
    # Optimization
    batch_size: int = 1
    # Tokenizer
    tokenizer_dim: int = 512
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 8
    tokenizer_num_heads: int = 8
    # LAM
    lam_dim: int = 512
    latent_action_dim: int = 32
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 8
    lam_num_heads: int = 8
    # Dynamics
    dyna_dim: int = 512
    dyna_num_blocks: int = 12
    dyna_num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    # Logging
    log: bool = True
    entity: str = "flair"
    project: str = "jafari"
    ckpt_dir: str = ""
    # Sampling
    checkpoint: str = ""
    maskgit_steps: int = 25
    temperature: float = 1.0
    sample_argmax: bool = False
    start_frame: int = 0

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
    dropout=args.dropout,
    mask_limit=args.mask_limit,
)
rng, _rng = jax.random.split(rng)
image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
dummy_inputs = dict(
    videos=jnp.zeros((args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32),
    mask_rng=_rng
)
rng, _rng = jax.random.split(rng)
params = genie.init(_rng, dummy_inputs)
params["params"].update(
    PyTreeCheckpointer().restore(args.checkpoint)["model"]["params"]["params"])

# --- Get video + latent actions ---
dataloader = get_dataloader(args.file_path, args.seq_len, args.batch_size)
for vids in dataloader:
    video_batch = jnp.array(vids, dtype=jnp.float32) / 255.0
    break
batch = dict(videos=video_batch)
lam_output = genie.apply(params, batch, False, method=Genie.vq_encode)
lam_output = lam_output.reshape(args.batch_size, args.seq_len-1, 1)

# --- Define autoregressive sampling loop ---
def _autoreg_sample(rng, video_batch):
    vid = video_batch[:, :args.start_frame+1]
    for frame_idx in range(args.start_frame+1, args.seq_len):
        # --- Sample next frame ---
        print("Frame", frame_idx)
        rng, _rng = jax.random.split(rng)
        batch = dict(videos=vid, latent_actions=lam_output[:, :frame_idx], rng=_rng)
        new_frame = genie.apply(
            params,
            batch,
            args.maskgit_steps,
            args.temperature,
            args.sample_argmax,
            method=Genie.sample
        )
        vid = jnp.concatenate([vid, new_frame], axis=1)
    return vid

# --- Sample + evaluate video ---
vid = _autoreg_sample(rng, video_batch)
gt = video_batch[:, :vid.shape[1]].clip(0, 1).reshape(-1, *video_batch.shape[2:])
recon = vid.clip(0, 1).reshape(-1, *vid.shape[2:])
psnr = pix.psnr(gt, recon).mean()
ssim = pix.ssim(gt, recon).mean()
print(f"PSNR: {psnr}, SSIM: {ssim}")

# --- Save generated video ---
original_frames = (video_batch * 255).astype(np.uint8)
interweaved_frames = np.zeros((vid.shape[0] * 2, *vid.shape[1:5]), dtype=np.uint8)
interweaved_frames[0::2] = original_frames[:, :vid.shape[1]]
interweaved_frames[1::2] = (vid * 255).astype(np.uint8)
flat_vid = einops.rearrange(interweaved_frames, "n t h w c -> (n h) (t w) c")
filename = f'interweaved_generation_{time.time()}.png'
plt.imsave(filename, flat_vid)
print(f"Generated video saved to {filename}")
