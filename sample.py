from dataclasses import dataclass
import time

from orbax.checkpoint import PyTreeCheckpointer
import numpy as np
import jax
import jax.numpy as jnp
import tyro

from data.dataloader import get_dataloader
from genie import Genie

ts = int(time.time())

@dataclass
class Args:
    # Experiment
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    file_path: str = "/home/duser/jafar/data/coinrun.npy"
    # Optimization
    batch_size: int = 3
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
    log: bool = False
    entity: str = "flair"
    project: str = "jafari"
    ckpt_dir: str = "/home/duser/jafar/checkpoints"
    # Sampling
    checkpoint: str = "/home/duser/jafar/checkpoints/genie_1721738387_200000"
    maskgit_steps: int = 25
    temperature: float = 1.0
    sample_argmax: bool = False

args = tyro.cli(Args)
rng = jax.random.PRNGKey(args.seed)

# --- Construct train state ---
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

# --- Get video ---
dataloader = get_dataloader(args.file_path, args.seq_len, args.batch_size)
for vids in dataloader:
    video_batch = jnp.array(vids, dtype=jnp.float32) / 255.0
    break

latent_actions = jnp.ones(args.num_latent_actions).repeat(args.batch_size)[:args.batch_size]
rng, _rng = jax.random.split(rng)
batch = dict(
    videos=video_batch[:, :1],  # Full video batch
    latent_actions=jnp.ones((args.batch_size, 1, 1), dtype=jnp.int32)*0,    # A single latent action per video frame, (B, T, 1)
    rng=_rng,
)

# --- Sample next frame ---
@jax.jit
def _sample(batch):
    return genie.apply(params, batch, args.maskgit_steps, args.temperature, args.sample_argmax, method=Genie.sample)

vid = _sample(batch)

# Autoregressive loop for generation
for i in range(args.seq_len - 1):
    batch["videos"] = vid  # Update the batch with the new video patch
    batch["latent_actions"] = jnp.concatenate([batch["latent_actions"], jnp.ones((args.batch_size, 1, 1), dtype=jnp.int32)*0], axis=1)
    vid = _sample(batch)  # Generate the next frame based on the updated video patch

# def imshow(img):
#     import cv2
#     import IPython
#     _,ret = cv2.imencode('.jpg', img)
#     i = IPython.display.Image(data=ret)
#     IPython.display.display(i)

import matplotlib.pyplot as plt
import time
t = time.time()
for b in range(vid.shape[0]):
    for i in range(vid.shape[1]):
        plt.imsave(f'gens/{t}_{b}_{i}.png', np.asarray(vid[b, i]))
        # imshow(np.asarray(vid[b, i]*255.0))