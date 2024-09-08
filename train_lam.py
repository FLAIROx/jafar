from dataclasses import dataclass
import os
import time

import einops
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import optax
import orbax
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import wandb
import tyro

from models.lam import LatentActionModel
from data.dataloader import get_dataloader

ts = int(time.time())


@dataclass
class Args:
    # Experiment
    num_steps: int = 200_000
    seed: int = 0
    seq_len: int = 16
    image_channels: int = 3
    image_resolution: int = 64
    file_path: str = "data/coinrun.npy"
    # Optimization
    batch_size: int = 36
    vq_beta: float = 0.25
    min_lr: float = 3e-6
    max_lr: float = 3e-5
    warmup_steps: int = 5000
    # LAM
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 6
    patch_size: int = 16
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 10000
    log_gradients: bool = False


args = tyro.cli(Args)
rng = jax.random.PRNGKey(args.seed)
if args.log:
    wandb.init(entity=args.entity, project=args.project, group="debug", config=args)

# --- Construct train state ---
lam = LatentActionModel(
    in_dim=args.image_channels,
    model_dim=args.model_dim,
    latent_dim=args.latent_dim,
    num_latents=args.num_latents,
    patch_size=args.patch_size,
    num_blocks=args.num_blocks,
    num_heads=args.num_heads,
    dropout=args.dropout,
    codebook_dropout=args.codebook_dropout,
)
image_shape = (args.image_resolution, args.image_resolution, args.image_channels)
rng, _rng = jax.random.split(rng)
inputs = dict(
    videos=jnp.zeros((args.batch_size, args.seq_len, *image_shape), dtype=jnp.float32),
    rng=_rng,
)
rng, _rng = jax.random.split(rng)
init_params = lam.init(_rng, inputs)
lr_schedule = optax.warmup_cosine_decay_schedule(
    args.min_lr, args.max_lr, args.warmup_steps, args.num_steps
)
tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4)
train_state = TrainState.create(apply_fn=lam.apply, params=init_params, tx=tx)


def lam_loss_fn(params, state, inputs):
    # --- Compute loss ---
    outputs = state.apply_fn(
        params, inputs, training=True, rngs={"dropout": inputs["rng"]}
    )
    gt_future_frames = inputs["videos"][:, 1:]
    mse = jnp.square(gt_future_frames - outputs["recon"]).mean()
    q_loss = jnp.square(jax.lax.stop_gradient(outputs["emb"]) - outputs["z"]).mean()
    commitment_loss = jnp.square(
        outputs["emb"] - jax.lax.stop_gradient(outputs["z"])
    ).mean()
    loss = mse + q_loss + args.vq_beta * commitment_loss

    # --- Compute validation metrics ---
    gt = gt_future_frames.clip(0, 1).reshape(-1, *gt_future_frames.shape[2:])
    recon = outputs["recon"].clip(0, 1).reshape(-1, *outputs["recon"].shape[2:])
    psnr = pix.psnr(gt, recon).mean()
    ssim = pix.ssim(gt, recon).mean()
    _, index_counts = jnp.unique_counts(
        outputs["indices"], size=args.num_latents, fill_value=0
    )
    codebook_usage = (index_counts != 0).mean()
    metrics = dict(
        loss=loss,
        mse=mse,
        q_loss=q_loss,
        commitment_loss=commitment_loss,
        psnr=psnr,
        ssim=ssim,
        codebook_usage=codebook_usage,
    )
    return loss, (outputs["recon"], metrics)


# --- Define train step ---
@jax.jit
def train_step(state, inputs):
    grad_fn = jax.value_and_grad(lam_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)
    state = state.apply_gradients(grads=grads)
    if args.log_gradients:
        metrics["encoder_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["encoder"]
        )
        metrics["vq_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["vq"]
        )
        metrics["decoder_gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["decoder"]
        )
    return state, loss, recon, metrics


# --- TRAIN LOOP ---
dataloader = get_dataloader(args.file_path, args.seq_len, args.batch_size)
step = 0
while step < args.num_steps:
    for videos in dataloader:
        # --- Train step ---
        rng, _rng = jax.random.split(rng)
        inputs = dict(
            videos=jnp.array(videos, dtype=jnp.float32) / 255.0,
            rng=_rng,
        )
        train_state, loss, recon, metrics = train_step(train_state, inputs)
        print(f"Step {step}, loss: {loss}")
        step += 1

        # --- Logging ---
        if args.log:
            if step % args.log_interval == 0:
                wandb.log({"loss": loss, "step": step, **metrics})
            if step % args.log_image_interval == 0:
                gt_seq = inputs["videos"][0][1:]
                recon_seq = recon[0].clip(0, 1)
                comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                comparison_seq = einops.rearrange(
                    comparison_seq * 255, "t h w c -> h (t w) c"
                )
                log_images = dict(
                    image=wandb.Image(np.asarray(gt_seq[0])),
                    recon=wandb.Image(np.asarray(recon_seq[0])),
                    true_vs_recon=wandb.Image(
                        np.asarray(comparison_seq.astype(np.uint8))
                    ),
                )
                wandb.log(log_images)
            if step % args.log_checkpoint_interval == 0:
                ckpt = {"model": train_state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(
                    os.path.join(args.ckpt_dir, f"lam_{ts}_{step}"),
                    ckpt,
                    save_args=save_args,
                )
        if step >= args.num_steps:
            break
