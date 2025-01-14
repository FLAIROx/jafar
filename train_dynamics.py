from dataclasses import dataclass
import os
import time

import einops
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import optax
import orbax
import numpy as np
import jax
import jax.numpy as jnp
import tyro
import wandb

from genie import Genie, restore_genie_components
from utils.dataloader import get_dataloader

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
    min_lr: float = 3e-6
    max_lr: float = 3e-5
    warmup_steps: int = 5000
    # Tokenizer
    tokenizer_dim: int = 512
    latent_patch_dim: int = 32
    num_patch_latents: int = 1024
    patch_size: int = 4
    tokenizer_num_blocks: int = 8
    tokenizer_num_heads: int = 8
    tokenizer_checkpoint: str = ""
    # LAM
    lam_dim: int = 512
    latent_action_dim: int = 32
    num_latent_actions: int = 6
    lam_patch_size: int = 16
    lam_num_blocks: int = 8
    lam_num_heads: int = 8
    lam_checkpoint: str = ""
    # Dynamics
    dyna_dim: int = 512
    dyna_num_blocks: int = 12
    dyna_num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    # Logging
    log: bool = False
    entity: str = ""
    project: str = ""
    log_interval: int = 5
    log_image_interval: int = 250
    ckpt_dir: str = ""
    log_checkpoint_interval: int = 25000
    log_gradients: bool = False


args = tyro.cli(Args)
rng = jax.random.PRNGKey(args.seed)
if args.log:
    wandb.init(entity=args.entity, project=args.project, group="debug", config=args)

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
    mask_rng=_rng,
)
rng, _rng = jax.random.split(rng)
init_params = genie.init(_rng, dummy_inputs)
init_params = restore_genie_components(
    init_params, args.tokenizer_checkpoint, args.lam_checkpoint
)
lr_schedule = optax.warmup_cosine_decay_schedule(
    args.min_lr, args.max_lr, args.warmup_steps, args.num_steps
)
tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4)
train_state = TrainState.create(apply_fn=genie.apply, params=init_params, tx=tx)


# --- Define dynamics loss + train step ---
def dynamics_loss_fn(params, state, inputs):
    """Compute masked dynamics loss"""
    outputs = state.apply_fn(
        params, inputs, training=True, rngs={"dropout": inputs["dropout_rng"]}
    )
    mask = outputs["mask"]
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(
        outputs["token_logits"], outputs["video_tokens"]
    )
    ce_loss = (mask * ce_loss).sum() / mask.sum()
    acc = outputs["token_logits"].argmax(-1) == outputs["video_tokens"]
    acc = (mask * acc).sum() / mask.sum()
    select_probs = jax.nn.softmax(outputs["token_logits"])
    metrics = dict(
        cross_entropy_loss=ce_loss,
        masked_token_accuracy=acc,
        select_logit=outputs["token_logits"].max(-1).mean(),
        select_p=select_probs.max(-1).mean(),
        entropy=jax.scipy.special.entr(select_probs).sum(-1).mean(),
    )
    return ce_loss, (outputs["recon"], metrics)


@jax.jit
def train_step(state, inputs):
    """Update state and compute metrics"""
    grad_fn = jax.value_and_grad(dynamics_loss_fn, has_aux=True, allow_int=True)
    (loss, (recon, metrics)), grads = grad_fn(state.params, state, inputs)
    state = state.apply_gradients(grads=grads)
    if args.log_gradients:
        metrics["gradients_std/"] = jax.tree.map(
            lambda x: x.std(), grads["params"]["dynamics"]
        )
    return state, loss, recon, metrics


# --- TRAIN LOOP ---
dataloader = get_dataloader(args.file_path, args.seq_len, args.batch_size)
step = 0
while step < args.num_steps:
    for videos in dataloader:
        # --- Train step ---
        rng, _rng, _mask_rng = jax.random.split(rng, 3)
        inputs = dict(
            videos=jnp.array(videos, dtype=jnp.float32) / 255.0,
            action=jnp.zeros((args.batch_size, args.seq_len), dtype=jnp.float32),
            dropout_rng=_rng,
            mask_rng=_mask_rng,
        )
        train_state, loss, recon, metrics = train_step(train_state, inputs)
        print(f"Step {step}, loss: {loss}")
        step += 1

        # --- Logging ---
        if args.log:
            if step % args.log_interval == 0:
                wandb.log({"loss": loss, "step": step, **metrics})
            if step % args.log_image_interval == 0:
                gt_seq = inputs["videos"][0]
                recon_seq = recon[0].clip(0, 1)
                comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                comparison_seq = einops.rearrange(
                    comparison_seq * 255, "t h w c -> h (t w) c"
                )
                log_images = dict(
                    image=wandb.Image(np.asarray(gt_seq[15])),
                    recon=wandb.Image(np.asarray(recon_seq[15])),
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
                    os.path.join(os.getcwd(), args.ckpt_dir, f"genie_{ts}_{step}"),
                    ckpt,
                    save_args=save_args,
                )
        if step >= args.num_steps:
            break
