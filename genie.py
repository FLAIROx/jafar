from typing import Dict, Any, Optional

from orbax.checkpoint import PyTreeCheckpointer
import jax
import jax.numpy as jnp
import flax.linen as nn

from models.dynamics import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE

class MaskGITStep(nn.Module):
    dynamics: nn.Module
    tokenizer: nn.Module
    temperature: float
    sample_argmax: bool
    steps: int

    @nn.compact
    def __call__(self, carry, x):
        rng, final_token_idxs, mask, token_idxs, action_tokens = carry
        step = x
        B, _, N = token_idxs.shape[:3]

        # --- Construct video ---
        vid_token_idxs = jnp.concatenate((token_idxs, jnp.expand_dims(final_token_idxs, 1)), axis=1)

        # --- Encode video ---
        vid_embed = self.dynamics.patch_embed(vid_token_idxs)
        vid_embed = vid_embed.at[:, -1].set(
            jnp.where(jnp.reshape(mask, (B, N, 1)), self.dynamics.mask_token[0], vid_embed[:, -1])
        )

        # --- Predict transition ---
        act_embed = self.dynamics.action_up(action_tokens)
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        final_logits = self.dynamics.dynamics(vid_embed)[:, -1] / self.temperature

        # --- Sample new tokens for final frame ---
        if self.sample_argmax:
            sampled_token_idxs = jnp.argmax(final_logits, axis=-1)
        else:
            rng, _rng = jax.random.split(rng)
            sampled_token_idxs = jnp.where(
                step == self.steps - 1,
                jnp.argmax(final_logits, axis=-1),
                jax.random.categorical(_rng, final_logits),
            )
        final_token_idxs = jnp.where(mask, sampled_token_idxs, final_token_idxs)
        final_token_probs = jnp.take(jax.nn.softmax(final_logits), final_token_idxs)

        # --- Update mask ---
        num_unmasked_tokens = jnp.round(
            N * jnp.cos(jnp.pi * step / (self.steps * 2))
        ).astype(int)
        final_token_probs += ~mask
        prob_threshold = jnp.sort(final_token_probs, axis=1, descending=True)[:, num_unmasked_tokens]
        mask = mask & (final_token_probs < jnp.expand_dims(prob_threshold, -1))

        new_carry = (rng, final_token_idxs, mask, token_idxs, action_tokens)
        return new_carry, final_token_idxs

class Genie(nn.Module):
    """Genie model"""

    # --- Tokenizer ---
    in_dim: int
    tokenizer_dim: int
    latent_patch_dim: int
    num_patch_latents: int
    patch_size: int
    tokenizer_num_blocks: int
    tokenizer_num_heads: int
    # --- LAM ---
    lam_dim: int
    latent_action_dim: int
    num_latent_actions: int
    lam_patch_size: int
    lam_num_blocks: int
    lam_num_heads: int
    # --- Dynamics ---
    dyna_dim: int
    dyna_num_blocks: int
    dyna_num_heads: int
    dropout: float
    mask_limit: float

    def setup(self):
        self.tokenizer = TokenizerVQVAE(
            in_dim=self.in_dim,
            model_dim=self.tokenizer_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_patch_latents,
            patch_size=self.patch_size,
            num_blocks=self.tokenizer_num_blocks,
            num_heads=self.tokenizer_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
        )
        self.lam = LatentActionModel(
            in_dim=self.in_dim,
            model_dim=self.lam_dim,
            latent_dim=self.latent_patch_dim,
            num_latents=self.num_latent_actions,
            patch_size=self.lam_patch_size,
            num_blocks=self.lam_num_blocks,
            num_heads=self.lam_num_heads,
            dropout=0.0,
            codebook_dropout=0.0,
        )
        self.dynamics = DynamicsMaskGIT(
            model_dim=self.dyna_dim,
            num_latents=self.num_patch_latents,
            num_blocks=self.dyna_num_blocks,
            num_heads=self.dyna_num_heads,
            dropout=self.dropout,
            mask_limit=self.mask_limit,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        tokenizer_outputs = self.tokenizer.vq_encode(batch["videos"], training=False)
        lam_outputs = self.lam.vq_encode(batch["videos"], training=False)
        outputs = dict(
            video_tokens=jax.lax.stop_gradient(tokenizer_outputs["indices"]),
            latent_actions=jax.lax.stop_gradient(lam_outputs["z_q"]),
        )
        outputs["mask_rng"] = batch["mask_rng"]
        dyna_outputs = self.dynamics(outputs, training)
        outputs.update(dyna_outputs)
        mle_indices = jnp.argmax(outputs["token_logits"], axis=-1)
        outputs["recon"] = self.tokenizer.decode(
            mle_indices, batch["videos"].shape[2:4]
        )
        return outputs

    @nn.compact
    def sample(self, batch: Dict[str, Any], steps: int = 25, temperature: int = 1, sample_argmax: bool = False) -> Any:
        # Tokenize video
        token_idxs = self.tokenizer.vq_encode(batch["videos"], training=False)['indices']
        new_frame = jnp.zeros_like(token_idxs)[:, 0]
        B, _, N = token_idxs.shape[:3]
        # Get action tokens
        action_tokens = self.lam.vq.get_codes(batch["latent_actions"])
        # Create mask
        init_mask = jnp.ones_like(token_idxs, dtype=bool)[:, 0]

        init_carry = (batch["rng"], new_frame, init_mask, token_idxs, action_tokens)
        MaskGITScanned = nn.scan(
            MaskGITStep,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=steps,
        )
        final_carry, outputs = MaskGITScanned(dynamics=self.dynamics,
            tokenizer=self.tokenizer,
            temperature=temperature,
            sample_argmax=sample_argmax,
            steps=steps)(init_carry, jnp.arange(steps))

        new_frame = final_carry[1]
        token_idxs = jnp.concatenate((token_idxs, jnp.expand_dims(new_frame, 1)), axis=1)
        vid_gen = self.tokenizer.decode(
            token_idxs,
            video_hw=batch['videos'].shape[2:4],
        )
        return vid_gen
        
    def vq_encode(self, batch, training) -> Dict[str, Any]:
        # --- Preprocess videos ---
        lam_output = self.lam.vq_encode(batch["videos"], training=training)
        return lam_output["indices"]

def restore_genie_checkpoint(
    params: Dict[str, Any], tokenizer: str, lam: str, dyna: Optional[str] = None
):
    """Restore pre-trained Genie components"""
    params["params"]["tokenizer"].update(
        PyTreeCheckpointer().restore(tokenizer)["model"]["params"]["params"]
    )
    params["params"]["lam"].update(
        PyTreeCheckpointer().restore(lam)["model"]["params"]["params"]
    )
    if dyna:
        params["params"]["dyna"].update(
            PyTreeCheckpointer().restore(dyna)["model"]["params"]["params"]
        )
    return params
