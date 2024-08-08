from typing import Dict, Any, Optional

from orbax.checkpoint import PyTreeCheckpointer
import jax
import jax.numpy as jnp
import flax.linen as nn

from models.dynamics import DynamicsMaskGIT
from models.lam import LatentActionModel
from models.tokenizer import TokenizerVQVAE


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
    
    def _sample_internal(self,
                         batch,
                         rng,
                         generation_steps: int = 25,
                         temp: float = 1.0):
        
        tokenizer_outputs = self.tokenizer.vq_encode(batch["videos"], training=False)
        lam_codes = self.lam.get_codebook[(batch['latent_actions'],)]

        dyna_outputs = self.dynamics._sample_internal(
            batch=dict(video_tokens=tokenizer_outputs["indices"],
                        latent_actions=lam_codes[:, :, None, :]),
            rng=rng,
            generation_steps=generation_steps,
            temp=temp,
        )

        vid_gen = self.tokenizer.decode(
            dyna_outputs["vid_gen"],
            video_hw=batch['videos'].shape[2:4],
        )
        return vid_gen #[:, -1:]
    
    def sample(self, params, batch, rng, generation_steps: int = 25, temp: float = 1.0):
        return self.apply(params, batch, rng, generation_steps, temp, method=self._sample_internal)


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
