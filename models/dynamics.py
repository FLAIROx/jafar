from typing import Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.nn import STTransformer


def cosine_schedule(
    step: jax.Array,
    logits: jax.Array,
    n_toks: int,
    generation_steps: int = 25,
    blend_factor: float = 1.0,
) -> jax.Array:
    """
    Apply a cosine-based schedule to determine the number of tokens to mask at each generation step.

    This function implements a cosine schedule for token masking in a generation process,
    typically used in models like MaskGIT. It calculates the number of tokens to mask
    based on the current step, total number of tokens, and a blend factor.

    Args:
        step (jax.Array): Current generation step. Should be a scalar or 1D array.
        logits (jax.Array): Logits from the model. Used to determine the shape of the output.
        n_toks (int): Total number of tokens in the sequence.
        generation_steps (int, optional): Total number of generation steps. Defaults to 25.
        blend_factor (float, optional): Factor to blend between cosine and inverse cosine schedules.
                                        Should be in the range [0, 1]. Defaults to 1.0.

    Returns:
        jax.Array: Number of tokens to mask at the current step. Has the same leading dimensions as `logits`.

    Note:
        - The function uses a combination of cosine and inverse cosine schedules to determine
          the masking capacity at each step.
        - The blend factor determines the mix between these two schedules.
        - The output is clipped to ensure at least one token is unmasked at each step,
          and all tokens are unmasked by the final step.
    """

    normalized_step = step / (generation_steps - 1)
    cos = jnp.cos(jnp.pi / 2 * normalized_step)
    inv_cos = 1 - jnp.cos(jnp.pi / 2 * (1 - normalized_step))
    gamma = (blend_factor + 1) / 2  # to [0, 1]
    capacity = cos * gamma + inv_cos * (1 - gamma)
    n_mask_toks = jnp.round((1 - capacity) * n_toks)
    n_mask_toks =  n_mask_toks.astype(jnp.int32)
    n_mask_toks = jnp.tile(n_mask_toks, logits.shape[:-2])
       
    max_toks = jnp.minimum(n_toks, step + 1)
    linspace = jnp.linspace(1, max_toks, n_mask_toks.shape[0], dtype=jnp.int32)
    n_mask_toks = jnp.maximum(n_mask_toks, linspace[:, None])
    condition = (step + 1) >= generation_steps
    n_mask_toks = n_mask_toks.at[condition].set(n_toks)

    return n_mask_toks

class DynamicsMaskGIT(nn.Module):
    """MaskGIT dynamics model"""

    model_dim: int
    num_latents: int
    num_blocks: int
    num_heads: int
    dropout: float
    mask_limit: float

    def setup(self):
        self.dynamics = STTransformer(
            self.model_dim,
            self.num_latents,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.patch_embed = nn.Embed(self.num_latents, self.model_dim)
        self.mask_token = self.param(
            "mask_token",
            nn.initializers.lecun_uniform(),
            (1, 1, 1, self.model_dim),
        )
        self.action_up = nn.Dense(self.model_dim)

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        # --- Mask videos ---
        vid_embed = self.patch_embed(batch["video_tokens"])
        if training:
            rng1, rng2 = jax.random.split(batch["mask_rng"])
            mask_prob = jax.random.uniform(rng1, minval=self.mask_limit)
            mask = jax.random.bernoulli(rng2, mask_prob, vid_embed.shape[:-1])
            mask = mask.at[:, 0].set(False)
            vid_embed = jnp.where(jnp.expand_dims(mask, -1), self.mask_token, vid_embed)
        else:
            mask = None

        # --- Predict transition ---
        act_embed = self.action_up(batch["latent_actions"])
        vid_embed += jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))
        logits = self.dynamics(vid_embed)
        return dict(token_logits=logits, mask=mask)

    def _sample_internal(self, 
                         batch: Dict[str, Any], 
                         rng: jax.random.PRNGKey,
                         generation_steps: int = 25,
                         temp: float = 1.0):

        act_embed = self.action_up(batch["latent_actions"])
        act_embed = jnp.pad(act_embed, ((0, 0), (1, 0), (0, 0), (0, 0)))

        B, T, N = batch["video_tokens"].shape
        vid_act_embed, gen_act_embed = act_embed[:, :T], act_embed[:, T:]

        vid_embed = self.patch_embed(batch["video_tokens"]) + vid_act_embed

        def gen_step(state, step):
            gen, mask, rng = state

            gen_embed = self.patch_embed(gen) + gen_act_embed
            gen_embed = jnp.where(jnp.expand_dims(mask, -1), 0, gen_embed)
            gen_embed = jnp.concatenate([vid_embed, gen_embed], axis=1)
            logits = self.dynamics(gen_embed)[:, T:]

            n_mask_toks = cosine_schedule(
                step, logits, N,
                generation_steps=generation_steps,
            )

            rng, rng_gen = jax.random.split(rng)
            next_gen = jax.random.categorical(rng_gen, logits / temp)

            p_tokens = jax.nn.softmax(logits)
            p_tokens = jnp.take_along_axis(p_tokens, next_gen[..., None], axis=-1).squeeze(-1) + mask

            def get_threshold(x, idx):
                return jax.lax.dynamic_slice(x, (idx,), (1,))[0]

            limit_indices = N - n_mask_toks
            p_tokens_sorted = jnp.sort(p_tokens, axis=-1)
            limit = jax.vmap(jax.vmap(get_threshold))(p_tokens_sorted, limit_indices)[..., None]
            next_mask = (p_tokens >= limit) & ~mask

            gen = jnp.where(next_mask, next_gen, gen)
            mask = mask | next_mask

            return (gen, mask, rng)
        
        mask = jnp.zeros((B, 1, N), dtype=jnp.bool)        
        generated_frame = jnp.zeros((B, 1 , N), dtype=jnp.int32)
        rng, rng_run = jax.random.split(rng)
        state = (generated_frame, mask, rng_run)

        for i in range(0, generation_steps):
            state = gen_step(state, i)

        generated_frame = state[0]
        vid_gen = jnp.concatenate([batch["video_tokens"], generated_frame], axis=1)

        return dict(vid_gen=vid_gen)
    
    def sample(
            self,
            params,
            batch,
            rng,
            generation_steps: int = 25,
            temp: float = 1.0,
    ):
        return self.apply(params, batch, rng, generation_steps, temp, method=self._sample_internal)
