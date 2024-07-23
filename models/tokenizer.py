from typing import Dict, Any, Tuple

import flax.linen as nn

from utils.preprocess import patchify, unpatchify
from utils.nn import STTransformer, VectorQuantizer


class TokenizerVQVAE(nn.Module):
    """ST-ViVit VQ-VAE"""

    in_dim: int
    model_dim: int
    latent_dim: int
    num_latents: int
    patch_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    codebook_dropout: float

    def setup(self):
        self.encoder = STTransformer(
            self.model_dim,
            self.latent_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )
        self.vq = VectorQuantizer(
            self.latent_dim,
            self.num_latents,
            self.codebook_dropout,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder = STTransformer(
            self.model_dim,
            self.out_dim,
            self.num_blocks,
            self.num_heads,
            self.dropout,
        )

    def __call__(self, batch: Dict[str, Any], training: bool = True) -> Dict[str, Any]:
        H, W = batch["videos"].shape[2:4]
        outputs = self.vq_encode(batch["videos"], training)
        recon = self.decoder(outputs["z_q"])  # (B, T, H_down * W_down, C)
        recon = nn.sigmoid(recon)
        outputs["recon"] = unpatchify(recon, self.patch_size, H, W)
        return outputs

    def vq_encode(self, videos: Any, training: bool = True) -> Dict[str, Any]:
        # --- Preprocess + encode ---
        B, T = videos.shape[:2]
        x = patchify(videos, self.patch_size)
        N = x.shape[2]
        x = self.encoder(x)  # (B, T, N, E)

        # --- Vector quantize ---
        x = x.reshape(B * T * N, self.latent_dim)
        z_q, z, emb, indices = self.vq(x, training)
        z_q = z_q.reshape(B, T, N, self.latent_dim)
        indices = indices.reshape(B, T, N)
        return dict(z_q=z_q, z=z, emb=emb, indices=indices)

    def decode(self, indices: Any, video_hw: Tuple[int]):
        z = self.vq.codebook[indices]
        recon = self.decoder(z)
        recon = nn.sigmoid(recon)
        return unpatchify(recon, self.patch_size, *video_hw)
