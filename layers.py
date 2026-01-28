"""Embedding and normalization layers for the DemoTransformer."""
import torch as t
import torch.nn as nn
import einops
from torch import Tensor
from jaxtyping import Float, Int

from config import Config


class LayerNorm(nn.Module):
    """Layer normalization."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_var = residual.var(dim=-1, keepdim=True, unbiased=False)
        residual = (residual - residual_mean) / (residual_var + self.cfg.layer_norm_eps).sqrt()
        return residual * self.w + self.b


class Embed(nn.Module):
    """Token embedding layer."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    """Positional embedding layer."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.d_model, cfg.n_ctx))
        nn.init.normal_(self.W_pos, std=cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq_len d_model -> batch seq_len d_model", batch=batch)


class Unembed(nn.Module):
    """Unembedding layer to convert hidden states to vocabulary logits."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )
