"""MLP (Feed-Forward Network) for the DemoTransformer."""
import torch as t
import torch.nn as nn
import einops
from torch import Tensor
from jaxtyping import Float
from transformer_lens.utils import gelu_new

from .config import Config


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            )
            + self.b_in
        )
        post = gelu_new(pre)
        mlp_out = (
            einops.einsum(
                post, self.W_out, "batch position d_mlp, d_mlp d_model -> batch position d_model"
            )
            + self.b_out
        )
        return mlp_out
