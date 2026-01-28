"""Transformer block and main DemoTransformer model."""
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from config import Config
from layers import LayerNorm, Embed, Unembed
from attention import Attention
from mlp import MLP


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)

    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class DemoTransformer(nn.Module):
    """Main transformer model."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch posn"]) -> Float[Tensor, "batch posn d_vocab"]:
        resid_pre = self.embed(tokens)
        for block in self.blocks:
            resid_pre = block(resid_pre)
        return self.unembed(resid_pre)  # logits
