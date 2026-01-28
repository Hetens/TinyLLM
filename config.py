"""Configuration for the DemoTransformer model."""
import torch as t
from dataclasses import dataclass

# Initialize CUDA device
device = 'cuda' if t.cuda.is_available() else 'cpu'


@dataclass
class Config:
    """Model configuration."""
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02  # for normal initialization the range
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_layers: int = 12
    n_heads: int = 12


@dataclass
class TransformerTrainingArgs:
    """Training configuration."""
    batch_size: int = 16
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None
