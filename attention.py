"""Attention mechanism for the DemoTransformer.

Supports optional KV-cache for fast autoregressive inference.
"""
import torch as t
import torch.nn as nn
import einops
from torch import Tensor
from jaxtyping import Float
from config import Config, device


class Attention(nn.Module):
    """Multi-head self-attention with causal masking and optional KV cache."""
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self,
        normalized_resid_pre: Float[Tensor, "batch posn d_model"],
        kv_cache: dict | None = None,
        cache_position: int | None = None,
    ) -> Float[Tensor, "batch posn d_model"]:
        """
        Forward pass with optional KV cache.

        Args:
            normalized_resid_pre: Input tensor.
            kv_cache: If provided, a dict with 'k' and 'v' tensors from
                      previous steps. Will be updated in-place with new K/V.
            cache_position: Total sequence length seen so far (including this step).
                            Required when kv_cache is provided.
        """
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # If we have a KV cache, concatenate cached keys/values with new ones
        if kv_cache is not None:
            if kv_cache["k"] is not None:
                k = t.cat([kv_cache["k"], k], dim=1)
                v = t.cat([kv_cache["v"], v], dim=1)
            # Update cache in-place
            kv_cache["k"] = k
            kv_cache["v"] = v

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(
            attn_scores / self.cfg.d_head**0.5,
            use_cache=kv_cache is not None,
        )
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out

    # why do we apply causal mask?
    # To prevent the model from attending to future tokens.
    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
        use_cache: bool = False,
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """Apply causal mask to prevent attending to future tokens.

        When use_cache=True, we skip masking because the query is only the
        newest token(s) and all keys are from the past (or current position),
        so causality is already enforced by construction.
        """
        if use_cache:
            # With KV cache, query positions only attend to past+current keys,
            # so no masking is needed.
            return attn_scores
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
