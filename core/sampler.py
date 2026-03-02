"""Sampling utilities for text generation with the DemoTransformer."""
import torch as t
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int

from .config import device
from .transformer import DemoTransformer


class TransformerSampler:
    """Handles text generation sampling from the DemoTransformer.

    Accepts any tokenizer with encode(text, return_tensors="pt") and decode(ids).
    Uses KV cache for fast autoregressive generation.
    """

    def __init__(self, model: DemoTransformer, tokenizer):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    def _encode_prompt(self, prompt: str) -> Tensor:
        """Encode prompt to 1D tensor of token IDs."""
        encoded = self.tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(encoded, list):
            encoded = t.tensor(encoded, dtype=t.long, device=device)
        else:
            encoded = encoded.to(device)
        if encoded.dim() == 2:
            return encoded[0]
        return encoded

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Uses KV cache: the prompt is processed in one forward pass, then each
        subsequent token only requires a single-token forward pass.

        Sampling terminates at max_tokens_generated, or when the model generates
        an end-of-sequence token.
        kwargs are passed to sample_next_token.
        """
        self.model.eval()
        input_ids = self._encode_prompt(prompt)

        # Prefill: process entire prompt and populate KV caches
        kv_caches = self.model.create_kv_caches()
        prompt_tokens = input_ids[None, -self.cfg.n_ctx :]  # (1, prompt_len)
        logits = self.model(prompt_tokens, kv_caches=kv_caches, cache_position=prompt_tokens.size(1))
        # Only need logits for the last token
        logits = logits[0, -1]
        next_token = t.tensor(
            [TransformerSampler.sample_next_token(input_ids, logits, **kwargs)],
            device=device,
        )
        input_ids = t.cat([input_ids, next_token], dim=-1)

        if verbose:
            print(self.tokenizer.decode(input_ids), end="\r")
        if next_token == getattr(self.tokenizer, "eos_token_id", None):
            return self.tokenizer.decode(input_ids)

        # Decode: generate one token at a time using KV cache
        for i in range(1, max_tokens_generated):
            # Only feed the last token; KV cache has the rest
            logits = self.model(
                next_token[None],  # (1, 1)
                kv_caches=kv_caches,
                cache_position=input_ids.size(0),
            )
            logits = logits[0, -1]
            next_token = t.tensor(
                [TransformerSampler.sample_next_token(input_ids, logits, **kwargs)],
                device=device,
            )
            input_ids = t.cat([input_ids, next_token], dim=-1)

            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """Returns the most likely token (as an int)."""
        return t.argmax(logits).item()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """Applies temperature scaling to the logits."""
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """Applies a frequency penalty to the logits."""
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """Samples from the distribution defined by the logits."""
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """Samples from the top k most likely tokens."""
        top_k_logits, top_k_ids = t.topk(logits, k)
        sampled_token = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        return top_k_ids[sampled_token].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """Samples from the most likely tokens which make up at least p cumulative probability."""
        # Sort logits, and get cumulative probabilities
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        # Choose which tokens to keep, in the set we sample from
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()
