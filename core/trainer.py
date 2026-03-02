"""Training utilities for the DemoTransformer."""
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from tqdm import tqdm

from .config import Config, TransformerTrainingArgs, device
from .transformer import DemoTransformer
from .sampler import TransformerSampler


class TransformerTrainer:
    """Handles training, evaluation, and checkpointing of the DemoTransformer."""
    def __init__(
        self,
        args: TransformerTrainingArgs,
        model: DemoTransformer,
        train_loader,
        test_loader,
        tokenizer,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = TransformerSampler(self.model, tokenizer)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        self.step = 0
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer

        # Pre-compute & pin the static sudoku loss mask so we don't rebuild it every step
        self._sudoku_loss_mask: Tensor | None = None

    def _get_sudoku_loss_mask(self, batch_size: int, seq_len: int) -> Tensor:
        """Return the cached sudoku answer-only loss mask, creating it on first call."""
        if self._sudoku_loss_mask is None or self._sudoku_loss_mask.size(0) != batch_size:
            mask = t.zeros(seq_len - 1, dtype=t.float32, device=device)
            mask[81:162] = 1  # Answer positions (predicting tokens 82-162)
            self._sudoku_loss_mask = mask.unsqueeze(0).expand(batch_size, -1).contiguous()
        return self._sudoku_loss_mask

    def get_log_probs(
        self,
        logits: Float[Tensor, "batch posn d_vocab"],
        tokens: Int[Tensor, "batch posn"],
        loss_mask: Int[Tensor, "batch posn-1"] | None = None,
    ) -> Float[Tensor, "batch posn-1"]:
        log_probs = logits.log_softmax(dim=-1)
        log_probs_for_tokens = (
            log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        )
        if loss_mask is not None:
            log_probs_for_tokens = log_probs_for_tokens * loss_mask
        return log_probs_for_tokens

    def training_step(
        self,
        batch: dict[str, Int[Tensor, "batch seq"]],
        loss_mask: Int[Tensor, "batch seq-1"] | None = None,
    ) -> Float[Tensor, ""]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device, non_blocking=True)
        logits = self.model(tokens)
        log_probs = self.get_log_probs(logits, tokens, loss_mask)
        if loss_mask is not None:
            loss = -log_probs.sum() / loss_mask.sum().clamp(min=1)
        else:
            loss = -log_probs.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        totally_correct, total_samples = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            tokens = batch["tokens"].to(device, non_blocking=True)
            logits: Tensor = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            totally_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        accuracy = totally_correct / total_samples
        self.model.train()
        return accuracy

    @t.inference_mode()
    def evaluate_sudoku_accuracy(self) -> float:
        """Exact accuracy: fraction of puzzles where all 81 answer digits are correct."""
        self.model.eval()
        exact_correct, total = 0, 0
        # Answer tokens are at positions 82-162 (0-indexed)
        answer_start, answer_end = 82, 163
        for batch in tqdm(self.test_loader, desc="Evaluating (sudoku exact)"):
            tokens = batch["tokens"].to(device, non_blocking=True)
            logits: Tensor = self.model(tokens)[:, answer_start - 1 : answer_end - 1]
            predicted = logits.argmax(dim=-1)
            targets = tokens[:, answer_start:answer_end]
            # Vectorized: check all 81 answer digits match per puzzle
            exact_correct += (predicted == targets).all(dim=1).sum().item()
            total += tokens.size(0)
        self.model.train()
        return exact_correct / total if total > 0 else 0.0

    def train(self, sudoku_mode: bool = False):
        import numpy as np

        accuracy = np.nan
        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs, desc="Training")

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss_mask = None
                if sudoku_mode:
                    tokens = batch["tokens"]
                    loss_mask = self._get_sudoku_loss_mask(tokens.size(0), tokens.size(1))
                loss = self.training_step(batch, loss_mask)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break
            if sudoku_mode:
                accuracy = self.evaluate_sudoku_accuracy()
                sample_puzzle = ".358.47.2.....71...4.....9.......3...........8..53.....5.4...1..9..2...31.2.7.4.8"
                #935814762286957134741263895519642387623798541874531926357486219498125673162379458
                sample_text = self.sampler.sample(
                    sample_puzzle + "|",
                    max_tokens_generated=81,
                    temperature=0,
                )
            else:
                accuracy = self.evaluate()
                sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

    def save_model(self, filepath: str, tokenizer_config: dict | None = None):
        """Save model checkpoint including model state, optimizer state, and training progress."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "model_config": {
                "d_model": self.model.cfg.d_model,
                "n_heads": self.model.cfg.n_heads,
                "d_head": self.model.cfg.d_head,
                "d_mlp": self.model.cfg.d_mlp,
                "n_layers": self.model.cfg.n_layers,
                "n_ctx": self.model.cfg.n_ctx,
                "d_vocab": self.model.cfg.d_vocab,
            },
        }
        if tokenizer_config is not None:
            checkpoint["tokenizer_config"] = tokenizer_config
        t.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, model: DemoTransformer) -> dict:
        """Load model checkpoint and return the checkpoint dict."""
        checkpoint = t.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {filepath}")
        return checkpoint
