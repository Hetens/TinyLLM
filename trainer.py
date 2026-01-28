"""Training utilities for the DemoTransformer."""
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from tqdm import tqdm

from config import Config, TransformerTrainingArgs, device
from transformer import DemoTransformer
from sampler import TransformerSampler


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

    def get_log_probs(
        self, logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
    ) -> Float[Tensor, "batch posn-1"]:
        log_probs = logits.log_softmax(dim=-1)
        # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
        log_probs_for_tokens = (
            log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        )
        return log_probs_for_tokens

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -self.get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        totally_correct, total_samples = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            tokens = batch["tokens"].to(device)
            logits: Tensor = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            totally_correct += (predicted_tokens == tokens[:, 1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        accuracy = totally_correct / total_samples
        self.model.train()
        return accuracy

    def train(self):
        import numpy as np

        accuracy = np.nan
        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs, desc="Training")

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break
            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

    def save_model(self, filepath: str):
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
        t.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, model: DemoTransformer) -> dict:
        """Load model checkpoint and return the checkpoint dict."""
        checkpoint = t.load(filepath, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {filepath}")
        return checkpoint
