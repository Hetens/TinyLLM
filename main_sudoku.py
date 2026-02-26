"""Training entry point for the Sudoku Q&A model."""
import os
import torch as t
import datasets
from torch.utils.data import DataLoader, Dataset

from config import Config, TransformerTrainingArgs, device
from transformer import DemoTransformer
from trainer import TransformerTrainer
from sudoku_tokenizer import SudokuTokenizer


SUDOKU_SEQ_LEN = 163  # 81 (question) + 1 (separator) + 81 (answer)


class SudokuDataset(Dataset):
    """Dataset of tokenized sudoku question|answer pairs."""

    def __init__(self, hf_dataset, tokenizer: SudokuTokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, t.Tensor]:
        row = self.data[idx]
        text = row["question"] + "|" + row["answer"]
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return {"tokens": tokens}


def main():
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA capability: {t.cuda.get_device_capability()}")
        # Enable TF32 for matmuls on Ampere+ GPUs (gives ~2x speedup with minimal precision loss).
        # On your GTX 1050 (Pascal) this is a no-op but doesn't hurt.
        t.backends.cuda.matmul.allow_tf32 = True
        t.backends.cudnn.allow_tf32 = True
        # Enable cuDNN benchmark to auto-tune convolutions (helps with fixed-size inputs)
        t.backends.cudnn.benchmark = True

    tokenizer = SudokuTokenizer()

    model_cfg = Config.sudoku_7m()
    model = DemoTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created DemoTransformer with {n_params / 1e6:.2f}M parameters")

    # Batch size 8 is better for your GTX 1050 - higher batch sizes increase
    # compute per step without proportional throughput gain on a small GPU.
    args = TransformerTrainingArgs(
        batch_size=8,
        epochs=10,
        max_steps_per_epoch=2000,
        lr=1e-3,
        weight_decay=1e-2,
        wandb_project="sudoku-qna",
    )

    print("Loading sudoku-extreme dataset...")
    train_hf = datasets.load_dataset("sapientinc/sudoku-extreme", split="train")
    test_hf = datasets.load_dataset("sapientinc/sudoku-extreme", split="test")

    train_dataset = SudokuDataset(train_hf, tokenizer)
    test_dataset = SudokuDataset(test_hf, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Starting training...")
    trainer = TransformerTrainer(
        args=args,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
    )
    trainer.train(sudoku_mode=True)

    os.makedirs("./saved_models", exist_ok=True)
    checkpoint_path = "./saved_models/sudoku_checkpoint.pt"
    trainer.save_model(checkpoint_path, tokenizer_config={"vocab": SudokuTokenizer.VOCAB})


def run_sudoku_inference(checkpoint_path: str = "./saved_models/sudoku_checkpoint.pt"):
    """Load sudoku checkpoint and run inference on a puzzle."""
    tokenizer = SudokuTokenizer()
    checkpoint = t.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    cfg = Config(
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        d_head=model_config["d_head"],
        d_mlp=model_config["d_mlp"],
        n_layers=model_config["n_layers"],
        n_ctx=model_config["n_ctx"],
        d_vocab=model_config["d_vocab"],
    )
    model = DemoTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from sampler import TransformerSampler

    sampler = TransformerSampler(model, tokenizer)

    print("\n" + "=" * 50)
    print("SUDOKU INFERENCE")
    print("Enter an 81-character puzzle (digits 1-9 and . for empty), or 'quit'.")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("Puzzle: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if len(prompt) != 81:
                print("Puzzle must be exactly 81 characters.")
                continue

            full_prompt = prompt + "|"
            solution = sampler.sample(
                full_prompt,
                max_tokens_generated=81,
                temperature=0,
                verbose=False,
            )
            answer = solution[len(full_prompt) :]
            print(f"Solution: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        path = sys.argv[2] if len(sys.argv) > 2 else "./saved_models/sudoku_checkpoint.pt"
        run_sudoku_inference(path)
    else:
        main()
