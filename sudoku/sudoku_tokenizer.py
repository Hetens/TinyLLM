"""Minimal tokenizer for the sudoku-extreme dataset.

Vocabulary (11 tokens):
- '1'–'9': IDs 0–8 (digits)
- '.': ID 9 (empty cell)
- '|': ID 10 (question–answer separator)
"""

import torch


class SudokuTokenizer:
    """Character-level tokenizer for sudoku puzzles and solutions."""

    VOCAB = {str(i): i for i in range(1, 10)}  # '1'->0, ..., '9'->8
    VOCAB["."] = 9
    VOCAB["|"] = 10

    ID_TO_CHAR = {v: k for k, v in VOCAB.items()}

    def __init__(self):
        self.pad_token_id = 10  # Use separator as pad (or could add dedicated pad)
        self.eos_token_id = 10  # Separator marks end of question / start of answer

    @property
    def vocab_size(self) -> int:
        return 11

    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        """Encode text to token IDs. Raises ValueError for invalid characters."""
        ids = []
        for c in text:
            if c not in self.VOCAB:
                raise ValueError(f"Invalid character '{c}' for sudoku tokenizer")
            ids.append(self.VOCAB[c])

        if return_tensors == "pt":
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(
        self,
        ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if i in self.ID_TO_CHAR:
                chars.append(self.ID_TO_CHAR[i])
        return "".join(chars)
