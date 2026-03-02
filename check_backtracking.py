"""
Check whether any Sudoku in the test set requires backtracking when solved
with util.sudoku_metrics (logical techniques + backtracking if needed).

Uses: datasets/sudoku-test-data.npy, inspect_samples (parsing), util.sudoku_metrics.
Run from project root: python check_backtracking.py
"""

from pathlib import Path
import numpy as np

from inspect_samples import parse_sudoku_example, sudoku_grid_from_example, sudoku_example_to_puzzle_string
from util import sudoku_metrics


def main():
    data_dir = Path(__file__).resolve().parent / "datasets"
    npy_files = list(data_dir.glob("sudoku*test*.npy"))
    if not npy_files:
        print("No sudoku test .npy file found in datasets/")
        return

    path = npy_files[0]
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n_total = len(arr)
    print(f"Loaded {path.name}: {n_total} puzzles")
    print("Solving each with sudoku_metrics (logical techniques then backtracking if needed)...")
    print()

    need_backtrack = []
    for idx in range(n_total):
        raw = arr[idx]
        puzzle = sudoku_example_to_puzzle_string(raw)
        m = sudoku_metrics(puzzle)
        if m.num_guesses > 0 or m.num_backtracks > 0:
            need_backtrack.append((idx, m))
        if (idx + 1) % 5000 == 0 or idx == 0:
            print(f"  Processed {idx + 1}/{n_total} ...")

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Total puzzles: {n_total}")
    print(f"Puzzles that required backtracking (num_guesses > 0 or num_backtracks > 0): {len(need_backtrack)}")
    if need_backtrack:
        print()
        print("First 5 examples (index, num_guesses, num_backtracks, difficulty):")
        for idx, m in need_backtrack[:5]:
            print(f"  Index {idx}: guesses={m.num_guesses}, backtracks={m.num_backtracks}, difficulty={m.difficulty}")
        if len(need_backtrack) > 5:
            print(f"  ... and {len(need_backtrack) - 5} more")
    else:
        print("No puzzle in the test set required backtracking; all were solved by logical techniques only.")


if __name__ == "__main__":
    main()
