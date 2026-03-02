"""Convenience entry point for Sudoku training. Delegates to sudoku.main_sudoku.

Run: python main_sudoku.py [--inference [checkpoint_path]]
"""

if __name__ == "__main__":
    import sys
    from sudoku.main_sudoku import main, run_sudoku_inference

    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        path = sys.argv[2] if len(sys.argv) > 2 else "./saved_models/sudoku_checkpoint.pt"
        run_sudoku_inference(path)
    else:
        main()
