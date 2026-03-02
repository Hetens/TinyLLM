"""Utilities for Sudoku visualization and analysis."""

from dataclasses import dataclass


@dataclass
class SudokuMetrics:
    """Metrics from analyzing a Sudoku puzzle."""

    num_givens: int = 0
    num_singles: int = 0
    num_hidden_singles: int = 0
    num_naked_pairs: int = 0
    num_hidden_pairs: int = 0
    num_pointing_pairs_triples: int = 0
    num_box_line_intersections: int = 0
    num_guesses: int = 0
    num_backtracks: int = 0
    difficulty: str = "Unknown"
    solution: str | None = None  # 81-char solution string when solved

    def __str__(self) -> str:
        lines = [
            f"Number of Givens: {self.num_givens}",
            f"Number of Singles: {self.num_singles}",
            f"Number of Hidden Singles: {self.num_hidden_singles}",
            f"Number of Naked Pairs: {self.num_naked_pairs}",
            f"Number of Hidden Pairs: {self.num_hidden_pairs}",
            f"Number of Pointing Pairs/Triples: {self.num_pointing_pairs_triples}",
            f"Number of Box/Line Intersections: {self.num_box_line_intersections}",
            f"Number of Guesses: {self.num_guesses}",
            f"Number of Backtracks: {self.num_backtracks}",
            f"Difficulty: {self.difficulty}",
        ]
        return "\n".join(lines)


def sudoku_to_grid(s: str) -> str:
    """Convert a flat string representation of a Sudoku to a visual grid.

    The string is read left-to-right, row by row (81 chars for 9x9).
    - '.' or '0' = empty cell
    - '1'-'9' = filled cell

    Example:
        >>> s = ".358.47.2.....71...4.....9.......3...........8..53.....5.4...1..9..2...31.2.7.4.8"
        >>> print(sudoku_to_grid(s))
    """
    s = s.strip().replace(" ", "")
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters for 9x9 Sudoku, got {len(s)}")

    # Normalize: treat '0' as empty like '.'
    def cell(c: str) -> str:
        return " " if c in ".0" else c

    rows = []
    for r in range(9):
        row_chars = [cell(s[r * 9 + c]) for c in range(9)]
        # Format: " a b c | d e f | g h i " (3x3 box separators)
        parts = [
            " ".join(row_chars[0:3]),
            " ".join(row_chars[3:6]),
            " ".join(row_chars[6:9]),
        ]
        rows.append(" " + " | ".join(parts) + " ")

    sep = "+-------+-------+-------+"
    out = [sep]
    for i in range(3):
        out.extend(rows[i * 3 : (i + 1) * 3])
        if i < 2:
            out.append(sep)
    out.append(sep)
    return "\n".join(out)


def print_sudoku(s: str) -> None:
    """Print a Sudoku string as a visual grid."""
    print(sudoku_to_grid(s))


def sudoku_metrics(s: str, *, guess_order: str = "fewest") -> SudokuMetrics:
    """Analyze a Sudoku puzzle string and return solving metrics.

    Technique order matches QQWing (https://github.com/stephenostermiller/qqwing):
    1. Naked single, 2. Hidden single (box→row→col), 3. Naked pairs,
    4. Pointing pairs, 5. Box/line, 6. Hidden pairs.

    The string is 81 chars (puzzle only) or puzzle|solution format.

    guess_order: When backtracking is needed, how to pick the cell to guess.
      - "fewest": pick cell with fewest candidates (matches QQWing)
      - "row_major": pick first empty cell in row-major order
    """
    s = s.strip().replace(" ", "")
    if "|" in s:
        s = s.split("|")[0]
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters for 9x9 Sudoku, got {len(s)}")

    m = SudokuMetrics()
    m.num_givens = sum(1 for c in s if c in "123456789")

    # Build grid: 0 = empty, 1-9 = filled
    grid = [[0] * 9 for _ in range(9)]
    for i, c in enumerate(s):
        if c in "123456789":
            grid[i // 9][i % 9] = int(c)

    # Candidate sets: candidates[r][c] = set of possible digits
    def init_candidates() -> list[list[set[int]]]:
        cand = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
        for r in range(9):
            for c in range(9):
                if grid[r][c]:
                    cand[r][c] = set()
                    for rr in range(9):
                        if rr != r:
                            cand[rr][c].discard(grid[r][c])
                    for cc in range(9):
                        if cc != c:
                            cand[r][cc].discard(grid[r][c])
                    br, bc = 3 * (r // 3), 3 * (c // 3)
                    for rr in range(br, br + 3):
                        for cc in range(bc, bc + 3):
                            if (rr, cc) != (r, c):
                                cand[rr][cc].discard(grid[r][c])
        for r in range(9):
            for c in range(9):
                if grid[r][c]:
                    cand[r][c] = {grid[r][c]}
        return cand

    def peers(r: int, c: int) -> set[tuple[int, int]]:
        out = set()
        for i in range(9):
            if i != r:
                out.add((i, c))
            if i != c:
                out.add((r, i))
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if (rr, cc) != (r, c):
                    out.add((rr, cc))
        return out

    def row_cells(r: int) -> list[tuple[int, int]]:
        return [(r, c) for c in range(9)]

    def col_cells(c: int) -> list[tuple[int, int]]:
        return [(r, c) for r in range(9)]

    def box_cells(box_r: int, box_c: int) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(box_r * 3, box_r * 3 + 3)
            for c in range(box_c * 3, box_c * 3 + 3)
        ]

    def set_cell(r: int, c: int, val: int) -> None:
        grid[r][c] = val
        cand[r][c] = {val}
        for (rr, cc) in peers(r, c):
            cand[rr][cc].discard(val)

    # Solve with technique tracking
    cand = init_candidates()
    used_advanced = False
    used_guessing = False

    def try_naked_single() -> bool:
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0 and len(cand[r][c]) == 1:
                    val = next(iter(cand[r][c]))
                    set_cell(r, c, val)
                    m.num_singles += 1
                    return True
        return False

    def try_hidden_single() -> bool:
        # QQWing order: section (box) first, then row, then column
        for unit_type, units in [
            ("box", [(i, box_cells(i // 3, i % 3)) for i in range(9)]),
            ("row", [(r, row_cells(r)) for r in range(9)]),
            ("col", [(c, col_cells(c)) for c in range(9)]),
        ]:
            for _, unit in units:
                for d in range(1, 10):
                    cells_with_d = [(r, c) for (r, c) in unit if grid[r][c] == 0 and d in cand[r][c]]
                    if len(cells_with_d) == 1:
                        r, c = cells_with_d[0]
                        set_cell(r, c, d)
                        m.num_hidden_singles += 1
                        return True
        return False

    def try_naked_pairs() -> bool:
        for unit_type, units in [
            ("row", [row_cells(r) for r in range(9)]),
            ("col", [col_cells(c) for c in range(9)]),
            ("box", [box_cells(i // 3, i % 3) for i in range(9)]),
        ]:
            for unit in units:
                empty = [(r, c) for (r, c) in unit if grid[r][c] == 0 and len(cand[r][c]) == 2]
                for i, (r1, c1) in enumerate(empty):
                    for (r2, c2) in empty[i + 1 :]:
                        if cand[r1][c1] == cand[r2][c2]:
                            pair = cand[r1][c1]
                            # Naked pair requires that EXACTLY two cells in the unit
                            # share this two-digit candidate set. If more than two
                            # cells have the same pair, this is not a valid naked pair.
                            same_pair_count = sum(
                                1 for (rr, cc) in empty if cand[rr][cc] == pair
                            )
                            if same_pair_count != 2:
                                continue
                            a, b = pair
                            others = [(r, c) for (r, c) in unit if (r, c) not in ((r1, c1), (r2, c2))]
                            if any(a in cand[r][c] or b in cand[r][c] for (r, c) in others):
                                changed = False
                                for (r, c) in others:
                                    if a in cand[r][c]:
                                        cand[r][c].discard(a)
                                        changed = True
                                    if b in cand[r][c]:
                                        cand[r][c].discard(b)
                                        changed = True
                                if changed:
                                    m.num_naked_pairs += 1
                                    return True
        return False

    def try_hidden_pairs() -> bool:
        for unit in (
            [row_cells(r) for r in range(9)]
            + [col_cells(c) for c in range(9)]
            + [box_cells(i // 3, i % 3) for i in range(9)]
        ):
            empty = [(r, c) for (r, c) in unit if grid[r][c] == 0]
            for d1 in range(1, 10):
                cells1 = [(r, c) for (r, c) in empty if d1 in cand[r][c]]
                if len(cells1) != 2:
                    continue
                for d2 in range(d1 + 1, 10):
                    cells2 = [(r, c) for (r, c) in empty if d2 in cand[r][c]]
                    if set(cells1) == set(cells2):
                        (r1, c1), (r2, c2) = cells1
                        extra = cand[r1][c1] | cand[r2][c2] - {d1, d2}
                        if extra:
                            changed = False
                            for (r, c) in cells1:
                                for x in extra:
                                    if x in cand[r][c]:
                                        cand[r][c].discard(x)
                                        changed = True
                            if changed:
                                m.num_hidden_pairs += 1
                                return True
        return False

    def try_pointing_pairs() -> bool:
        for br in range(3):
            for bc in range(3):
                box = box_cells(br, bc)
                for d in range(1, 10):
                    cells_with_d = [(r, c) for (r, c) in box if grid[r][c] == 0 and d in cand[r][c]]
                    if len(cells_with_d) < 2:
                        continue
                    rows = set(r for (r, _) in cells_with_d)
                    cols = set(c for (_, c) in cells_with_d)
                    if len(rows) == 1:
                        r = next(iter(rows))
                        to_elim = [(r, c) for c in range(9) if (r, c) not in box]
                        if any(d in cand[r][c] for (r, c) in to_elim):
                            for (rr, cc) in to_elim:
                                cand[rr][cc].discard(d)
                            m.num_pointing_pairs_triples += 1
                            return True
                    if len(cols) == 1:
                        c = next(iter(cols))
                        to_elim = [(r, c) for r in range(9) if (r, c) not in box]
                        if any(d in cand[r][c] for (r, c) in to_elim):
                            for (rr, cc) in to_elim:
                                cand[rr][cc].discard(d)
                            m.num_pointing_pairs_triples += 1
                            return True
        return False

    def try_box_line() -> bool:
        for idx in range(9):
            for line_type, unit in [
                ("row", row_cells(idx)),
                ("col", col_cells(idx)),
            ]:
                for d in range(1, 10):
                    cells_with_d = [(r, c) for (r, c) in unit if grid[r][c] == 0 and d in cand[r][c]]
                    if len(cells_with_d) < 2:
                        continue
                    boxes = set((r // 3, c // 3) for (r, c) in cells_with_d)
                    if len(boxes) == 1:
                        br, bc = next(iter(boxes))
                        box = set(box_cells(br, bc))
                        to_elim = [(r, c) for (r, c) in box if (r, c) not in unit]
                        if any(d in cand[r][c] for (r, c) in to_elim):
                            for (rr, cc) in to_elim:
                                cand[rr][cc].discard(d)
                            m.num_box_line_intersections += 1
                            return True
        return False

    def is_solved() -> bool:
        return all(grid[r][c] for r in range(9) for c in range(9))

    def has_empty_with_no_candidates() -> bool:
        return any(
            grid[r][c] == 0 and len(cand[r][c]) == 0
            for r in range(9)
            for c in range(9)
        )

    def _solve_with_backtrack() -> bool:
        nonlocal cand, grid
        while not is_solved():
            if has_empty_with_no_candidates():
                return False
            if try_naked_single() or try_hidden_single():
                continue
            if try_naked_pairs() or try_pointing_pairs() or try_box_line() or try_hidden_pairs():
                continue
            best_r, best_c = -1, -1
            best_size = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] == 0 and len(cand[r][c]) < best_size:
                        best_size = len(cand[r][c])
                        best_r, best_c = r, c
            if best_r < 0:
                return True
            best_r, best_c = -1, -1
            if guess_order == "row_major":
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] == 0 and len(cand[r][c]) > 0:
                            best_r, best_c = r, c
                            break
                    if best_r >= 0:
                        break
            else:
                best_size = 10
                for r in range(9):
                    for c in range(9):
                        if grid[r][c] == 0 and len(cand[r][c]) < best_size:
                            best_size = len(cand[r][c])
                            best_r, best_c = r, c
            save_grid = [row[:] for row in grid]
            save_cand = [[c.copy() for c in row] for row in cand]
            for val in list(cand[best_r][best_c]):
                m.num_guesses += 1
                set_cell(best_r, best_c, val)
                if _solve_with_backtrack():
                    return True
                # QQWing counts 2 backtracks per failed guess (rollbackRound called twice)
                m.num_backtracks += 2
                grid = [row[:] for row in save_grid]
                cand = [[c.copy() for c in row] for row in save_cand]
            return False
        return True

    # Main solve loop: QQWing order (naked→hidden→naked pairs→pointing→box/line→hidden pairs)
    while not is_solved():
        if try_naked_single():
            continue
        if try_hidden_single():
            continue
        if try_naked_pairs():
            used_advanced = True
            continue
        if try_pointing_pairs():
            used_advanced = True
            continue
        if try_box_line():
            used_advanced = True
            continue
        if try_hidden_pairs():
            used_advanced = True
            continue

        # No logical move: must guess (backtrack)
        best_r, best_c = -1, -1
        if guess_order == "row_major":
            for r in range(9):
                for c in range(9):
                    if grid[r][c] == 0 and len(cand[r][c]) > 0:
                        best_r, best_c = r, c
                        break
                if best_r >= 0:
                    break
        else:
            best_size = 10
            for r in range(9):
                for c in range(9):
                    if grid[r][c] == 0 and len(cand[r][c]) < best_size:
                        best_size = len(cand[r][c])
                        best_r, best_c = r, c

        if best_r < 0:
            break  # no empty cell (solved) or unsolvable

        used_guessing = True

        save_grid = [row[:] for row in grid]
        save_cand = [[c.copy() for c in row] for row in cand]
        choices = list(cand[best_r][best_c])

        for val in choices:
            m.num_guesses += 1
            set_cell(best_r, best_c, val)
            if _solve_with_backtrack():
                m.solution = "".join(str(grid[r][c]) for r in range(9) for c in range(9))
                m.difficulty = "Expert"
                return m
            # QQWing counts 2 backtracks per failed guess (rollbackRound called twice)
            m.num_backtracks += 2
            grid = [row[:] for row in save_grid]
            cand = [[c.copy() for c in row] for row in save_cand]
        break  # all choices failed (unsolvable)

    # Difficulty: QQWing-style (Expert > Intermediate > Easy > Simple)
    if m.num_guesses > 0:
        m.difficulty = "Expert"
    elif m.num_box_line_intersections > 0 or m.num_pointing_pairs_triples > 0:
        m.difficulty = "Intermediate"
    elif m.num_naked_pairs > 0 or m.num_hidden_pairs > 0:
        m.difficulty = "Intermediate"
    elif m.num_hidden_singles > 0:
        m.difficulty = "Easy"
    elif m.num_singles > 0:
        m.difficulty = "Simple"
    else:
        m.difficulty = "Unknown"

    if all(grid[r][c] for r in range(9) for c in range(9)):
        m.solution = "".join(str(grid[r][c]) for r in range(9) for c in range(9))
    return m


def solve_sudoku(s: str) -> str:
    """Return the solution of a Sudoku puzzle as an 81-character string.

    Uses constraint propagation (naked + hidden singles) and backtracking.
    Same format as input: digits 1-9 for filled cells, left-to-right row by row.
    Raises ValueError if the puzzle has no solution.
    """
    s = s.strip().replace(" ", "")
    if "|" in s:
        s = s.split("|")[0]
    if len(s) != 81:
        raise ValueError(f"Expected 81 characters for 9x9 Sudoku, got {len(s)}")

    grid = [[0] * 9 for _ in range(9)]
    for i, c in enumerate(s):
        if c in "123456789":
            grid[i // 9][i % 9] = int(c)

    cand = [[set(range(1, 10)) for _ in range(9)] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                v = grid[r][c]
                cand[r][c] = set()
                for i in range(9):
                    cand[r][i].discard(v)
                    cand[i][c].discard(v)
                br, bc = 3 * (r // 3), 3 * (c // 3)
                for i in range(3):
                    for j in range(3):
                        cand[br + i][bc + j].discard(v)

    def set_cell(r: int, c: int, v: int) -> None:
        grid[r][c] = v
        cand[r][c] = set()
        for i in range(9):
            cand[r][i].discard(v)
            cand[i][c].discard(v)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(3):
            for j in range(3):
                cand[br + i][bc + j].discard(v)

    def propagate() -> bool:
        """Apply naked and hidden singles until no progress."""
        while True:
            changed = False
            for r in range(9):
                for c in range(9):
                    if grid[r][c] == 0 and len(cand[r][c]) == 1:
                        set_cell(r, c, next(iter(cand[r][c])))
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                for unit in (
                    [(r, [(r, c) for c in range(9)]) for r in range(9)]
                    + [(c, [(r, c) for r in range(9)]) for c in range(9)]
                    + [
                        (b, [(3 * (b // 3) + i, 3 * (b % 3) + j) for i in range(3) for j in range(3)])
                        for b in range(9)
                    ]
                ):
                    for d in range(1, 10):
                        cells = [(r, c) for r, c in unit[1] if grid[r][c] == 0 and d in cand[r][c]]
                        if len(cells) == 1:
                            r, c = cells[0]
                            set_cell(r, c, d)
                            changed = True
                            break
                    if changed:
                        break
            if not changed:
                break
        return True

    def solve() -> bool:
        propagate()
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    if len(cand[r][c]) == 0:
                        return False
                    for v in list(cand[r][c]):
                        save_grid = [row[:] for row in grid]
                        save_cand = [[x.copy() for x in row] for row in cand]
                        set_cell(r, c, v)
                        if solve():
                            return True
                        for rr in range(9):
                            for cc in range(9):
                                grid[rr][cc] = save_grid[rr][cc]
                                cand[rr][cc] = save_cand[rr][cc].copy()
                    return False
        return True

    if not solve():
        raise ValueError("Puzzle has no solution")
    return "".join(str(grid[r][c]) for r in range(9) for c in range(9))

puzzle = "2.439....79....4.8....8....51............8...4..27..5.8.3.4.71..5973............5"
solution = solve_sudoku(puzzle)
print(solution)  # 583427169974136528216859374792364851351298746648715293865971432137642985429583617
print(sudoku_to_grid(solution))
print(sudoku_metrics(puzzle))