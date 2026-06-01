"""A small configurable terminal maze game.

Run with:
    python3 maze_game.py

Move with arrow keys or WASD. Reach the exit square.
"""

from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Maze configuration
# ---------------------------------------------------------------------------
# Change these values to resize the maze. The default starts at 16x16.
DEFAULT_WIDTH = 35
DEFAULT_HEIGHT = 35

START = (0, 0)

# Use None for the bottom-right cell, or set a fixed coordinate like (15, 15).
EXIT: tuple[int, int] | None = None

# Wall placement block:
# Each tuple is (x1, y1, x2, y2). Horizontal and vertical lines are supported.
# Coordinates are grid cells, starting at (0, 0) in the top-left corner.
# Keep gaps by splitting a wall into shorter line segments.
WALL_LINES = [
    (0, 3, 29, 3),
    (32, 3, 34, 3),
    (0, 7, 2, 7),
    (5, 7, 34, 7),
    (0, 11, 29, 11),
    (32, 11, 34, 11),
    (0, 15, 2, 15),
    (5, 15, 34, 15),
    (0, 19, 29, 19),
    (32, 19, 34, 19),
    (0, 23, 2, 23),
    (5, 23, 34, 23),
    (0, 27, 29, 27),
    (32, 27, 34, 27),
    (0, 31, 2, 31),
    (5, 31, 34, 31),
    (10, 1, 10, 2),
    (21, 5, 21, 6),
    (13, 9, 13, 10),
    (25, 13, 25, 14),
    (8, 17, 8, 18),
    (19, 21, 19, 22),
    (12, 25, 12, 26),
    (24, 29, 24, 30),
    (5, 4, 5, 5),
    (29, 8, 29, 9),
    (6, 12, 6, 13),
    (28, 16, 28, 17),
    (7, 20, 7, 21),
    (27, 24, 27, 25),
    (9, 28, 9, 29),
    (17, 32, 17, 33),
]

# Optional one-off wall cells can go here.
EXTRA_WALLS = {
    (2, 2),
    (15, 1),
    (16, 1),
    (17, 1),
    (11, 5),
    (12, 5),
    (23, 9),
    (24, 9),
    (14, 13),
    (15, 13),
    (20, 17),
    (21, 17),
    (11, 21),
    (12, 21),
    (22, 25),
    (23, 25),
    (14, 29),
    (15, 29),
    (26, 33),
    (27, 33),
}

SPIKEY_CELLS = {
    (3, 3),
    (30, 3),
    (3, 7)
}

# ---------------------------------------------------------------------------
# Game code
# ---------------------------------------------------------------------------
WALL_TILE = "##"
FLOOR_TILE = "  "
SPIKE_TILE = "><"
PLAYER_TILE = "@@"
EXIT_TILE = "[]"


@dataclass(frozen=True)
class Maze:
    width: int
    height: int
    start: tuple[int, int]
    exit: tuple[int, int]
    spikes: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_open(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and cell not in self.walls


def cells_from_line(line: tuple[int, int, int, int]) -> Iterable[tuple[int, int]]:
    x1, y1, x2, y2 = line

    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step, step):
            yield (x1, y)
        return

    if y1 == y2:
        step = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step, step):
            yield (x, y1)
        return

    raise ValueError(f"Wall line must be horizontal or vertical: {line}")


def resolve_exit(width: int, height: int) -> tuple[int, int]:
    if EXIT is None:
        return (width - 1, height - 1)

    if not (0 <= EXIT[0] < width and 0 <= EXIT[1] < height):
        raise ValueError(f"EXIT {EXIT} is outside a {width}x{height} maze.")

    return EXIT


def build_walls(
    width: int,
    height: int,
    start: tuple[int, int],
    exit_cell: tuple[int, int],
    spikes: frozenset[tuple[int, int]] = frozenset(SPIKEY_CELLS),
) -> frozenset[tuple[int, int]]:
    walls = set(EXTRA_WALLS)

    for line in WALL_LINES:
        walls.update(cells_from_line(line))

    walls = {cell for cell in walls if 0 <= cell[0] < width and 0 <= cell[1] < height}
    walls.discard(start)
    walls.discard(exit_cell)
    spikes = set(spikes)
    spikes.discard(start)
    spikes.discard(exit_cell)
    return frozenset(walls)


class MazeGame:
    def __init__(self, maze: Maze, screen: curses.window) -> None:
        self.maze = maze
        self.screen = screen
        self.player = maze.start
        self.moves = 0
        self.won = False

    def run(self) -> None:
        self.setup_screen()

        while True:
            self.draw()
            key = self.screen.getch()

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                self.reset()
            elif key in (curses.KEY_UP, ord("w"), ord("W")):
                self.try_move(0, -1)
            elif key in (curses.KEY_DOWN, ord("s"), ord("S")):
                self.try_move(0, 1)
            elif key in (curses.KEY_LEFT, ord("a"), ord("A")):
                self.try_move(-1, 0)
            elif key in (curses.KEY_RIGHT, ord("d"), ord("D")):
                self.try_move(1, 0)
            elif self.player in self.maze.spikes:
                self.reset()  # Reset the game if the player steps on a spike.
    def setup_screen(self) -> None:
        self.screen.keypad(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass

        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
                default_background = -1
            except curses.error:
                default_background = curses.COLOR_BLACK
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_BLUE, default_background)
            curses.init_pair(3, curses.COLOR_GREEN, default_background)
            curses.init_pair(4, curses.COLOR_WHITE, default_background)

    def reset(self) -> None:
        self.player = self.maze.start
        self.moves = 0
        self.won = False


    def try_move(self, dx: int, dy: int) -> None:
        if self.won:
            return

        x, y = self.player
        next_cell = (x + dx, y + dy)

        if not self.maze.is_open(next_cell):
            return

        self.player = next_cell
        self.moves += 1

        if self.player == self.maze.exit:
            self.won = True

    def draw(self) -> None:
        self.screen.erase()

        if self.terminal_too_small():
            self.draw_too_small()
        else:
            self.draw_grid()
            self.draw_status()

        self.screen.refresh()

    def terminal_too_small(self) -> bool:
        rows, cols = self.screen.getmaxyx()
        required_rows = self.maze.height + 4
        required_cols = self.maze.width * 2 + 2
        return rows < required_rows or cols < required_cols

    def draw_too_small(self) -> None:
        rows, cols = self.screen.getmaxyx()
        required_rows = self.maze.height + 4
        required_cols = self.maze.width * 2 + 2
        lines = [
            "Terminal is too small for this maze.",
            f"Current: {cols} cols x {rows} rows",
            f"Needed:  {required_cols} cols x {required_rows} rows",
            "Resize the terminal, or press Q/Esc to quit.",
        ]
        for row, line in enumerate(lines):
            self.safe_addstr(row, 0, line[: max(0, cols - 1)])

    def draw_grid(self) -> None:
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                cell = (x, y)
                tile = FLOOR_TILE
                attr = self.color(4)

                if cell in self.maze.walls:
                    tile = WALL_TILE
                    attr = self.color(1) | curses.A_BOLD
                elif cell == self.maze.exit:
                    tile = EXIT_TILE
                    attr = self.color(3) | curses.A_BOLD

                if cell == self.player:
                    tile = PLAYER_TILE
                    attr = self.color(2) | curses.A_BOLD

                self.safe_addstr(y, x * 2, tile, attr)

    def draw_status(self) -> None:
        row = self.maze.height + 1
        if self.won:
            message = f"Exit reached in {self.moves} moves. Press R to restart, Q/Esc to quit."
        else:
            message = f"Moves: {self.moves}   Use arrows/WASD. Press R to restart, Q/Esc to quit."

        self.safe_addstr(row, 0, message)
        self.safe_addstr(row + 1, 0, f"Start: {self.maze.start}   Exit: {self.maze.exit}")

    def safe_addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        try:
            self.screen.addstr(row, col, text, attr)
        except curses.error:
            pass

    def color(self, pair_number: int) -> int:
        if not curses.has_colors():
            return 0
        return curses.color_pair(pair_number)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a simple configurable maze.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.width < 2 or args.height < 2:
        raise SystemExit("Maze width and height must both be at least 2.")

    if not (0 <= START[0] < args.width and 0 <= START[1] < args.height):
        raise SystemExit(f"START {START} is outside a {args.width}x{args.height} maze.")

    try:
        exit_cell = resolve_exit(args.width, args.height)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    maze = Maze(
        width=args.width,
        height=args.height,
        start=START,
        exit=exit_cell,
        walls=build_walls(args.width, args.height, START, exit_cell),
    )
    curses.wrapper(lambda screen: MazeGame(maze, screen).run())


if __name__ == "__main__":
    main()
