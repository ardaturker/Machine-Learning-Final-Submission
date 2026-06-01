"""
main.py — Q-Learning vs SARSA on all three mazes.

Place this file in the same folder as maze_1.py, maze_2.py, maze_3.py.
Run with:
    python3 main.py
"""

from __future__ import annotations

import random
from collections import deque

import maze_1
import maze_2
import maze_3


# ---------------------------------------------------------------------------
# Per-maze configuration
# Set "enabled": False to skip a maze entirely.
# ---------------------------------------------------------------------------
MAZE_CONFIGS = {
    "maze_1": {
        "enabled":          True,
        "run_q_learning":   False,
        "run_sarsa":        True,
        "episodes":         10000,
        # Learning
        "alpha":            0.153,
        "gamma":            0.995,
        "epsilon":          1.0,
        "epsilon_decay":    0.995,
        "min_epsilon":      0.05,
        # Rewards
        "reward_exit":      100,
        "reward_step":      -1,
        "reward_wall":      -5,
    },
    "maze_2": {
        "enabled":          True,
        "run_q_learning":   False,
        "run_sarsa":        True,
        "episodes":         10000,
        "alpha":            0.153,
        "gamma":            0.995,
        "epsilon":          1.0,
        "epsilon_decay":    0.995,
        "min_epsilon":      0.05,
        "reward_exit":      100,
        "reward_step":      -1,
        "reward_wall":      -5,
    },
    "maze_3": {
        "enabled":          True,
        "run_q_learning":   False,
        "run_sarsa":        True,
        "episodes":         10000,
        "alpha":            0.153,
        "gamma":            0.995,
        "epsilon":          1.0,
        "epsilon_decay":    0.995,
        "min_epsilon":      0.05,
        "reward_exit":      100,
        "reward_step":      -1,
        "reward_wall":      -5,
    },
}

RANDOM_SEED = 1

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTIONS = {
    "up":    (0, -1),
    "down":  (0,  1),
    "left":  (-1, 0),
    "right": (1,  0),
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def build_maze(maze_module):
    """Build a Maze dataclass from any of the three maze modules."""
    w  = maze_module.DEFAULT_WIDTH
    h  = maze_module.DEFAULT_HEIGHT
    ex = maze_module.resolve_exit(w, h)
    return maze_module.Maze(
        width=w,
        height=h,
        start=maze_module.START,
        exit=ex,
        walls=maze_module.build_walls(w, h, maze_module.START, ex),
    )


def move(maze, state, action, cfg):
    """Apply action; return (next_state, reward, done)."""
    dx, dy = ACTIONS[action]
    x, y   = state
    ns     = (x + dx, y + dy)
    if not maze.is_open(ns):
        return state, cfg["reward_wall"], False
    if ns == maze.exit:
        return ns, cfg["reward_exit"], True
    return ns, cfg["reward_step"], False


def get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)


def best_action(q_table, state):
    """Greedy argmax over Q."""
    return max(ACTIONS, key=lambda a: get_q(q_table, state, a))


def choose_action(q_table, state, epsilon):
    """Epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.choice(list(ACTIONS))
    return best_action(q_table, state)


def extract_path(maze, q_table, cfg):
    """Follow the greedy policy from start; return the list of cells visited."""
    state     = maze.start
    path      = [state]
    max_steps = maze.width * maze.height * 4
    for _ in range(max_steps):
        ns, _, done = move(maze, state, best_action(q_table, state), cfg)
        if ns == state:
            break
        path.append(ns)
        state = ns
        if done:
            break
    return path


def shortest_path_bfs(maze, cfg):
    """BFS benchmark — returns the true shortest path length."""
    queue = deque([maze.start])
    dist  = {maze.start: 0}
    while queue:
        s = queue.popleft()
        if s == maze.exit:
            return dist[s]
        for a in ACTIONS:
            ns, _, _ = move(maze, s, a, cfg)
            if ns != s and ns not in dist:
                dist[ns] = dist[s] + 1
                queue.append(ns)
    return None


def evaluate(maze, q_table, cfg, n_episodes=200):
    """Run n_episodes with epsilon=0 (pure greedy). Returns (win_rate, avg_steps)."""
    max_steps = maze.width * maze.height * 4
    wins, steps_list = 0, []
    for _ in range(n_episodes):
        state = maze.start
        for step in range(1, max_steps + 1):
            next_state, _, done = move(maze, state, best_action(q_table, state), cfg)
            state = next_state
            if done:
                wins += 1
                steps_list.append(step)
                break
    win_rate  = wins / n_episodes
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else float("inf")
    return win_rate, avg_steps


def print_path(maze, path):
    """ASCII grid showing walls (##), path (..), start (S) and exit (E)."""
    path_cells = set(path)
    for y in range(maze.height):
        row = ""
        for x in range(maze.width):
            c = (x, y)
            if   c == maze.start: row += "S "
            elif c == maze.exit:  row += "E "
            elif c in maze.walls: row += "##"
            elif c in path_cells: row += ".."
            else:                 row += "  "
        print(row)


# ---------------------------------------------------------------------------
# Q-Learning  (off-policy)
# ---------------------------------------------------------------------------
def train_q_learning(maze, cfg, bfs_len, log_every=500):
    """
    Off-policy TD control.
    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    The next-state value uses the greedy maximum, independent of what the
    agent actually does next.
    Returns (q_table, first_exit_ep, first_optimal_ep).
    """
    q_table          = {}
    epsilon          = cfg["epsilon"]
    max_steps        = maze.width * maze.height * 4
    wins             = deque(maxlen=100)
    first_exit_ep    = None
    first_optimal_ep = None

    for ep in range(1, cfg["episodes"] + 1):
        state = maze.start
        won   = False

        for step in range(1, max_steps + 1):
            action              = choose_action(q_table, state, epsilon)
            next_state, r, done = move(maze, state, action, cfg)

            old_q       = get_q(q_table, state, action)
            next_best_q = max(get_q(q_table, next_state, a) for a in ACTIONS)
            q_table[(state, action)] = old_q + cfg["alpha"] * (
                r + cfg["gamma"] * next_best_q - old_q
            )

            state = next_state
            if done:
                won = True
                if first_exit_ep is None:
                    first_exit_ep = ep
                if first_optimal_ep is None and step == bfs_len:
                    first_optimal_ep = ep
                break

        wins.append(won)
        epsilon = max(cfg["min_epsilon"], epsilon * cfg["epsilon_decay"])

        if ep % log_every == 0:
            print(
                f"  [Q-Learning] episode={ep:>5}  "
                f"success={sum(wins)/len(wins):.0%}  "
                f"epsilon={epsilon:.3f}"
            )

    return q_table, first_exit_ep, first_optimal_ep


# ---------------------------------------------------------------------------
# SARSA  (on-policy)
# ---------------------------------------------------------------------------
def train_sarsa(maze, cfg, bfs_len, log_every=500):
    """
    On-policy TD control (SARSA).
    Update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    where a' is sampled from the current epsilon-greedy policy, not the
    greedy maximum.  The full quintuple (S, A, R, S', A') is used before
    advancing to the next step.
    Returns (q_table, first_exit_ep, first_optimal_ep).
    """
    q_table          = {}
    epsilon          = cfg["epsilon"]
    max_steps        = maze.width * maze.height * 4
    wins             = deque(maxlen=100)
    first_exit_ep    = None
    first_optimal_ep = None

    for ep in range(1, cfg["episodes"] + 1):
        state  = maze.start
        action = choose_action(q_table, state, epsilon)  # choose first A
        won    = False

        for step in range(1, max_steps + 1):
            next_state, r, done = move(maze, state, action, cfg)
            next_action         = choose_action(q_table, next_state, epsilon)

            old_q  = get_q(q_table, state, action)
            next_q = get_q(q_table, next_state, next_action)  # uses actual A'
            q_table[(state, action)] = old_q + cfg["alpha"] * (
                r + cfg["gamma"] * next_q - old_q
            )

            state  = next_state
            action = next_action   # carry A' forward — the SARSA quintuple
            if done:
                won = True
                if first_exit_ep is None:
                    first_exit_ep = ep
                if first_optimal_ep is None and step == bfs_len:
                    first_optimal_ep = ep
                break

        wins.append(won)
        epsilon = max(cfg["min_epsilon"], epsilon * cfg["epsilon_decay"])

        if ep % log_every == 0:
            print(
                f"  [SARSA] episode={ep:>5}  "
                f"success={sum(wins)/len(wins):.0%}  "
                f"epsilon={epsilon:.3f}"
            )

    return q_table, first_exit_ep, first_optimal_ep


# ---------------------------------------------------------------------------
# Per-maze runner
# ---------------------------------------------------------------------------
def run_maze(label, maze_module):
    """Train both algorithms on one maze and print a comparison table."""
    cfg  = MAZE_CONFIGS[label]
    maze = build_maze(maze_module)
    bfs_len = shortest_path_bfs(maze, cfg)

    print()
    print("=" * 62)
    print(
        f"  {label.upper()}  {maze.width}x{maze.height}  "
        f"walls={len(maze.walls)}  "
        f"BFS optimal={bfs_len}  "
        f"episodes={cfg['episodes']}"
    )
    print("=" * 62)

    # Q-Learning
    if cfg["run_q_learning"]:
        print("\n--- Q-Learning ---")
        random.seed(RANDOM_SEED)
        qt_ql, ql_first_exit, ql_first_opt = train_q_learning(maze, cfg, bfs_len)
        path_ql = extract_path(maze, qt_ql, cfg)
        eval_ql = evaluate(maze, qt_ql, cfg)
    else:
        print("\n[skipping Q-Learning — run_q_learning=False]")
        path_ql = None
        eval_ql = None
        ql_first_exit = ql_first_opt = None

    # SARSA
    if cfg["run_sarsa"]:
        print("\n--- SARSA ---")
        random.seed(RANDOM_SEED)
        qt_sa, sa_first_exit, sa_first_opt = train_sarsa(maze, cfg, bfs_len)
        path_sa = extract_path(maze, qt_sa, cfg)
        eval_sa = evaluate(maze, qt_sa, cfg)
    else:
        print("\n[skipping SARSA — run_sarsa=False]")
        path_sa = None
        eval_sa = None
        sa_first_exit = sa_first_opt = None

    # Results table
    def _fmt_path(path):
        if path is None:
            return "─", "─", "─"
        length  = len(path) - 1
        reached = path[-1] == maze.exit
        return str(reached), str(length), str(length == bfs_len)

    def _fmt_eval(ev):
        if ev is None:
            return "─", "─"
        win_rate, avg_steps = ev
        return f"{win_rate:.0%}", f"{avg_steps:.1f}"

    def _fmt_ep(ep):
        return str(ep) if ep is not None else "never"

    ql_exit, ql_len, ql_opt   = _fmt_path(path_ql)
    sa_exit, sa_len, sa_opt   = _fmt_path(path_sa)
    ql_winrate, ql_avgsteps   = _fmt_eval(eval_ql)
    sa_winrate, sa_avgsteps   = _fmt_eval(eval_sa)

    print()
    print(f"{'─' * 62}")
    print(f"{'Metric':<30} {'Q-Learning':>14} {'SARSA':>14}")
    print(f"{'─' * 62}")
    print(f"{'Reached exit (greedy)':<30} {ql_exit:>14} {sa_exit:>14}")
    print(f"{'Greedy path length':<30} {ql_len:>14} {sa_len:>14}")
    print(f"{'BFS shortest path':<30} {bfs_len:>14} {bfs_len:>14}")
    print(f"{'Matches optimal':<30} {ql_opt:>14} {sa_opt:>14}")
    print(f"{'─' * 62}")
    print(f"{'First exit found (ep)':<30} {_fmt_ep(ql_first_exit):>14} {_fmt_ep(sa_first_exit):>14}")
    print(f"{'First optimal path (ep)':<30} {_fmt_ep(ql_first_opt):>14} {_fmt_ep(sa_first_opt):>14}")
    print(f"{'─' * 62}")
    print(f"{'Eval win rate (200 eps)':<30} {ql_winrate:>14} {sa_winrate:>14}")
    print(f"{'Eval avg steps (winners)':<30} {ql_avgsteps:>14} {sa_avgsteps:>14}")
    print(f"{'─' * 62}")

    if path_ql is not None:
        print("\nQ-Learning path:")
        print_path(maze, path_ql)
    if path_sa is not None:
        print("\nSARSA path:")
        print_path(maze, path_sa)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
MAZES = [
    ("maze_1", maze_1),
    ("maze_2", maze_2),
    ("maze_3", maze_3),
]


def main():
    for label, module in MAZES:
        if MAZE_CONFIGS[label]["enabled"]:
            run_maze(label, module)
        else:
            print(f"\n[skipping {label} — enabled=False]")


if __name__ == "__main__":
    main()
