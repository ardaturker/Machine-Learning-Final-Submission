"""
SARSA  —  On-policy tabular TD control.

Based on: Rummery & Niranjan (1994) "On-line Q-learning using connectionist
systems"; Sutton & Barto "Reinforcement Learning: An Introduction" §6.4.

Key ideas implemented:
  1. Q-table     — dictionary mapping (state, action) → value
  2. On-policy   — next action sampled from ε-greedy policy (not argmax)
  3. Epsilon-greedy — ε decays from 1.0 → 0.05 over training

Select the maze via MAZE_ID in config.py (1, 2, or 3).

Run:
    python agents/sarsa_train.py

Outputs (saved to Results/SARSA/):
    training_SARSA_maze<N>.csv   — per-episode reward, steps, epsilon
    maze_layout_maze<N>.png      — maze map with BFS optimal path
    episode_animation_maze<N>.gif
    training_plot_SARSA_maze<N>.png  — learning curve
"""

from __future__ import annotations

import importlib
import os
import sys
import csv
import random

import numpy as np


# ---------------------------------------------------------------------------
# Hyper-parameters  (including maze selection)
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import SARSA, MAZE_ID

EPISODES      = SARSA.EPISODES
ALPHA         = SARSA.ALPHA         # learning rate
GAMMA         = SARSA.GAMMA         # discount factor
EPSILON_START = SARSA.EPSILON       # initial exploration rate
EPSILON_DECAY = SARSA.EPSILON_DECAY # multiplicative decay per episode
EPSILON_MIN   = SARSA.MIN_EPSILON   # floor for exploration


# ---------------------------------------------------------------------------
# Maze environment  —  resolved from MAZE_ID in config.py
# ---------------------------------------------------------------------------
_MAZE_INFO = {
    1: ("maze_1", "maze_env",    "16×16"),
    2: ("maze_2", "maze_2_env",  "25×25"),
    3: ("maze_3", "maze_3_env",  "35×35"),
}

if MAZE_ID not in _MAZE_INFO:
    raise ValueError(f"MAZE_ID must be 1, 2, or 3 — got {MAZE_ID!r}")

_maze_dir, _maze_module, MAZE_SIZE_LABEL = _MAZE_INFO[MAZE_ID]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", _maze_dir))
_env_mod    = importlib.import_module(_maze_module)
MazeEnv     = _env_mod.MazeEnv
NUM_ACTIONS = _env_mod.NUM_ACTIONS

MAX_STEPS = _env_mod.WIDTH * _env_mod.HEIGHT * 4   # max moves per episode


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Machine Learning", "Results", "SARSA")
os.makedirs(RESULTS_DIR, exist_ok=True)

_SUFFIX = f"_maze{MAZE_ID}"


# ---------------------------------------------------------------------------
# 1. SARSA Agent
# ---------------------------------------------------------------------------
class SARSAAgent:
    """
    Tabular SARSA agent.

    The Q-table is a plain dict keyed by (state, action).
    Missing entries default to 0.0 so the table grows lazily as states
    are visited for the first time.
    """

    def __init__(self) -> None:
        self.q_table: dict[tuple, float] = {}
        self.epsilon: float = EPSILON_START

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------

    def get_q(self, state: tuple, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def best_action(self, state: tuple) -> int:
        return max(range(NUM_ACTIONS), key=lambda a: self.get_q(state, a))

    # ------------------------------------------------------------------
    # Action selection  —  epsilon-greedy
    # ------------------------------------------------------------------

    def select_action(self, state: tuple) -> int:
        """
        With probability ε pick a random action (explore).
        Otherwise pick argmax Q(s, ·) (exploit).
        """
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        return self.best_action(state)

    # ------------------------------------------------------------------
    # SARSA update  —  on-policy Bellman step
    # ------------------------------------------------------------------

    def update(
        self,
        state:       tuple,
        action:      int,
        reward:      float,
        next_state:  tuple,
        next_action: int,
    ) -> None:
        """
        SARSA update:
            Q(s, a) ← Q(s, a) + α · [r + γ·Q(s', a') − Q(s, a)]

        a' is sampled from the current ε-greedy policy, not taken as the
        greedy maximum.  This is the key on-policy distinction from Q-learning.
        """
        old_q  = self.get_q(state, action)
        next_q = self.get_q(next_state, next_action)
        self.q_table[(state, action)] = old_q + ALPHA * (reward + GAMMA * next_q - old_q)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ---------------------------------------------------------------------------
# 2. Training loop
# ---------------------------------------------------------------------------

def train() -> tuple:
    env   = MazeEnv(max_steps=MAX_STEPS)
    agent = SARSAAgent()

    # ---- Data collectors (what we'll plot / export) ----
    episode_rewards:  list[float] = []
    episode_steps:    list[int]   = []
    episode_epsilons: list[float] = []
    episode_solved:   list[bool]  = []

    csv_path = os.path.join(RESULTS_DIR, f"training_SARSA{_SUFFIX}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "solved", "epsilon"])

        print(f"Training SARSA on Maze {MAZE_ID} ({MAZE_SIZE_LABEL})")
        print(f"{'Episode':>8}  {'Reward':>8}  {'Steps':>6}  {'Epsilon':>7}  {'Solved':>6}")
        print("-" * 50)

        for ep in range(1, EPISODES + 1):
            env.reset()
            state  = env._player
            action = agent.select_action(state)

            total_reward = 0.0
            solved       = False

            for _ in range(MAX_STEPS):
                _, reward, terminated, truncated, _ = env.step(action)
                next_state  = env._player
                next_action = agent.select_action(next_state)

                agent.update(state, action, reward, next_state, next_action)

                state  = next_state
                action = next_action
                total_reward += reward

                if terminated:
                    solved = True
                if terminated or truncated:
                    break

            agent.decay_epsilon()

            episode_rewards.append(total_reward)
            episode_steps.append(env._steps)
            episode_epsilons.append(agent.epsilon)
            episode_solved.append(solved)

            writer.writerow([ep, f"{total_reward:.2f}", env._steps,
                             int(solved), f"{agent.epsilon:.4f}"])

            if ep % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                print(f"{ep:>8}  {avg_r:>8.2f}  {env._steps:>6}  "
                      f"{agent.epsilon:>7.4f}  {str(solved):>6}")

    print("\nTraining complete.")
    print(f"Log saved  → {csv_path}")

    return agent, episode_rewards, episode_steps, episode_epsilons, episode_solved


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------

def _build_maze_data():
    """Assemble a MazeData from the module-level env-module globals."""
    from visualisation import MazeData
    return MazeData(
        width      = _env_mod.WIDTH,
        height     = _env_mod.HEIGHT,
        start      = _env_mod.START,
        exit       = _env_mod.EXIT_CELL,
        walls      = _env_mod.WALLS,
        spikes     = getattr(_env_mod, 'SPIKES', frozenset()),
        maze_id    = MAZE_ID,
        size_label = MAZE_SIZE_LABEL,
    )


def _run_vis_episode(agent: SARSAAgent) -> list[dict]:
    """Run one fully-greedy episode and collect per-step animation frames."""
    env = MazeEnv(max_steps=MAX_STEPS)
    env.reset()
    path = [env._player]

    frames  = []
    _ANAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for _ in range(MAX_STEPS):
        prev_pos = env._player
        action   = agent.best_action(env._player)   # greedy — no exploration
        _, reward, terminated, truncated, _ = env.step(action)

        moved = env._player != prev_pos
        path.append(env._player)

        frames.append({
            'path':   list(path),
            'agent':  env._player,
            'step':   env._steps,
            'dist':   env._manhattan(),
            'reward': reward,
            'action': _ANAMES.get(action, '?'),
            'moved':  moved,
            'exit':   terminated and reward > 0,
        })
        if terminated or truncated:
            break

    return frames


def visualise(
    agent:            SARSAAgent,
    episode_rewards:  list,
    episode_steps:    list,
    episode_epsilons: list,
    episode_solved:   list,
) -> None:
    """Generate maze layout PNG, episode animation GIF, and training curves."""
    from visualisation import (
        save_maze_layout,
        save_episode_animation,
        save_sarsa_training_plot,
    )

    print('\nGenerating visualisations...')
    maze = _build_maze_data()

    save_maze_layout(maze, RESULTS_DIR, _SUFFIX)

    frames   = _run_vis_episode(agent)
    gif_path = os.path.join(RESULTS_DIR, f'episode_animation{_SUFFIX}.gif')
    save_episode_animation(maze, frames, gif_path, fps=6)

    save_sarsa_training_plot(
        episode_rewards, episode_steps, episode_epsilons, episode_solved,
        MAZE_ID, MAZE_SIZE_LABEL, RESULTS_DIR, _SUFFIX,
    )
    print('Visualisation complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent, ep_rewards, ep_steps, ep_epsilons, ep_solved = train()
    visualise(agent, ep_rewards, ep_steps, ep_epsilons, ep_solved)
