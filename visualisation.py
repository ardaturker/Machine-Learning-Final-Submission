"""
Shared visualisation utilities for DQN3, SAC, and SARSA maze agents.

Layout mirrors the DDPG visualisation in the Mauro branch.

Provides:
    MazeData                   — dataclass holding static maze layout
    draw_maze()                — render one frame onto a matplotlib axis
    find_shortest_path()       — BFS from start to exit
    save_maze_layout()         — maze PNG with BFS optimal path overlay
    save_episode_animation()   — animated GIF of a test episode
    save_training_steps_gif()  — animated GIF of per-step training snapshots
    save_sac_training_plot()   — 2×2 SAC learning curve figure
    save_sarsa_training_plot() — 2×2 SARSA learning curve figure
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter


# ---------------------------------------------------------------------------
# Colour palette  (matches Mauro-branch DDPG style)
# ---------------------------------------------------------------------------
C_WALL  = '#2C3E50'
C_FLOOR = '#FDFEFE'
C_START = '#27AE60'
C_EXIT  = '#E74C3C'
C_SPIKE = '#8E44AD'
C_PATH  = '#F39C12'
C_AGENT = '#2980B9'
C_EDGE  = '#BDC3C7'

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


# ---------------------------------------------------------------------------
# MazeData  — everything needed to render the maze
# ---------------------------------------------------------------------------
@dataclass
class MazeData:
    width:      int
    height:     int
    start:      tuple
    exit:       tuple
    walls:      frozenset
    spikes:     frozenset = field(default_factory=frozenset)
    maze_id:    int       = 1
    size_label: str       = ''

    @property
    def figsize(self) -> int:
        """Scale figure with maze size: 9 / 11 / 13 for mazes 1 / 2 / 3."""
        return 7 + self.maze_id * 2


# ---------------------------------------------------------------------------
# Core grid renderer
# ---------------------------------------------------------------------------
def draw_maze(
    maze:      MazeData,
    path=None,
    agent_pos=None,
    title:     str | None = None,
    ax=None,
):
    """
    Draw the maze grid.

    When ax is None a new figure is created and (fig, ax) is returned so the
    caller can save or embed it.  When ax is provided the function draws in-
    place and returns nothing (used inside FuncAnimation updates).
    """
    standalone = (ax is None)
    if standalone:
        fig, ax = plt.subplots(figsize=(maze.figsize, maze.figsize))

    ax.set_xlim(0, maze.width)
    ax.set_ylim(0, maze.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    path_set = set(path) if path else set()

    for y in range(maze.height):
        for x in range(maze.width):
            cell = (x, y)
            if   cell in maze.walls:  colour, label = C_WALL,  ''
            elif cell == maze.start:  colour, label = C_START, 'S'
            elif cell == maze.exit:   colour, label = C_EXIT,  'E'
            elif cell in maze.spikes: colour, label = C_SPIKE, ''
            elif cell in path_set:    colour, label = C_PATH,  ''
            else:                     colour, label = C_FLOOR, ''

            ax.add_patch(patches.Rectangle(
                (x, y), 1, 1,
                linewidth=0.3, edgecolor=C_EDGE, facecolor=colour,
            ))
            if label:
                ax.text(x + 0.5, y + 0.5, label,
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    if agent_pos is not None:
        ax.add_patch(patches.Circle(
            (agent_pos[0] + 0.5, agent_pos[1] + 0.5), 0.35,
            color=C_AGENT, zorder=5,
        ))

    ax.set_xticks(range(maze.width  + 1))
    ax.set_yticks(range(maze.height + 1))
    ax.tick_params(labelsize=6)
    ax.set_xlabel('X (column)', fontsize=10)
    ax.set_ylabel('Y (row)',    fontsize=10)
    ax.set_title(
        title or f'Maze {maze.maze_id} ({maze.size_label})',
        fontsize=12, fontweight='bold', pad=8,
    )

    legend_items = [
        patches.Patch(facecolor=C_WALL,  label='Wall'),
        patches.Patch(facecolor=C_FLOOR, edgecolor='gray', label='Floor'),
        patches.Patch(facecolor=C_START, label=f'Start {maze.start}'),
        patches.Patch(facecolor=C_EXIT,  label=f'Exit  {maze.exit}'),
        patches.Patch(facecolor=C_PATH,  label='Path taken'),
        patches.Patch(facecolor=C_AGENT, label='Agent'),
    ]
    if maze.spikes:
        legend_items.insert(4, patches.Patch(facecolor=C_SPIKE, label='Spike'))
    ax.legend(handles=legend_items, loc='upper right',
              bbox_to_anchor=(1.28, 1.0), fontsize=8)

    if standalone:
        plt.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# BFS shortest path
# ---------------------------------------------------------------------------
def find_shortest_path(maze: MazeData) -> list | None:
    """BFS from start to exit; returns ordered list of (x, y) cells or None."""
    queue   = deque([(maze.start, [maze.start])])
    visited = {maze.start}
    while queue:
        pos, path = queue.popleft()
        if pos == maze.exit:
            return path
        x, y = pos
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nxt = (x + dx, y + dy)
            if (nxt not in visited
                    and nxt not in maze.walls
                    and 0 <= nxt[0] < maze.width
                    and 0 <= nxt[1] < maze.height):
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


# ---------------------------------------------------------------------------
# Maze layout PNG  (BFS optimal path overlay)
# ---------------------------------------------------------------------------
def save_maze_layout(maze: MazeData, results_dir: str, suffix: str) -> None:
    """Save a static PNG of the maze with the BFS shortest path highlighted."""
    bfs = find_shortest_path(maze)
    steps_txt = f' — optimal path ({len(bfs)} steps, BFS)' if bfs else ''
    title = f'Maze {maze.maze_id} ({maze.size_label}){steps_txt}'

    fig, _ = draw_maze(maze, path=bfs, title=title)
    out = os.path.join(results_dir, f'maze_layout{suffix}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Episode animation GIF
# ---------------------------------------------------------------------------
def save_episode_animation(
    maze:     MazeData,
    frames:   list,
    filename: str,
    fps:      int = 6,
) -> None:
    """
    Save an animated GIF of one episode.

    Each element of *frames* must be a dict with keys:
        path    — list of (x, y) cells visited so far
        agent   — (x, y) current position
        step    — int step counter
        dist    — int distance to exit
        reward  — float step reward
        action  — str action name (UP/DOWN/LEFT/RIGHT)
        moved   — bool whether the agent actually moved
        exit    — bool whether the exit was reached this step
    """
    if not frames:
        print('No frames — episode GIF skipped.')
        return

    fig, ax = plt.subplots(figsize=(maze.figsize, maze.figsize))

    def update(i):
        ax.clear()
        f      = frames[i]
        status = 'EXIT!' if f['exit'] else ('BLOCKED' if not f['moved'] else 'OK')
        draw_maze(
            maze,
            path      = f['path'],
            agent_pos = f['agent'],
            title     = (
                f'Step {f["step"]:3d}  |  {f["action"]:5s}  |  '
                f'Dist: {f["dist"]:3d}  |  '
                f'Reward: {f["reward"]:+6.2f}  |  {status}'
            ),
            ax=ax,
        )

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=int(1000 / fps), repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'Saved: {filename}  ({len(frames)} frames, {fps} fps)')


# ---------------------------------------------------------------------------
# Training step snapshot GIF  (for agents that record per-step data)
# ---------------------------------------------------------------------------
def save_training_steps_gif(
    maze:      MazeData,
    snapshots: list,
    filename:  str,
    fps:       int = 15,
) -> None:
    """
    Save an animated GIF of training step snapshots.

    Each element of *snapshots* must be a dict with keys:
        timestep — int global step counter
        episode  — int episode number
        path     — list of (x, y) cells in the current episode so far
        agent    — (x, y) current position
    """
    if not snapshots:
        print('No step snapshots — training GIF skipped.')
        return

    fig, ax = plt.subplots(figsize=(maze.figsize, maze.figsize))

    def update(i):
        ax.clear()
        s = snapshots[i]
        draw_maze(
            maze,
            path      = s['path'],
            agent_pos = s['agent'],
            title     = f'Training step {s["timestep"]:,}  |  Episode {s["episode"]}',
            ax=ax,
        )

    anim = FuncAnimation(fig, update, frames=len(snapshots),
                         interval=int(1000 / fps), repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'Saved: {filename}  ({len(snapshots)} frames, {fps} fps)')


# ---------------------------------------------------------------------------
# Rolling mean helper
# ---------------------------------------------------------------------------
def _rolling_mean(values: list, window: int = 50) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = np.mean(values[i - window + 1 : i + 1])
    return out




# ---------------------------------------------------------------------------
# SAC training plot  (eval reward / distance / eval steps / train reward)
# ---------------------------------------------------------------------------
def save_sac_training_plot(
    training_history: list,
    maze_id:          int,
    size_label:       str,
    results_dir:      str,
    suffix:           str,
) -> None:
    """Save a 2×2 figure showing SAC learning curves."""
    if not training_history:
        print('No training history — SAC plot skipped.')
        return

    episodes  = [h['episode']      for h in training_history]
    rewards   = [h['total_reward'] for h in training_history]
    distances = [h['final_dist']   for h in training_history]
    steps     = [h['steps']        for h in training_history]
    train_rew = [h['train_reward'] for h in training_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'SAC Training — Maze {maze_id} ({size_label})', fontsize=14)

    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color='#2980B9', label='raw')
    ax.plot(episodes, _rolling_mean(rewards), color='#2980B9',
            linewidth=2, label='50-ep avg')
    success_ep = [h['episode']      for h in training_history if h['success']]
    success_r  = [h['total_reward'] for h in training_history if h['success']]
    if success_ep:
        ax.scatter(success_ep, success_r, color='#27AE60',
                   zorder=5, s=40, label='Exit reached')
    ax.set_title('Eval reward per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes, distances, alpha=0.3, color='#E74C3C', label='raw')
    ax.plot(episodes, _rolling_mean(distances), color='#E74C3C',
            linewidth=2, label='50-ep avg')
    ax.axhline(0, color='#27AE60', linestyle='--', linewidth=1.2,
               label='Exit (dist=0)')
    ax.set_title('Final Manhattan distance to exit')
    ax.set_xlabel('Episode'); ax.set_ylabel('Distance')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, steps, alpha=0.3, color='coral', label='raw')
    ax.plot(episodes, _rolling_mean(steps), color='coral',
            linewidth=2, label='50-ep avg')
    ax.set_title('Eval steps per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, train_rew, alpha=0.3, color='green', label='raw')
    ax.plot(episodes, _rolling_mean(train_rew), color='green',
            linewidth=2, label='50-ep avg')
    ax.set_title('Training reward per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, f'training_plot_SAC{suffix}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Plot saved  → {out}')


# ---------------------------------------------------------------------------
# SARSA training plot  (reward / steps / success rate / epsilon)
# ---------------------------------------------------------------------------
def save_sarsa_training_plot(
    rewards:    list,
    steps:      list,
    epsilons:   list,
    solved:     list,
    maze_id:    int,
    size_label: str,
    results_dir: str,
    suffix:     str,
) -> None:
    """Save a 2×2 figure showing SARSA learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'SARSA Training — Maze {maze_id} ({size_label})', fontsize=14)

    eps_range = range(1, len(rewards) + 1)

    ax = axes[0, 0]
    ax.plot(eps_range, rewards, alpha=0.3, color='steelblue', label='raw')
    ax.plot(eps_range, _rolling_mean(rewards), color='steelblue',
            linewidth=2, label='50-ep avg')
    ax.set_title('Total reward per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps_range, steps, alpha=0.3, color='coral', label='raw')
    ax.plot(eps_range, _rolling_mean(steps), color='coral',
            linewidth=2, label='50-ep avg')
    ax.set_title('Steps per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    solved_float = [float(s) for s in solved]
    ax.plot(eps_range, _rolling_mean(solved_float), color='green', linewidth=2)
    ax.set_title('Success rate (50-ep rolling)')
    ax.set_xlabel('Episode'); ax.set_ylabel('Rate')
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps_range, epsilons, color='purple', linewidth=2)
    ax.set_title('Epsilon (exploration rate)')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, f'training_plot_SARSA{suffix}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Plot saved  → {out}')


# ---------------------------------------------------------------------------
# DQN3 training plot  (reward / steps / loss / epsilon) — Double DQN variant
# ---------------------------------------------------------------------------
def save_dqn3_training_plot(
    rewards:    list,
    steps:      list,
    losses:     list,
    epsilons:   list,
    maze_id:    int,
    size_label: str,
    results_dir: str,
    suffix:     str,
) -> None:
    """Save a 2×2 figure showing DQN3 (Double DQN, TensorFlow) learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'DQN3 (Double DQN) Training — Maze {maze_id} ({size_label})', fontsize=14)

    eps_range = range(1, len(rewards) + 1)

    ax = axes[0, 0]
    ax.plot(eps_range, rewards, alpha=0.3, color='steelblue', label='raw')
    ax.plot(eps_range, _rolling_mean(rewards), color='steelblue',
            linewidth=2, label='50-ep avg')
    ax.set_title('Total reward per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(eps_range, steps, alpha=0.3, color='coral', label='raw')
    ax.plot(eps_range, _rolling_mean(steps), color='coral',
            linewidth=2, label='50-ep avg')
    ax.set_title('Steps per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(eps_range, losses, alpha=0.3, color='green', label='raw')
    ax.plot(eps_range, _rolling_mean(losses), color='green',
            linewidth=2, label='50-ep avg')
    ax.set_title('Average Huber loss per episode')
    ax.set_xlabel('Episode'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(eps_range, epsilons, color='purple', linewidth=2)
    ax.set_title('Epsilon (exploration rate)')
    ax.set_xlabel('Episode'); ax.set_ylabel('ε')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, f'training_plot_DQN3{suffix}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Plot saved  → {out}')


# ---------------------------------------------------------------------------
# PPO training plot  (success rate / losses / entropy / eval solved)
# ---------------------------------------------------------------------------
def save_ppo_training_plot(
    training_log: list,
    maze_id:      int,
    size_label:   str,
    results_dir:  str,
    suffix:       str,
) -> None:
    """Save a 2×2 figure showing PPO learning curves.

    Each entry in *training_log* must have keys:
        update, rollout_success, policy_loss, value_loss, entropy, kl
    Entries that have eval data additionally carry:
        eval_reached, eval_optimal, eval_tasks, eval_avg_len
    """
    if not training_log:
        print('No training log — PPO plot skipped.')
        return

    updates          = [r['update']          for r in training_log]
    rollout_success  = [r['rollout_success']  for r in training_log]
    policy_loss      = [r['policy_loss']      for r in training_log]
    value_loss       = [r['value_loss']       for r in training_log]
    entropy          = [r['entropy']          for r in training_log]

    eval_records  = [r for r in training_log if 'eval_reached' in r]
    eval_updates  = [r['update']                                     for r in eval_records]
    eval_solved   = [r['eval_reached'] / max(1, r['eval_tasks'])     for r in eval_records]
    eval_optimal  = [r['eval_optimal'] / max(1, r['eval_tasks'])     for r in eval_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'PPO Training — Maze {maze_id} ({size_label})', fontsize=14)

    ax = axes[0, 0]
    ax.plot(updates, rollout_success, alpha=0.35, color='#2980B9', label='rollout win rate')
    ax.plot(updates, _rolling_mean(rollout_success, min(10, len(rollout_success))),
            color='#2980B9', linewidth=2)
    if eval_updates:
        ax.plot(eval_updates, eval_solved,  color='#27AE60', linewidth=2,
                marker='o', markersize=4, label='eval solved')
        ax.plot(eval_updates, eval_optimal, color='#F39C12', linewidth=2,
                marker='s', markersize=4, label='eval optimal')
    ax.set_title('Success rate')
    ax.set_xlabel('Update'); ax.set_ylabel('Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(updates, policy_loss, alpha=0.35, color='coral', label='raw')
    ax.plot(updates, _rolling_mean(policy_loss, min(10, len(policy_loss))),
            color='coral', linewidth=2, label='10-update avg')
    ax.set_title('Policy loss')
    ax.set_xlabel('Update'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(updates, value_loss, alpha=0.35, color='green', label='raw')
    ax.plot(updates, _rolling_mean(value_loss, min(10, len(value_loss))),
            color='green', linewidth=2, label='10-update avg')
    ax.set_title('Value loss')
    ax.set_xlabel('Update'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(updates, entropy, color='purple', linewidth=2)
    ax.set_title('Entropy (exploration pressure)')
    ax.set_xlabel('Update'); ax.set_ylabel('Entropy')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, f'training_plot_PPO{suffix}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Plot saved: {out}')
