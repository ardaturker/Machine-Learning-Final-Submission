"""
SAC Maze Solver — 16x16
========================


Run:
    python ddpg_maze_complete.py              # full training + 1 test (aproxx 20 min)
    python ddpg_maze_complete.py --test-only  # reproduce test without retraining (1 min)

Expected outputs: 
-episode.animation.gif 
-all_step_traing.gif
-2 pdf for the learning curve and to understand the rewarding system.

Algorithm: SAC (Soft Actor-Critic)
- Off-policy actor-critic with entropy regularisation (DDPG suffer of policy collapse)
- Entropy term prevents policy collapse (the main failure mode of DDPG)
- No external action noise needed — exploration is built into the objective


Reward system explanation:
  -0.1       per step
  +1.0       moving closer to exit (BFS distance)
  -0.3       moving farther from exit (BFS distance)
  -3.0       wall collision
  +0.5       first visit to a new cell (exploration bonus, introduced because in previous experiment the model was not willing to explore but he stayed in his comfort zone )
  -1.0*n     revisiting cell n times (capped at -20), to avoid terrible loops
  +(6-d)     proximity bonus when BFS distance d <= 5, sometimes when is closer and too far away from the "safe route" the agent prefer to came back, this is a little incentive to pursue the exit 
  +200       reaching the exit (overrides all others)
"""

import os
import sys
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

SEED = 42
#for reproducibility
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

print('All imports OK.')


# =============================================================================
# SECTION 1 — MAZE ENVIRONMENT
# =============================================================================

class MazeEnv:
    """
    25x25 grid maze (maze_2).

    Agent starts at (0, 0)  top-left.
    Agent must reach (24, 24)  bottom-right.

    Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT

    Observation (7 floats, all in [0, 1]):
      [norm_row, norm_col, norm_bfs_dist,
       can_up, can_down, can_left, can_right]

    Wall-visibility flags tell the agent which directions are physically
    open at the current position, saving wasted wall-hit steps.
    """

    def __init__(self):
        self.GRID_ROWS = 25
        self.GRID_COLS = 25

        # Wall layout from maze_2/maze_2.py — coordinates converted from
        # maze_2's (x=col, y=row) format to our (row, col) format.
        # 25x25 snake-like maze: horizontal walls with alternating gaps
        # on the right (col 3, cols 21-22) and left (cols 2-3).
        self.walls = {
            # Row 3: gap at col 3 and cols 21-22
            *[(3,c) for c in list(range(0,3)) + list(range(4,21)) + list(range(23,25))],
            # Row 7: gap at cols 2-3
            *[(7,c) for c in list(range(0,2)) + list(range(4,25))],
            # Row 11: gap at cols 21-22
            *[(11,c) for c in list(range(0,21)) + list(range(23,25))],
            # Row 15: gap at cols 2-3
            *[(15,c) for c in list(range(0,2)) + list(range(4,25))],
            # Row 19: gap at cols 21-22
            *[(19,c) for c in list(range(0,21)) + list(range(23,25))],
            # Row 22: gap at cols 2-3
            *[(22,c) for c in list(range(0,2)) + list(range(4,25))],
            # Small vertical walls
            (1,8),(2,8),
            (5,16),(6,16),
            (9,10),(10,10),
            (13,17),(14,17),
            (17,7),(18,7),
            (4,5),(5,5),
            (8,21),(9,21),
            (12,3),(13,3),
            (16,22),(17,22),
            # Single wall cells
            (20,14),(20,9),
            # Extra walls
            (2,2),(1,12),(1,13),(9,4),(9,5),
            (13,18),(13,19),(17,11),(17,12),(21,20),(21,21),
        }

        self.start_pos = (0, 0)
        self.exit_pos  = (self.GRID_ROWS - 1, self.GRID_COLS - 1)  # (24,24)

        self.player_row, self.player_col = self.start_pos
        self.steps_taken = 0
        self.visit_count = {}
        self.max_steps   = 1000

        
        # bfs_dist[(r,c)] = shortest path steps from (r,c) to exit through the
        # actual maze (respects walls).
        self.bfs_dist     = self._compute_bfs_distances()
        self.max_bfs_dist = max(self.bfs_dist.values()) if self.bfs_dist else 30

        self.action_space = spaces.Discrete(4)

        # 7-float observation: position (2 , x,y) + BFS distance (how muche is distance from exit) + wall flags (he can go up, down, left , right , there are walls?)
        self.observation_space = spaces.Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.ones(7,  dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.player_row, self.player_col = self.start_pos
        self.steps_taken = 0
        self.visit_count = {self.start_pos: 1}
        obs  = self._get_obs()
        info = {
            'position':         self.start_pos,
            'distance_to_exit': self._distance_to_exit(),
            'steps_taken':      0,
        }
        return obs, info

    def _compute_bfs_distances(self):
        """BFS from exit outward — fills shortest path distance for every reachable cell."""
        from collections import deque
        dist  = {self.exit_pos: 0}
        queue = deque([(self.exit_pos, 0)])
        while queue:
            (r, c), d = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nxt = (r+dr, c+dc)
                if nxt not in dist and self._is_free(r+dr, c+dc):
                    dist[nxt] = d + 1
                    queue.append((nxt, d+1))
        return dist

    def _is_free(self, r, c):
        """True if (r, c) is inside the grid and not a wall."""
        return 0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS and (r, c) not in self.walls

    def _get_obs(self):
        """7-float observation — uses BFS distance for accurate maze-aware signal."""
        r, c = self.player_row, self.player_col
        return np.array([
            r / (self.GRID_ROWS - 1),
            c / (self.GRID_COLS - 1),
            self._distance_to_exit() / self.max_bfs_dist,
            float(self._is_free(r-1, c)),
            float(self._is_free(r+1, c)),
            float(self._is_free(r, c-1)),
            float(self._is_free(r, c+1)),
        ], dtype=np.float32)

    def _distance_to_exit(self):
        return self.bfs_dist.get((self.player_row, self.player_col), self.max_bfs_dist)

    def step(self, action):
        self.steps_taken += 1

        old_dist = self._distance_to_exit()
        old_pos  = (self.player_row, self.player_col)

        new_row, new_col = self.player_row, self.player_col
        if   action == 0: new_row -= 1   # UP
        elif action == 1: new_row += 1   # DOWN
        elif action == 2: new_col -= 1   # LEFT
        elif action == 3: new_col += 1   # RIGHT

        moved = self._is_free(new_row, new_col)
        if moved:
            self.player_row, self.player_col = new_row, new_col

        new_pos  = (self.player_row, self.player_col)
        new_dist = self._distance_to_exit()

        # ── Reward system ──────────────────────────────────────────────────
        reward           = 0.0
        reward_breakdown = []

        # 1. Step penalty — reduced to 0.05 for the 25x25 maze because the
        # optimal path is ~150 steps; -0.1 per step was too punishing.
        reward -= 0.05
        reward_breakdown.append(('step_penalty', -0.05))

        # 2. Wall collision
        if not moved:
            reward -= 3.0
            reward_breakdown.append(('wall_collision', -3.0))

        # 3. BFS distance shaping — reward getting closer, penalise going away
        if moved:
            if new_dist < old_dist:
                reward += 1.0
                reward_breakdown.append(('closer_to_exit', +1.0))
            elif new_dist > old_dist:
                reward -= 0.3
                reward_breakdown.append(('farther_from_exit', -0.3))

        # 4. Exploration bonus + revisit penalty.
        # +0.5 for first visit pushes the agent to cover new ground.
        # Revisit penalty capped at 20 makes loops unsustainable.
        if moved:
            n = self.visit_count.get(new_pos, 0)
            if n == 0:
                reward += 1.5
                reward_breakdown.append(('exploration_bonus', +1.5))
            else:
                # Cap raised to 5: prevents loops but tolerates the inevitable
                # backtracking needed to traverse 24-cell-wide snake corridors.
                penalty = min(1.0 * n, 5.0)
                reward -= penalty
                reward_breakdown.append(('revisit_penalty', -round(penalty, 2)))
            self.visit_count[new_pos] = n + 1

        # 5. Proximity bonus — extra incentive within 20 BFS steps of exit.
        # Threshold raised to 20 because the 25x25 snake maze has a path of
        # ~150 steps; a threshold of 5 was too small to guide the last mile.
        if moved and new_dist <= 20:
            prox = float(21 - new_dist)  # dist=1 -> +20, dist=20 -> +1
            reward += prox
            reward_breakdown.append(('proximity_bonus', prox))

        # 6. Exit reward — overrides all shaping
        terminated   = False
        truncated    = False
        reached_exit = (new_pos == self.exit_pos)
        if reached_exit:
            reward           = 200.0
            reward_breakdown = [('reached_exit', 200.0)]
            terminated       = True

        # 7. Time limit
        if self.steps_taken >= self.max_steps:
            truncated = True

        obs  = self._get_obs()
        info = {
            'old_pos':          old_pos,
            'new_pos':          new_pos,
            'old_distance':     old_dist,
            'new_distance':     new_dist,
            'moved':            moved,
            'reward_breakdown': reward_breakdown,
            'steps_taken':      self.steps_taken,
            'reached_exit':     reached_exit,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f'Step {self.steps_taken:4d} | '
            f'Pos: ({self.player_row:2d},{self.player_col:2d}) | '
            f'BFS dist: {self._distance_to_exit():2d}'
        )


# =============================================================================
# SECTION 2 — VISUALIZATION
# =============================================================================

C_WALL  = '#2C3E50'
C_FLOOR = '#FDFEFE'
C_START = '#27AE60'
C_EXIT  = '#E74C3C'
C_PATH  = '#F39C12'
C_AGENT = '#2980B9'
C_EDGE  = '#BDC3C7'


def draw_maze(env, path=None, agent_pos=None, title='16x16 Maze',
              ax=None, figsize=(11, 11)):
    standalone = (ax is None)
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    rows, cols = env.GRID_ROWS, env.GRID_COLS
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    path_set = set(path) if path else set()

    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if   cell in env.walls:       colour, label = C_WALL,  ''
            elif cell == env.start_pos:   colour, label = C_START, 'S'
            elif cell == env.exit_pos:    colour, label = C_EXIT,  'E'
            elif cell in path_set:        colour, label = C_PATH,  ''
            else:                         colour, label = C_FLOOR, ''

            rect = patches.Rectangle(
                (c, r), 1, 1,
                linewidth=0.3, edgecolor=C_EDGE, facecolor=colour
            )
            ax.add_patch(rect)

            if label:
                ax.text(c+0.5, r+0.5, label,
                        ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    if agent_pos is not None:
        ar, ac = agent_pos
        ax.add_patch(patches.Circle((ac+0.5, ar+0.5), 0.35,
                                     color=C_AGENT, zorder=5))

    ax.set_xticks(range(cols+1))
    ax.set_yticks(range(rows+1))
    ax.tick_params(labelsize=7)
    ax.set_xlabel('Column', fontsize=10)
    ax.set_ylabel('Row',    fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    legend_items = [
        patches.Patch(facecolor=C_WALL,  label='Wall'),
        patches.Patch(facecolor=C_FLOOR, edgecolor='gray', label='Floor'),
        patches.Patch(facecolor=C_START, label='Start (0,0)'),
        patches.Patch(facecolor=C_EXIT,  label='Exit (15,15)'),
        patches.Patch(facecolor=C_PATH,  label='Path taken'),
        patches.Patch(facecolor=C_AGENT, label='Agent'),
    ]
    ax.legend(handles=legend_items, loc='upper right',
              bbox_to_anchor=(1.28, 1.0), fontsize=8)

# =============================================================================
# SECTION 3 — SHORTEST PATH (BFS)
# =============================================================================

def find_shortest_path(env):
    """
    BFS from start to exit on the actual maze graph.
    Returns the list of (row, col) cells forming the shortest path,
    or None if no path exists.
    """
    from collections import deque
    start    = env.start_pos
    exit_pos = env.exit_pos
    queue    = deque([(start, [start])])
    visited  = {start}

    while queue:
        pos, path = queue.popleft()
        if pos == exit_pos:
            return path
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if nxt not in visited and env._is_free(r + dr, c + dc):
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None
# =============================================================================
# SECTION 4 — SAC WRAPPER  (4D argmax action space)
# =============================================================================

class ContinuousMazeWrapper(gym.Env):
    """
    Wraps MazeEnv for SAC with a 4-dimensional continuous action space.

    The actor network outputs 4 values [s_up, s_down, s_left, s_right].
    The discrete action is:choose of those 4 scores based on the values .

    Each direction gets its own independent score, so SAC can learn to
    suppress bad directions (e.g. a wall to the right) while boosting
    good ones (eg down exit!!!).
    """

    def __init__(self):
        super().__init__()
        self.env = MazeEnv()
        self.observation_space = self.env.observation_space

        self.action_space = spaces.Box(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4,  1.0, dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        scores   = np.asarray(action).reshape(-1)
        discrete = int(np.argmax(scores))   # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        obs, reward, terminated, truncated, info = self.env.step(discrete)
        info['action_scores']   = scores.tolist()
        info['discrete_action'] = discrete

        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        pass
# =============================================================================
# SECTION 5 — TRAINING CALLBACK
# =============================================================================

class TrainingRecorderCallback(BaseCallback):
    """
    Runs a evaluation episode at the end of every N training episodes
    and stores the path, reward, and distance to exit (useful to understand if the model is actually learning)
    Saves step snapshots for the training GIF and checkpoints the best model
    by reward(useful against collpase policy) AND by minimum distance reached (dual criterion).
    """

    def __init__(self, record_every=1, snap_every=20,
                 best_model_path='models/sac_best', verbose=0):
        super().__init__(verbose)
        self.record_every     = record_every
        self.snap_every       = snap_every
        self.best_model_path  = best_model_path
        self.episode_count    = 0
        self.training_history = []
        self.step_snapshots   = []
        self._ep_reward       = 0.0
        self._cur_path        = [(0, 0)]
        self._cur_episode     = 0
        self.best_reward      = -float('inf')
        self.best_min_dist    = float('inf')

    def _on_step(self) -> bool:
        new_obs = self.locals['new_obs'][0]
        row = round(float(new_obs[0]) * 15)
        col = round(float(new_obs[1]) * 15)
        self._cur_path.append((row, col))

        self._ep_reward += float(self.locals['rewards'][0])

        if self.num_timesteps % self.snap_every == 0:
            self.step_snapshots.append({
                'timestep': self.num_timesteps,
                'episode':  self._cur_episode + 1,
                'path':     list(self._cur_path),
                'agent':    (row, col),
            })

        if self.locals['dones'][0]:
            self.episode_count += 1
            self._cur_episode  += 1
            ep_reward           = self._ep_reward
            self._ep_reward     = 0.0
            self._cur_path      = [(0, 0)]

            if self.episode_count % self.record_every == 0:
                rec = self._eval_greedy()
                rec['episode']      = self.episode_count
                rec['timestep']     = self.num_timesteps
                rec['train_reward'] = ep_reward
                self.training_history.append(rec)

                # Dual checkpoint: save when reward improves OR agent gets closer to exit.
                is_best_reward = rec['total_reward'] > self.best_reward
                is_best_dist   = rec['min_dist']     < self.best_min_dist
                is_best        = is_best_reward or is_best_dist

                if is_best_reward:
                    self.best_reward   = rec['total_reward']
                if is_best_dist:
                    self.best_min_dist = rec['min_dist']
                if is_best:
                    os.makedirs('models', exist_ok=True)
                    self.model.save(self.best_model_path)

                print(
                    f'  [ep {self.episode_count:3d} | t={self.num_timesteps:6,}]  '
                    f'reward={rec["total_reward"]:+7.1f}  '
                    f'final_dist={rec["final_dist"]:2d}  '
                    f'min_dist={rec["min_dist"]:2d}  '
                    f'{"* BEST " if is_best else ""}'
                    f'{"EXIT **" if rec["success"] else ""}'
                )
        return True

    def _eval_greedy(self):
        """Run one noiseless episode with the current policy and return stats."""
        eval_env = ContinuousMazeWrapper()
        obs, _   = eval_env.reset()
        path         = [eval_env.env.start_pos]
        total_reward = 0.0
        final_dist   = eval_env.env._distance_to_exit()
        min_dist     = final_dist
        steps        = 0
        success      = False

        for _ in range(1000):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            total_reward += reward
            path.append(info['new_pos'])
            final_dist = info['new_distance']
            min_dist   = min(min_dist, final_dist)
            steps     += 1
            if info['reached_exit']:
                success = True
            if term or trunc:
                break

        return {
            'path':         path,
            'total_reward': total_reward,
            'final_dist':   final_dist,
            'min_dist':     min_dist,
            'steps':        steps,
            'success':      success,
        }
#not necessary, but just for me during the training to understand as a checkpoint if is learning and what the agent is learning 

# =============================================================================
# SECTION 6 — TRAINING
# =============================================================================

def train(timesteps=150_000, save_path='models/sac_maze_v3', record_every=1):
    """
    Train SAC on the maze.

    SAC (Soft Actor-Critic) advantages over DDPG:
    - Built-in entropy regularisation prevents policy collapse
    - Stochastic policy naturally maintains exploration near the exit
    - More sample-efficient: converges in ~20k timesteps on this maze (due to the experiments )

    Hyperparameters:
    - learning_rate   3e-4    stable without being too slow
    - buffer_size     200k    diverse experiences for better gradient estimates
    - batch_size      256     larger batches for smoother loss surface
    - gamma           0.99    agent values future rewards almost as much as present
    - tau             0.005   slow target-network update for stable Q-values
    - learning_starts 1000    random exploration before updates begin
    - ent_coef        auto    SAC tunes entropy coefficient automatically
    - net_arch     [256,256]  two hidden layers to represent the 7D->4D mapping
    """
    env = ContinuousMazeWrapper()
    env = Monitor(env)

    policy_kwargs = dict(net_arch=[256, 256])

    model = SAC(
        'MlpPolicy',
        env,
        policy_kwargs   = policy_kwargs,
        learning_rate   = 3e-4,
        buffer_size     = 200_000,
        batch_size      = 256,
        gamma           = 0.99,
        tau             = 0.005,
        learning_starts = 1_000,
        ent_coef        = 'auto',
        seed            = SEED,
        verbose         = 1,
    )

    best_path = save_path.replace('sac_maze_v3', 'sac_best')
    # snap_every=300 -> 150000/300 = 500 snapshots -> ~33s GIF at 15fps
    callback = TrainingRecorderCallback(
        record_every=record_every, snap_every=300,
        best_model_path=best_path,
    )

    print(f'Training SAC for {timesteps:,} timesteps...')
    print('(* BEST = new checkpoint saved)')
    model.learn(total_timesteps=timesteps, log_interval=10, callback=callback)

    os.makedirs('models', exist_ok=True)
    model.save(save_path)
    print(f'Final model : {save_path}.zip')
    print(f'Best model  : {best_path}.zip  (reward={callback.best_reward:.1f})')
    print(f'Training episodes: {callback.episode_count}  |  '
          f'Eval records: {len(callback.training_history)}')
    return model, callback.training_history, callback.step_snapshots, best_path

# =============================================================================
# SECTION 7 — EPISODE RUNNER
# =============================================================================
ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

def run_episode_logged(model, env_wrapper, max_steps=1000, deterministic=True):
    """Run one episode and return a per-step log, path, success flag, total reward."""
    obs, info = env_wrapper.reset()
    total_reward = 0.0
    log  = []
    path = [env_wrapper.env.start_pos]

    for step_n in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env_wrapper.step(action)
        total_reward += reward

        scores = info['action_scores']
        disc_a = info['discrete_action']

        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        if   margin > 0.5: confidence = 'HIGH'
        elif margin > 0.2: confidence = 'MED'
        else:               confidence = 'LOW'

        log.append({
            'Step':        step_n + 1,
            'Pos Before':  info['old_pos'],
            'Action':      ACTION_NAMES[disc_a],
            'Confidence':  confidence,
            'Pos After':   info['new_pos'],
            'Dist Before': info['old_distance'],
            'Dist After':  info['new_distance'],
            'Delta Dist':  info['new_distance'] - info['old_distance'],
            'Reward':      round(reward, 3),
            'Moved?':      'YES' if info['moved'] else ' NO',
        })

        path.append(info['new_pos'])

        if terminated or truncated:
            break

    success = info.get('reached_exit', False)
    return log, path, success, total_reward

# =============================================================================
# SECTION 8 — EPISODE ANIMATION
# =============================================================================

def save_episode_animation(env_wrapper, model, filename='episode_animation.gif',
                           max_steps=1000, fps=6):
    """Run one episode and save it as an animated GIF, for better understanding ."""
    obs, _ = env_wrapper.reset()
    frames      = []
    path_so_far = [env_wrapper.env.start_pos]

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env_wrapper.step(action)
        path_so_far.append(info['new_pos'])

        frames.append({
            'path':   list(path_so_far),
            'agent':  info['new_pos'],
            'step':   info['steps_taken'],
            'dist':   info['new_distance'],
            'reward': reward,
            'action': ACTION_NAMES.get(info['discrete_action'], '?'),
            'moved':  info['moved'],
            'exit':   info['reached_exit'],
        })
        if term or trunc:
            break

    fig, ax = plt.subplots(figsize=(9, 9))

    def update(i):
        ax.clear()
        f = frames[i]
        status = 'EXIT!' if f['exit'] else ('BLOCKED' if not f['moved'] else 'OK')
        draw_maze(
            env_wrapper.env,
            path      = f['path'],
            agent_pos = f['agent'],
            title     = (
                f'Step {f["step"]:3d}  |  {f["action"]:5s}  |  '
                f'Dist: {f["dist"]:2d}  |  '
                f'Reward: {f["reward"]:+6.2f}  |  {status}'
            ),
            ax=ax
        )

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=int(1000 / fps), repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  Saved: {filename}  ({len(frames)} frames)')


# =============================================================================
# SECTION 9 — TRAINING STEP GIF
# =============================================================================

def save_all_steps_gif(env, step_snapshots, filename='training_all_steps.gif', fps=15):
    """Animate every recorded training step snapshot, to understanding if is training good ."""
    if not step_snapshots:
        print('No step snapshots to animate.')
        return

    fig, ax = plt.subplots(figsize=(9, 9))

    def update(i):
        ax.clear()
        s = step_snapshots[i]
        draw_maze(
            env,
            path      = s['path'],
            agent_pos = s['agent'],
            title     = f'Training step {s["timestep"]:,}  |  Episode {s["episode"]}',
            ax=ax,
        )

    anim = FuncAnimation(fig, update, frames=len(step_snapshots),
                         interval=int(1000 / fps), repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'Saved: {filename}  ({len(step_snapshots)} frames, {fps}fps)')


# =============================================================================
# SECTION 10 — TRAINING LEARNING CURVES
# =============================================================================

def save_training_curves(training_history):
    """Save training_reward_curve.png and training_distance_curve.png."""
    if not training_history:
        print('No training history — curves skipped.')
        return

    steps_x     = [h['timestep']     for h in training_history]
    rewards_y   = [h['total_reward'] for h in training_history]
    distances_y = [h['final_dist']   for h in training_history]
    success_x   = [h['timestep']     for h in training_history if h['success']]
    success_r   = [h['total_reward'] for h in training_history if h['success']]
    success_d   = [0                 for h in training_history if h['success']]

    # Reward curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps_x, rewards_y, color='#2980B9', linewidth=1.4,
            label='Eval episode reward')
    ax.fill_between(steps_x, rewards_y, alpha=0.12, color='#2980B9')
    if success_x:
        ax.scatter(success_x, success_r, color='#27AE60', zorder=5,
                   s=60, label='Exit reached')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Total Reward (eval episode)', fontsize=12)
    ax.set_title('SAC Training — Reward per Episode', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_reward_curve.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('Saved: training_reward_curve.png')

    # Distance curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps_x, distances_y, color='#E74C3C', linewidth=1.4,
            label='Final BFS distance to exit')
    ax.fill_between(steps_x, distances_y, alpha=0.12, color='#E74C3C')
    if success_x:
        ax.scatter(success_x, success_d, color='#27AE60', zorder=5,
                   s=60, label='Exit reached')
    ax.axhline(0, color='#27AE60', linestyle='--', linewidth=1.2,
               label='Exit (distance = 0)')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Final BFS Distance to Exit', fontsize=12)
    ax.set_title('SAC Training — Distance to Exit per Episode', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_distance_curve.png', dpi=130, bbox_inches='tight')
    plt.close()
    print('Saved: training_distance_curve.png')


# =============================================================================
# SECTION 11 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-only', action='store_true',
        help='Skip training — load saved best model and run one test episode.'
    )
    args = parser.parse_args()

    set_global_seed(SEED)

    print()
    print('=' * 54)
    print('  SAC MAZE SOLVER  --  16x16')
    print(f'  Seed: {SEED}  |  Mode: {"TEST ONLY" if args.test_only else "TRAIN + TEST"}')
    print('=' * 54)
    print()

    model_path = 'models/sac_maze_v3'
    best_path  = 'models/sac_best'

    if args.test_only:
        if not os.path.exists(best_path + '.zip'):
            print(f'ERROR: no saved model found at {best_path}.zip')
            print('Run without --test-only first to train and save a model.')
            sys.exit(1)
        print(f'Loading best model from {best_path}.zip ...')
        training_history = []
        step_snapshots   = []
        used_model       = best_path
    else:
        print('Validating environment...')
        check_env(ContinuousMazeWrapper(), warn=True)
        print('  OK\n')

        model, training_history, step_snapshots, best_path = train(
            timesteps=150_000, save_path=model_path, record_every=1
        )
        used_model = best_path if os.path.exists(best_path + '.zip') else model_path
        print()


    # Test episode
    print()
    print(f'Running test episode -- model: {used_model}.zip')
    env_ep   = ContinuousMazeWrapper()
    model_ep = SAC.load(used_model, env=env_ep)
    log, path, success, total_reward = run_episode_logged(
        model_ep, env_ep, max_steps=1000, deterministic=True
    )
    print(f'  Steps: {len(log)}  |  Reward: {total_reward:.1f}  |  '
          f'Exit: {"YES *" if success else "NO"}  |  '
          f'Final dist: {log[-1]["Dist After"]}')
    print()

    # Output 1: training GIF
    if step_snapshots:
        print('Saving training_all_steps.gif ...')
        env_anim = ContinuousMazeWrapper()
        save_all_steps_gif(env_anim.env, step_snapshots,
                           filename='training_all_steps.gif', fps=15)
    else:
        print('(training_all_steps.gif skipped -- test-only mode)')

    # Output 2: episode GIF
    print('Saving episode_animation.gif ...')
    env_ep2   = ContinuousMazeWrapper()
    model_ep2 = SAC.load(used_model, env=env_ep2)
    save_episode_animation(env_ep2, model_ep2,
                           filename='episode_animation.gif',
                           max_steps=1000, fps=6)

    # Output 3 & 4: learning curves
    if training_history:
        print('Saving training curve PNGs ...')
        save_training_curves(training_history)
    else:
        print('(training curves skipped -- test-only mode)')

    print()
    print('=' * 54)
    print('  OUTPUT FILES')
    print('=' * 54)
    print('  maze_layout.png             -- maze + BFS optimal path')
    if step_snapshots:
        print('  training_all_steps.gif      -- all 30,000 training steps')
    print('  episode_animation.gif       -- deterministic test episode')
    if training_history:
        print('  training_reward_curve.png   -- reward vs training step')
        print('  training_distance_curve.png -- BFS distance vs training step')
    print(f'  Model used : {used_model}.zip')
    print(f'  Seed       : {SEED}')
    print()
    print('  To reproduce without retraining:')
    print('  python ddpg_maze_complete.py --test-only')
    print('=' * 54)


if __name__ == '__main__':
    main()
