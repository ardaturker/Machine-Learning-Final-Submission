from __future__ import annotations

import importlib
import os
import sys
import csv
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Config & maze selection  (same pattern as dqn_train.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import SAC as SAC_CFG, MAZE_ID

_MAZE_INFO = {
    1: ('maze_1', 'maze_env',   '16x16'),
    2: ('maze_2', 'maze_2_env', '25x25'),
    3: ('maze_3', 'maze_3_env', '35x35'),
}
if MAZE_ID not in _MAZE_INFO:
    raise ValueError(f'MAZE_ID must be 1, 2, or 3 — got {MAZE_ID!r}')

_maze_dir, _maze_module, MAZE_SIZE_LABEL = _MAZE_INFO[MAZE_ID]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'maps', _maze_dir))
_env_mod    = importlib.import_module(_maze_module)
MazeEnv     = _env_mod.MazeEnv
OBS_SIZE    = _env_mod.OBS_SIZE
NUM_ACTIONS = _env_mod.NUM_ACTIONS

SEED    = 42
_SUFFIX = f'_maze{MAZE_ID}'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Results', 'SAC')
os.makedirs(RESULTS_DIR, exist_ok=True)

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


# =============================================================================
# SECTION 1 — SAC WRAPPER  (4D argmax action space)
# =============================================================================

class ContinuousMazeWrapper(gym.Env):
    """
    Thin gym.Env wrapper around the shared MazeEnv for use with SAC.

    The actor network outputs 4 scores [s_up, s_down, s_left, s_right].
    The discrete action is argmax of those scores.

    Observation and rewards are unchanged from the shared MazeEnv so that
    all agents operate in the same environment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.env = MazeEnv(max_steps=SAC_CFG.MAX_STEPS)

        # high=4 covers maze-3's spike value (0=floor,1=wall,2=player,3=exit,4=spike)
        self.observation_space = spaces.Box(
            low=0.0, high=4.0,
            shape=(OBS_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.full(NUM_ACTIONS, -1.0, dtype=np.float32),
            high=np.full(NUM_ACTIONS,  1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        scores   = np.asarray(action).reshape(-1)
        discrete = int(np.argmax(scores))

        obs, reward, terminated, truncated, info = self.env.step(discrete)

        # exit gives the only positive terminal reward across all maze variants
        info['reached_exit']    = bool(terminated and reward > 0)
        info['action_scores']   = scores.tolist()
        info['discrete_action'] = discrete
        info['manhattan_dist']  = self.env._manhattan()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# SECTION 2 — TRAINING CALLBACK
# =============================================================================

class TrainingRecorderCallback(BaseCallback):
    """
    Runs a greedy evaluation episode at the end of every SAC_CFG.RECORD_EVERY
    training episodes and stores reward, steps, and distance stats.
    Saves a checkpoint whenever eval reward improves.
    """

    def __init__(self, best_model_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.best_model_path   = best_model_path
        self.episode_count     = 0
        self.training_history: list[dict] = []
        self._ep_reward        = 0.0
        self.best_reward       = -float('inf')

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals['rewards'][0])

        if self.locals['dones'][0]:
            self.episode_count += 1
            ep_reward       = self._ep_reward
            self._ep_reward = 0.0

            if self.episode_count % SAC_CFG.RECORD_EVERY == 0:
                rec                 = self._eval_greedy()
                rec['episode']      = self.episode_count
                rec['timestep']     = self.num_timesteps
                rec['train_reward'] = ep_reward
                self.training_history.append(rec)

                is_best = rec['total_reward'] > self.best_reward
                if is_best:
                    self.best_reward = rec['total_reward']
                    self.model.save(self.best_model_path)

                print(
                    f'  [ep {self.episode_count:4d} | t={self.num_timesteps:7,}]  '
                    f'reward={rec["total_reward"]:+8.1f}  '
                    f'steps={rec["steps"]:4d}  '
                    f'dist={rec["final_dist"]:4d}  '
                    f'{"* BEST " if is_best else "       "}'
                    f'{"EXIT **" if rec["success"] else ""}'
                )
        return True

    def _eval_greedy(self) -> dict:
        """One noiseless episode with the current policy."""
        eval_env     = ContinuousMazeWrapper()
        obs, _       = eval_env.reset()
        total_reward = 0.0
        final_dist   = eval_env.env._manhattan()
        min_dist     = final_dist
        steps        = 0
        success      = False

        for _ in range(SAC_CFG.MAX_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = eval_env.step(action)
            total_reward += reward
            final_dist    = info['manhattan_dist']
            min_dist      = min(min_dist, final_dist)
            steps        += 1
            if info['reached_exit']:
                success = True
            if term or trunc:
                break

        return {
            'total_reward': total_reward,
            'steps':        steps,
            'final_dist':   final_dist,
            'min_dist':     min_dist,
            'success':      success,
        }


# =============================================================================
# SECTION 3 — TRAINING
# =============================================================================

def train() -> tuple:
    """
    Train SAC on the selected maze.
    Hyperparameters are read from config.py (SAC section).
    Returns (model, training_history, best_model_path).
    """
    train_env = Monitor(ContinuousMazeWrapper())

    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs          = dict(net_arch=[SAC_CFG.HIDDEN_SIZE, SAC_CFG.HIDDEN_SIZE]),
        learning_rate          = SAC_CFG.LEARNING_RATE,
        buffer_size            = SAC_CFG.BUFFER_SIZE,
        batch_size             = SAC_CFG.BATCH_SIZE,
        gamma                  = SAC_CFG.GAMMA,
        tau                    = SAC_CFG.TAU,
        learning_starts        = SAC_CFG.TRAIN_START,
        ent_coef               = 'auto' if SAC_CFG.AUTO_TUNE_ALPHA else SAC_CFG.ALPHA,
        target_update_interval = SAC_CFG.TARGET_UPDATE_INTERVAL,
        seed                   = SEED,
        verbose                = 0,
    )

    best_path = os.path.join(RESULTS_DIR, f'sac_best{_SUFFIX}')
    callback  = TrainingRecorderCallback(best_model_path=best_path)

    print(f'Training SAC on Maze {MAZE_ID} ({MAZE_SIZE_LABEL}) '
          f'for {SAC_CFG.TIMESTEPS:,} timesteps...')
    print('(* BEST = new checkpoint saved)')
    print(f'{"Episode":>9}  {"Timestep":>9}  {"Reward":>9}  '
          f'{"Steps":>6}  {"Dist":>5}  {"Best":>6}  {"Exit":>4}')
    print('-' * 65)

    model.learn(total_timesteps=SAC_CFG.TIMESTEPS, log_interval=None, callback=callback)

    final_path = os.path.join(RESULTS_DIR, f'sac_model{_SUFFIX}')
    model.save(final_path)
    print(f'\nFinal model : {final_path}.zip')
    print(f'Best model  : {best_path}.zip  (reward={callback.best_reward:.1f})')
    print(f'Episodes    : {callback.episode_count}  |  '
          f'Eval records: {len(callback.training_history)}')
    return model, callback.training_history, best_path


# =============================================================================
# SECTION 4 — EPISODE RUNNER
# =============================================================================

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


def run_episode_logged(model, env: ContinuousMazeWrapper,
                       deterministic: bool = True) -> tuple:
    """Run one episode and return (log, success, total_reward)."""
    obs, _ = env.reset()
    total_reward = 0.0
    log     = []

    for step_n in range(SAC_CFG.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        scores   = info['action_scores']
        disc_a   = info['discrete_action']
        sorted_s = sorted(scores, reverse=True)
        margin   = sorted_s[0] - sorted_s[1]
        confidence = 'HIGH' if margin > 0.5 else ('MED' if margin > 0.2 else 'LOW')

        log.append({
            'Step':       step_n + 1,
            'Action':     ACTION_NAMES[disc_a],
            'Confidence': confidence,
            'Dist':       info['manhattan_dist'],
            'Reward':     round(reward, 3),
        })

        if terminated or truncated:
            break

    success = info.get('reached_exit', False)
    return log, success, total_reward


# =============================================================================
# SECTION 5 — TRAINING CSV + PLOT
# =============================================================================

def save_training_csv(training_history: list) -> None:
    if not training_history:
        return
    csv_path = os.path.join(RESULTS_DIR, f'training_SAC{_SUFFIX}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps', 'solved',
                         'timestep', 'final_dist', 'min_dist', 'train_reward'])
        for h in training_history:
            writer.writerow([
                h['episode'],
                f"{h['total_reward']:.2f}",
                h['steps'],
                int(h['success']),
                h['timestep'],
                h['final_dist'],
                h['min_dist'],
                f"{h['train_reward']:.2f}",
            ])
    print(f'Log saved  → {csv_path}')


# =============================================================================
# SECTION 6 — MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true',
                        help='Skip training — load saved best model and run test episode.')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    print()
    print('=' * 60)
    print(f'  SAC MAZE SOLVER  --  Maze {MAZE_ID} ({MAZE_SIZE_LABEL})')
    print(f'  Seed: {SEED}  |  Mode: {"TEST ONLY" if args.test_only else "TRAIN + TEST"}')
    print('=' * 60)
    print()

    best_path  = os.path.join(RESULTS_DIR, f'sac_best{_SUFFIX}')
    final_path = os.path.join(RESULTS_DIR, f'sac_model{_SUFFIX}')

    if args.test_only:
        if not os.path.exists(best_path + '.zip'):
            print(f'ERROR: no saved model at {best_path}.zip')
            print('Run without --test-only first.')
            sys.exit(1)
        print(f'Loading best model from {best_path}.zip ...')
        training_history = []
        used_model       = best_path
    else:
        print('Validating environment...')
        check_env(ContinuousMazeWrapper(), warn=True)
        print('  OK\n')

        _, training_history, best_path = train()
        used_model = best_path if os.path.exists(best_path + '.zip') else final_path
        print()

    # Test episode
    print(f'Running test episode — model: {used_model}.zip')
    test_env   = ContinuousMazeWrapper()
    test_model = SAC.load(used_model, env=test_env)
    log, success, total_reward = run_episode_logged(test_model, test_env)
    print(f'  Steps: {len(log)}  |  Reward: {total_reward:.1f}  |  '
          f'Exit: {"YES *" if success else "NO"}  |  '
          f'Final dist: {log[-1]["Dist"]}')
    print()

    # CSV
    if training_history:
        print('Saving training CSV...')
        save_training_csv(training_history)
    else:
        print('(CSV skipped — test-only mode)')

    visualise(test_model, training_history)

    print()
    print('=' * 60)
    print('  OUTPUT FILES  (in Results/SAC/)')
    print('=' * 60)
    if training_history:
        print(f'  training_SAC{_SUFFIX}.csv')
        print(f'  training_plot_SAC{_SUFFIX}.png')
    print(f'  maze_layout{_SUFFIX}.png')
    print(f'  episode_animation{_SUFFIX}.gif')
    print(f'  sac_model{_SUFFIX}.zip')
    print(f'  sac_best{_SUFFIX}.zip')
    print(f'  Seed: {SEED}')
    print()
    print('  To reproduce without retraining:')
    print('  python sac_train.py --test-only')
    print('=' * 60)


# =============================================================================
# SECTION 7 — VISUALISATION
# =============================================================================

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


def _run_vis_episode(model, env: ContinuousMazeWrapper) -> list[dict]:
    """Run one deterministic episode and collect per-step animation frames."""
    obs, _ = env.reset()
    path   = [env.env._player]
    frames = []
    _ANAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for step_n in range(SAC_CFG.MAX_STEPS):
        prev_pos  = env.env._player
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        moved = env.env._player != prev_pos
        path.append(env.env._player)

        frames.append({
            'path':   list(path),
            'agent':  env.env._player,
            'step':   step_n + 1,
            'dist':   info['manhattan_dist'],
            'reward': reward,
            'action': _ANAMES.get(info['discrete_action'], '?'),
            'moved':  moved,
            'exit':   info['reached_exit'],
        })
        if terminated or truncated:
            break

    return frames


def visualise(model, training_history: list) -> None:
    """Generate maze layout PNG, episode animation GIF, and training curves."""
    from visualisation import (
        save_maze_layout,
        save_episode_animation,
        save_sac_training_plot,
    )

    print('\nGenerating visualisations...')
    maze = _build_maze_data()

    save_maze_layout(maze, RESULTS_DIR, _SUFFIX)

    vis_env  = ContinuousMazeWrapper()
    frames   = _run_vis_episode(model, vis_env)
    gif_path = os.path.join(RESULTS_DIR, f'episode_animation{_SUFFIX}.gif')
    save_episode_animation(maze, frames, gif_path, fps=6)

    if training_history:
        save_sac_training_plot(
            training_history, MAZE_ID, MAZE_SIZE_LABEL, RESULTS_DIR, _SUFFIX,
        )
    print('Visualisation complete.')


if __name__ == '__main__':
    main()
