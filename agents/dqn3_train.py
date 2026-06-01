from __future__ import annotations

import importlib
import os
import sys
import csv
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow GPU memory growth on DirectML / CUDA backends.
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)


# Hyper-parameters  (from config.py — edit there to tune)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DQN3, MAZE_ID

EPISODES        = DQN3.EPISODES
MAX_STEPS       = DQN3.MAX_STEPS

GAMMA           = DQN3.GAMMA
ALPHA           = DQN3.LEARNING_RATE

EPSILON_START   = DQN3.EPSILON_START
EPSILON_MIN     = DQN3.EPSILON_MIN
EPSILON_DECAY   = DQN3.EPSILON_DECAY

BUFFER_SIZE     = DQN3.BUFFER_SIZE
BATCH_SIZE      = DQN3.BATCH_SIZE
TRAIN_START     = DQN3.TRAIN_START

TAU                = DQN3.TAU
TARGET_UPDATE_FREQ = DQN3.TARGET_UPDATE_FREQ

GRAD_CLIP       = DQN3.GRAD_CLIP
TRAIN_FREQ      = DQN3.TRAIN_FREQ
REVISIT_PENALTY = DQN3.REVISIT_PENALTY

# Grid cell values in the maze obs: 0=floor, 1=wall, 2=player, 3=exit, 4=spike.
# Dividing by 4.0 maps all values to [0, 1].
OBS_NORM = 4.0



# Maze environment  —  resolved from MAZE_ID in config.py

_MAZE_INFO = {
    1: ("maze_1", "maze_env",   "16×16"),
    2: ("maze_2", "maze_2_env", "25×25"),
    3: ("maze_3", "maze_3_env", "35×35"),
}

if MAZE_ID not in _MAZE_INFO:
    raise ValueError(f"MAZE_ID must be 1, 2, or 3 — got {MAZE_ID!r}")

_maze_dir, _maze_module, MAZE_SIZE_LABEL = _MAZE_INFO[MAZE_ID]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", _maze_dir))
_env_mod    = importlib.import_module(_maze_module)
MazeEnv     = _env_mod.MazeEnv
OBS_SIZE    = _env_mod.OBS_SIZE
NUM_ACTIONS = _env_mod.NUM_ACTIONS


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "DQN3")
os.makedirs(RESULTS_DIR, exist_ok=True)

_SUFFIX = f"_maze{MAZE_ID}"


# ---------------------------------------------------------------------------
# 1. Q-network
# ---------------------------------------------------------------------------
def build_q_network() -> keras.Model:

    model = keras.Sequential([
        keras.layers.Input(shape=(OBS_SIZE,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(NUM_ACTIONS, activation="linear"),
    ], name="dqn3_network")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA))
    return model

# 2. Experience Replay Buffer

class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.

    Observations are stored already normalised.  Random mini-batch sampling
    breaks temporal correlations that would make gradient updates unstable if
    we trained step-by-step on the raw trajectory.
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# 3. DQN3 Agent — Double DQN + soft target updates

class DQN3Agent:


    def __init__(self) -> None:
        self.q_net      = build_q_network()
        self.target_net = build_q_network()
        self._hard_copy_target()

        self.buffer  = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

        self._huber  = keras.losses.Huber(delta=1.0)

    # Action selection — ε-greedy

    def select_action(self, norm_state: np.ndarray) -> int:
        """
        With probability ε: random action (explore).
        Otherwise: argmax Q_online(s, ·) (exploit).
        norm_state must already be divided by OBS_NORM.
        """
        if np.random.rand() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        q_values = self.q_net(norm_state[np.newaxis], training=False)  # (1, A)
        return int(tf.argmax(q_values[0]).numpy())

    # Double DQN Bellman update and gradient step

    def learn(self) -> float:

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # ── Step 1: online network selects the greedy action in s' ────
        next_q_online = self.q_net(next_states, training=False)         # (B, A)
        best_actions  = tf.argmax(next_q_online, axis=1)                # (B,)

        # ── Step 2: target network evaluates that specific action ─────
        next_q_target = self.target_net(next_states, training=False)    # (B, A)
        action_mask   = tf.one_hot(best_actions, NUM_ACTIONS)           # (B, A)
        max_next_q    = tf.reduce_sum(next_q_target * action_mask, axis=1)  # (B,)

        # ── Bellman target (stop gradient so it's treated as constant) ─
        targets = tf.stop_gradient(
            rewards + GAMMA * max_next_q * (1.0 - dones)
        )

        # ── Huber loss + clipped gradient step ────────────────────────
        with tf.GradientTape() as tape:
            all_q      = self.q_net(states, training=True)              # (B, A)
            taken_mask = tf.one_hot(actions, NUM_ACTIONS)               # (B, A)
            q_taken    = tf.reduce_sum(all_q * taken_mask, axis=1)     # (B,)
            loss       = self._huber(targets, q_taken)                  # scalar

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP)
        self.q_net.optimizer.apply_gradients(
            zip(grads, self.q_net.trainable_variables)
        )
        return float(loss.numpy())

    # Soft target update  (Polyak averaging)


    def soft_update_target(self) -> None:
        for online_var, target_var in zip(
            self.q_net.variables, self.target_net.variables
        ):
            target_var.assign(TAU * online_var + (1.0 - TAU) * target_var)

    def _hard_copy_target(self) -> None:
        """One-time hard copy at initialisation — both networks start identical."""
        self.target_net.set_weights(self.q_net.get_weights())

    # Epsilon decay

    def decay_epsilon(self) -> None:
        """Multiplicative decay per episode; floors at EPSILON_MIN."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# Helper — normalise raw environment observation

def _normalise(obs: np.ndarray) -> np.ndarray:
    """Divide raw grid encoding by 4.0 → [0, 1] range for stable training."""
    return obs / OBS_NORM


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train() -> tuple:
    env   = MazeEnv(max_steps=MAX_STEPS)
    agent = DQN3Agent()

    episode_rewards:  list[float] = []
    episode_steps:    list[int]   = []
    episode_losses:   list[float] = []
    episode_epsilons: list[float] = []

    csv_path    = os.path.join(RESULTS_DIR, f"training_DQN3{_SUFFIX}.csv")
    global_step = 0  # total env steps across all episodes (drives soft-update timing)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "solved", "avg_loss", "epsilon"])

        print(f"Training DQN3 (Double DQN, TensorFlow) on Maze {MAZE_ID} ({MAZE_SIZE_LABEL})")
        print(f"  Improvements over DQN: Double Q-targets | Soft updates (τ={TAU}) | "
              f"Huber loss | Grad clip ({GRAD_CLIP}) | Revisit penalty ({REVISIT_PENALTY})")
        print(f"{'Episode':>8}  {'Reward':>8}  {'Steps':>6}  {'Loss':>8}  {'Epsilon':>7}  {'Solved':>6}")
        print("-" * 60)

        for ep in range(1, EPISODES + 1):
            raw_obs, _ = env.reset()
            obs        = _normalise(raw_obs)

            total_reward = 0.0
            losses: list[float] = []
            solved = False

            # ── Per-episode visit counter for revisit penalty ─────────
            # First time at a cell → no penalty.
            # Each repeat visit → REVISIT_PENALTY × min(extra_visits, 5).
            # Cap at 5× so one badly-trapped corner doesn't swamp the signal.
            visit_counts: dict[tuple, int] = {}

            for _ in range(MAX_STEPS):
                action                          = agent.select_action(obs)
                raw_next, reward, terminated, truncated, _ = env.step(action)
                next_obs                        = _normalise(raw_next)
                global_step                    += 1

                # ── Revisit penalty ───────────────────────────────────
                pos               = env._player
                visit_counts[pos] = visit_counts.get(pos, 0) + 1
                extra             = visit_counts[pos] - 1          # 0 on first visit
                if extra > 0:
                    reward += REVISIT_PENALTY * min(extra, 5)

                done = terminated or truncated
                agent.buffer.push(obs, action, reward, next_obs, done)

                obs          = next_obs
                total_reward += reward

                # ── Gradient update ───────────────────────────────────
                if (len(agent.buffer) >= BATCH_SIZE
                        and ep > TRAIN_START
                        and global_step % TRAIN_FREQ == 0):
                    losses.append(agent.learn())

                # ── Soft target update ────────────────────────────────
                if global_step % TARGET_UPDATE_FREQ == 0:
                    agent.soft_update_target()

                if terminated:
                    solved = True
                if done:
                    break

            if ep > TRAIN_START:
                agent.decay_epsilon()

            avg_loss = float(np.mean(losses)) if losses else 0.0

            episode_rewards.append(total_reward)
            episode_steps.append(env._steps)
            episode_losses.append(avg_loss)
            episode_epsilons.append(agent.epsilon)

            writer.writerow([ep, f"{total_reward:.2f}", env._steps,
                             int(solved), f"{avg_loss:.4f}", f"{agent.epsilon:.4f}"])

            if ep % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                print(f"{ep:>8}  {avg_r:>8.2f}  {env._steps:>6}  "
                      f"{avg_loss:>8.4f}  {agent.epsilon:>7.4f}  {str(solved):>6}")

    print("\nTraining complete.")
    print(f"Log saved   → {csv_path}")

    model_path = os.path.join(RESULTS_DIR, f"dqn3_model{_SUFFIX}.keras")
    agent.q_net.save(model_path)
    print(f"Model saved → {model_path}")

    return agent, episode_rewards, episode_steps, episode_losses, episode_epsilons


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def _build_maze_data():
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


def _run_vis_episode(agent: DQN3Agent) -> list[dict]:
    """Run one fully-greedy episode (ε = 0) and record animation frames."""
    env    = MazeEnv(max_steps=MAX_STEPS)
    obs, _ = env.reset()
    obs    = _normalise(obs)
    path   = [env._player]

    saved_eps     = agent.epsilon
    agent.epsilon = 0.0   # no exploration during visualisation

    frames  = []
    _ANAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for _ in range(MAX_STEPS):
        prev_pos = env._player
        action   = agent.select_action(obs)
        raw_next, reward, terminated, truncated, _ = env.step(action)
        obs      = _normalise(raw_next)

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

    agent.epsilon = saved_eps
    return frames


def visualise(
    agent:            DQN3Agent,
    episode_rewards:  list,
    episode_steps:    list,
    episode_losses:   list,
    episode_epsilons: list,
) -> None:
    """Save maze layout PNG, episode animation GIF, and training curves."""
    from visualisation import (
        save_maze_layout,
        save_episode_animation,
        save_dqn3_training_plot,
    )

    print('\nGenerating visualisations...')
    maze = _build_maze_data()

    save_maze_layout(maze, RESULTS_DIR, _SUFFIX)

    frames   = _run_vis_episode(agent)
    gif_path = os.path.join(RESULTS_DIR, f'episode_animation{_SUFFIX}.gif')
    save_episode_animation(maze, frames, gif_path, fps=6)

    save_dqn3_training_plot(
        episode_rewards, episode_steps, episode_losses, episode_epsilons,
        MAZE_ID, MAZE_SIZE_LABEL, RESULTS_DIR, _SUFFIX,
    )
    print('Visualisation complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
    print()
    agent, ep_rewards, ep_steps, ep_losses, ep_epsilons = train()
    visualise(agent, ep_rewards, ep_steps, ep_losses, ep_epsilons)
