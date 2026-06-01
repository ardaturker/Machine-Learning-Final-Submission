"""
config.py — all the hyperparameters in one place

change MAZE_ID at the top to switch mazes, everything else
adjusts automatically. edit the dataclass fields if u want
to tune something.

from config import DQN3, PPO, SARSA, QLearning, SAC, MAZE_ID
"""

from __future__ import annotations
from dataclasses import dataclass


# ---- pick your maze here ----
# 1 = 16x16, simple, trains fast
# 2 = 25x25, medium
# 3 = 35x35, hard, has spikes
MAZE_ID = 2

# max steps scales with maze size (cells * 4 directions roughly)
# sarsa and ppo figure this out themselves, others use this
_MAX_STEPS = {1: 16 * 16 * 4, 2: 25 * 25 * 4, 3: 35 * 35 * 4}   # 1024 / 2500 / 4900
_MAX_STEPS_NOW = _MAX_STEPS[MAZE_ID]

# sac needs way more timesteps on bigger mazes to fill its buffer
_SAC_TIMESTEPS = {1: 150_000, 2: 400_000, 3: 750_000}


# shared reward values — used by most agents
MAZE_FINISHED = 50   # exit reward
WALL_PENALTY  = -1   # hit a wall
STEP_PENALTY  = -0.01  # cost per step


# ---- DQN3 (tensorflow, Double DQN + soft target updates) ----
@dataclass(frozen=True)
class _DQN3Config:
    # more episodes than DQN so the double-dqn convergence advantage is visible
    EPISODES:           int   = 1500
    MAX_STEPS:          int   = _MAX_STEPS_NOW

    # bellman
    GAMMA:              float = 0.995      # same as DQN — long paths need high discount

    # huber loss is less sensitive to big TD errors, so a slightly higher lr is safe
    LEARNING_RATE:      float = 0.0003

    # epsilon-greedy — starts fully random, decays to 5 % floor around ep 1100
    EPSILON_START:      float = 1.0
    EPSILON_MIN:        float = 0.05
    EPSILON_DECAY:      float = 0.997

    # replay buffer — bigger than DQN (100k vs 50k) for more diverse samples
    BUFFER_SIZE:        int   = 100_000
    BATCH_SIZE:         int   = 64
    TRAIN_START:        int   = 350         # warmup before first gradient step

    # soft target update (Polyak averaging): θ_t ← τ·θ + (1-τ)·θ_t
    # smoother than DQN's hard copy every C episodes
    TAU:                float = 0.005
    TARGET_UPDATE_FREQ: int   = 4          # soft update every N env steps

    # gradient clipping — prevents exploding gradients during early exploration
    GRAD_CLIP:          float = 10.0

    # learn every N env steps (same frequency as DQN's every-4-steps schedule)
    TRAIN_FREQ:         int   = 4

    # revisit penalty added on top of the environment reward
    # discourages loop behaviour even after epsilon has mostly decayed
    REVISIT_PENALTY:    float = -0.25

DQN3 = _DQN3Config()


# ---- PPO (lasse's masked multi-task ppo, pytorch) ----
@dataclass(frozen=True)
class _PPOConfig:
    # behaviour cloning - runs before ppo to give it a head start from bfs paths
    BC_EPOCHS:            int   = 300
    BC_BATCH_SIZE:        int   = 512
    BC_LEARNING_RATE:     float = 1e-3

    # ppo updates
    UPDATES:              int   = 160
    ROLLOUT_STEPS:        int   = 512   # steps collected before each update
    UPDATE_EPOCHS:        int   = 3     # how many passes over rollout data
    MINIBATCH_SIZE:       int   = 1024

    # network
    HIDDEN_SIZE:          int   = 192
    TASK_EMBEDDING_SIZE:  int   = 16    # for multi-maze, one embedding per maze

    # discount + advantage
    GAMMA:                float = 0.985
    GAE_LAMBDA:           float = 0.95  # standard gae value

    # clipping - the "proximal" part
    CLIP_EPSILON:         float = 0.18

    LEARNING_RATE:        float = 2.5e-4

    # loss coefficients
    ENTROPY_START:        float = 0.025  # anneals down during training
    ENTROPY_END:          float = 0.004
    VALUE_COEF:           float = 0.5
    MAX_GRAD_NORM:        float = 0.75
    TARGET_KL:            float = 0.04   # stop early if kl gets too big

    # reward shaping - ppo uses its own, not the shared ones above
    # (so dont compare raw ppo rewards directly to dqn/sarsa)
    FINISH_REWARD:        float = 20.0
    STEP_PENALTY:         float = -0.03
    PROGRESS_REWARD:      float = 0.20   # bonus per bfs step closer
    REPEAT_PENALTY:       float = -0.015 # visiting same cell again
    TIMEOUT_PENALTY:      float = -5.0

    # biases action logits toward progress at the start
    DISTANCE_PRIOR_SCALE: float = 3.0

    # eval + early stopping
    EVAL_EVERY:           int   = 10
    EARLY_STOP_PATIENCE:  int   = 3
    SEED:                 int   = 7

PPO = _PPOConfig()


# ---- SARSA (tabular, on-policy) ----
@dataclass(frozen=True)
class _SARSAConfig:
    EPISODES:      int   = 5000  # needs more episodes than neural nets

    # td update
    ALPHA:         float = 0.2   # learning rate
    GAMMA:         float = 0.99  # discount

    # epsilon greedy
    EPSILON:       float = 1.0
    EPSILON_DECAY: float = 0.999
    MIN_EPSILON:   float = 0.05

SARSA = _SARSAConfig()


# ---- Q-Learning (tabular, off-policy, not really used but kept) ----
@dataclass(frozen=True)
class _QLearningConfig:
    EPISODES:      int   = 5000

    ALPHA:         float = 0.1   # lr
    GAMMA:         float = 0.95

    EPSILON:       float = 1.0
    EPSILON_DECAY: float = 0.995
    MIN_EPSILON:   float = 0.05

QLearning = _QLearningConfig()


# ---- SAC (stable-baselines3, continuous wrapper) ----
@dataclass(frozen=True)
class _SACConfig:
    # timesteps + max steps get overridden below based on MAZE_ID
    TIMESTEPS:         int   = _SAC_TIMESTEPS[MAZE_ID]
    MAX_STEPS:         int   = _MAX_STEPS_NOW

    GAMMA:             float = 0.99

    LEARNING_RATE:     float = 3e-4    # same lr for actor critic and alpha

    # entropy temp - auto tuned during training
    ALPHA:             float = 0.2
    AUTO_TUNE_ALPHA:   bool  = True
    TARGET_ENTROPY:    float = -1.0

    # replay buffer
    BUFFER_SIZE:       int   = 200_000  # needs to be big
    BATCH_SIZE:        int   = 256      # bigger than dqn bc 3 networks
    TRAIN_START:       int   = 1_000   # random steps before any learning

    # soft target update (different from dqn's hard sync)
    TAU:               float = 0.005   # theta_target = tau*theta + (1-tau)*theta_target
    TARGET_UPDATE_INTERVAL: int = 1

    HIDDEN_SIZE:       int   = 256     # 2 hidden layers

    RECORD_EVERY:      int   = 1       # eval every N episodes

    # reward shaping - sac has the most shaping terms
    STEP_PENALTY:      float = -0.1
    WALL_PENALTY:      float = -3.0
    CLOSER_REWARD:     float =  1.0    # bfs dist went down
    FARTHER_PENALTY:   float = -0.3
    EXPLORE_BONUS:     float =  0.5    # first time visiting a cell
    REVISIT_BASE:      float =  1.0    # penalty per revisit, multiplied by visit count
    REVISIT_CAP:       float = 20.0    # dont let revisit penalty go crazy
    PROX_THRESHOLD:    int   =  5      # extra bonus when bfs dist is <= this
    EXIT_REWARD:       float = 200.0   # big reward so it dominates entropy bonus
    SPIKE_PENALTY:     float = -10.0   # maze 3 only

SAC = _SACConfig()
