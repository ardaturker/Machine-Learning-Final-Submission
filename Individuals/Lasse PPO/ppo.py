from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import maze_1 as maze_file

ACTIONS = [
    ("up", (0, -1)),
    ("down", (0, 1)),
    ("left", (-1, 0)),
    ("right", (1, 0)),
]

MAZE_FINISHED = 100

EPISODES = 400
ROLLOUT_STEPS = 2048     #Amount of game steps to collect before update
GAMMA = 0.95            #How much future awards matter
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2      #Prevent large policy updates (PPO's incremntal learning)
LEARNING_RATE = 3e-4
UPDATE_EPOCHS = 6
MINIBATCH_SIZE = 128
ENTROPY_COEF = 0.1     #Encourages exploration
VALUE_COEF = 0.5


# Builds Maze by getting dimenstions from maze.py. Uses build_walls function from maze.py1
def build_maze():
    width = maze_file.DEFAULT_WIDTH
    height = maze_file.DEFAULT_HEIGHT
    exit_cell = maze_file.resolve_exit(width, height)

    return maze_file.Maze(
        width=width,
        height=height,
        start=maze_file.START,
        exit=exit_cell,
        walls=maze_file.build_walls(width, height, maze_file.START, exit_cell),
    )

def move(maze, state, action_index):
    _name, (dx, dy) = ACTIONS[action_index]
    x, y = state
    next_state = (x + dx, y + dy)

    if not maze.is_open(next_state):
        return state, -5, False     # hit wall -> stay in same state -> reward -5 -> episode not done

    if next_state == maze.exit:
        return next_state, MAZE_FINISHED, True      # reached exit -> reward 100 -> episode done

    return next_state, -1, False    # valid normal step -> move to next state -> reward -1 -> episode not done

# is the devision by maze size necessary or maybe a codex complication ?
def encode_state(maze, state):
    x, y = state
    ex, ey = maze.exit

    wall_up = 0 if maze.is_open((x, y - 1)) else 1
    wall_down = 0 if maze.is_open((x, y + 1)) else 1
    wall_left = 0 if maze.is_open((x - 1, y)) else 1
    wall_right = 0 if maze.is_open((x + 1, y)) else 1

    return torch.tensor(
        [
            x / max(1, maze.width - 1),
            y / max(1, maze.height - 1),
            (ex - x) / max(1, maze.width - 1),
            (ey - y) / max(1, maze.height - 1),
            wall_up,
            wall_down,
            wall_left,
            wall_right,
        ],
        dtype=torch.float32,
    )


#
class ActorCritic(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, states):
        hidden = self.shared(states)
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values

    def act(self, state_tensor):
        logits, value = self.forward(state_tensor.unsqueeze(0))
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action).squeeze(0), value.squeeze(0)
    
def collect_rollout(maze, model, max_steps):
    rollout = Rollout([], [], [], [], [], [])
    episode_rewards = []
    episode_wins = []
    visits = {}

    state = maze.start
    episode_reward = 0
    episode_step = 0

    while len(rollout.states) < ROLLOUT_STEPS:
        state_tensor = encode_state(maze, state)
        visits[state] = visits.get(state, 0) + 1

        with torch.no_grad():
            action, log_prob, value = model.act(state_tensor)

        next_state, reward, done = move(maze, state, action)

        rollout.states.append(state_tensor)
        rollout.actions.append(action)
        rollout.rewards.append(reward)
        rollout.dones.append(done)
        rollout.log_probs.append(log_prob)
        rollout.values.append(value)

        state = next_state
        episode_reward += reward
        episode_step += 1

        timed_out = episode_step >= max_steps
        if timed_out and not done:
            rollout.rewards[-1] += -50
            episode_reward += -50

        if done or timed_out: 
            episode_rewards.append(episode_reward)
            episode_wins.append(done)

            state = maze.start
            episode_reward = 0
            episode_step = 0

    return rollout, episode_rewards, episode_wins, visits

def compute_advantages(rollout):
    rewards = rollout.rewards
    dones = rollout.dones
    values = torch.stack(rollout.values)

    advantages = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0

        delta = rewards[step] + GAMMA * next_value * mask - values[step].item()
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae

        advantages.insert(0, gae)
        next_value = values[step].item()

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values.detach()

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns



def print_path(maze, path):
    path_cells = set(path)

    for y in range(maze.height):
        row = ""
        for x in range(maze.width):
            cell = (x, y)

            if cell == maze.start:
                row += "S "
            elif cell == maze.exit:
                row += "E "
            elif cell in maze.walls:
                row += "##"
            elif cell in path_cells:
                row += ".."
            else:
                row += "  "

        print(row)

def print_heatmap(maze, visits):
    max_visit = max(visits.values(), default=1)

    for y in range(maze.height):
        row = ""

        for x in range(maze.width):
            cell = (x, y)

            if cell == maze.start:
                row += "S "
            elif cell == maze.exit:
                row += "E "
            elif cell in maze.walls:
                row += "##"
            else:
                count = visits.get(cell, 0)

                if count == 0:
                    row += "  "
                elif count < max_visit * 0.25:
                    row += ".."
                elif count < max_visit * 0.50:
                    row += "::"
                elif count < max_visit * 0.75:
                    row += "**"
                else:
                    row += "@@"

        print(row)


def shortest_path_length(maze):
    queue = deque([maze.start])
    distances = {maze.start: 0}

    while queue:
        state = queue.popleft()

        if state == maze.exit:
            return distances[state]
        
        for action_index in range(len(ACTIONS)):
            next_state, _reward, _done = move(maze, state, action_index)

            if next_state != state and next_state not in distances:
                distances[next_state] = distances[state] + 1
                queue.append(next_state)
        
    return None

def update_model(model, optimizer, rollout):
    states = torch.stack(rollout.states)
    actions = torch.tensor(rollout.actions, dtype=torch.long)
    old_log_probs = torch.stack(rollout.log_probs).detach()

    advantages, returns = compute_advantages(rollout)

    total_steps = len(states)
    indices = list(range(total_steps))

    for _ in range(UPDATE_EPOCHS):
        random.shuffle(indices)

        for start in range(0, total_steps, MINIBATCH_SIZE):
            batch_indices = indices[start : start + MINIBATCH_SIZE]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            logits, values = model(batch_states)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON,
            ) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (batch_returns - values).pow(2).mean()

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(maze):
    input_size = len(encode_state(maze, maze.start))
    action_size = len(ACTIONS)

    model = ActorCritic(input_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    max_steps = maze.width * maze.height * 4
    recent_wins = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)

    all_visits = {}
    total_env_steps = 0

    for episode in range(1, EPISODES + 1):
        rollout, rewards, wins, visits = collect_rollout(maze, model, max_steps)
        total_env_steps += len(rollout.states)

        for cell, count in visits.items():
            all_visits[cell] = all_visits.get(cell, 0) + count

        update_model(model, optimizer, rollout)

        recent_rewards.extend(rewards)
        recent_wins.extend(wins)

        if episode % 50 == 0:
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            success_rate = sum(recent_wins) / max(1, len(recent_wins))

            print(
                f"episode={episode} "
                f"env_steps={total_env_steps} "
                f"avg_reward={avg_reward:.1f} "
                f"recent_success={success_rate:.0%}"
            )

    print(f"total_env_steps={total_env_steps}")
    return model, all_visits

def choose_greedy_action(model, maze, state):
    state_tensor = encode_state(maze, state)

    with torch.no_grad():
        logits, _value = model(state_tensor.unsqueeze(0))

    return torch.argmax(logits, dim=-1).item()


def extract_path(maze, model):
    state = maze.start
    path = [state]
    max_steps = maze.width * maze.height * 4

    for _ in range(max_steps):
        action = choose_greedy_action(model, maze, state)
        next_state, _reward, done = move(maze, state, action)

        if next_state == state:
            break

        path.append(next_state)
        state = next_state

        if done:
            break

    return path


@dataclass
class Rollout:
    states: list
    actions: list
    rewards: list
    dones: list
    log_probs: list
    values: list

def main():
    random.seed(1)
    torch.manual_seed(1)

    maze = build_maze()
    model, all_visits = train(maze)
    torch.save(model.state_dict(), "ppo_maze_1.pt")

    path = extract_path(maze, model)

    print()
    print(f"learned_path_length={len(path) - 1}")
    print(f"shortest_path_length={shortest_path_length(maze)}")
    print(f"reached_exit={path[-1] == maze.exit}")

    print_heatmap(maze, all_visits)
    print()
    print_path(maze, path)


if __name__ == "__main__":
    main()
