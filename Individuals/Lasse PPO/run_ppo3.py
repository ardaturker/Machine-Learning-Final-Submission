from __future__ import annotations

import torch

import maze_3 as maze_file
from ppo import ACTIONS, ActorCritic, encode_state, extract_path, print_path, shortest_path_length, move


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

def print_step_by_step_path(maze, model):
    state = maze.start
    max_steps = maze.width * maze.height * 4

    print("PPO attempted path:")
    print(f"start={state}")

    for step in range(1, max_steps + 1):
        state_tensor = encode_state(maze, state)

        with torch.no_grad():
            logits, _value = model(state_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        action_index = torch.argmax(probs).item()
        action_name, _ = ACTIONS[action_index]

        next_state, reward, done = move(maze, state, action_index)

        print(
            f"step={step:03d} "
            f"state={state} "
            f"action={action_name:<5} "
            f"next={next_state} "
            f"reward={reward:>4} "
            f"done={done} "
            f"probs=["
            f"up={probs[0]:.2f}, "
            f"down={probs[1]:.2f}, "
            f"left={probs[2]:.2f}, "
            f"right={probs[3]:.2f}]"
        )

        if done:
            print("Reached exit.")
            break

        if next_state == state:
            print("Stopped because the chosen action hit a wall or boundary.")
            break

        state = next_state
    else:
        print("Stopped because max_steps was reached.")

def main():
    maze = build_maze()

    input_size = len(encode_state(maze, maze.start))
    action_size = len(ACTIONS)

    model = ActorCritic(input_size, action_size)
    model.load_state_dict(torch.load("ppo_maze_33.pt"))
    model.eval()

    print_step_by_step_path(maze, model)

    path = extract_path(maze, model)
    print()
    print(f"learned_path_length={len(path) - 1}")
    print(f"shortest_path_length={shortest_path_length(maze)}")
    print(f"reached_exit={path[-1] == maze.exit}")
    print()
    print_path(maze, path)



if __name__ == "__main__":
    main()
